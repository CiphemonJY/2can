"""Reproduce 2Can's calibration and verify thresholds from public labels (LLM-AggreFact).

  1) score:  python scripts/fit_calibration.py score dev  --out dev.jsonl   (and again for test)
     Sends a per-dataset-capped sample through a running 2Can server's /v1/verify.
     Needs `datasets` + access to lytang/LLM-AggreFact (accept its terms on the Hub first).
  2) fit:    python scripts/fit_calibration.py fit dev.jsonl test.jsonl [--policy two_can/policy.json]
     Fits the logistic calibrator and the pass/fail thresholds on dev, reports held-out test,
     and prints the policy fields to paste (it never overwrites the policy by itself).

The score step records the source document's sha1 (not its text) so thresholds can be
capped per document. Requires numpy and scipy.
"""
import argparse, hashlib, json, os, sys, time, urllib.request, collections
import numpy as np

EPS = 1e-4
GRID_P = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
GRID_F = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.01]
TARGET, CAP = 0.10, 3


# ---------------------------------------------------------------- score
def score(a):
    import pandas as pd
    if a.parquet:
        d = pd.read_parquet(a.parquet)
    else:
        from datasets import load_dataset
        d = load_dataset("lytang/LLM-AggreFact", split=a.split).to_pandas()
    d = d.reset_index().rename(columns={"index": "row"})
    d = pd.concat([g.sample(min(a.cap, len(g)), random_state=0) for _, g in d.groupby("dataset")])
    hdr = {"Content-Type": "application/json"}
    if a.token_file:
        hdr["Authorization"] = "Bearer " + open(a.token_file).read().strip()
    done = {json.loads(l)["row"] for l in open(a.out)} if os.path.exists(a.out) else set()
    t0, n = time.time(), 0
    with open(a.out, "a") as f:
        for doc, g in d[~d["row"].isin(done)].groupby("doc", sort=False):
            rows = list(g.itertuples())
            for k in range(0, len(rows), 32):
                part = rows[k:k + 32]
                req = urllib.request.Request(a.url.rstrip("/") + "/v1/verify", method="POST", headers=hdr,
                                             data=json.dumps({"source": doc, "claims": [r.claim for r in part]}).encode())
                res = json.load(urllib.request.urlopen(req, timeout=600))["results"]
                for r, x in zip(part, res):
                    f.write(json.dumps({"row": int(r.row), "dataset": r.dataset, "y": int(r.label), "src": hashlib.sha1(doc.encode()).hexdigest(),
                                        "pm": x.get("p_modernbert"), "pc": x.get("p_minicheck"), "pf": x.get("p_factcg"),
                                        "noul_reason": x.get("noul_reason")}) + "\n")
                n += len(part)
            f.flush()
    print("scored", n, "rows in", round(time.time() - t0), "s ->", a.out)


# ---------------------------------------------------------------- fit
def cp_upper(k, n, alpha=0.05):
    from scipy.stats import beta
    return 1.0 if n == 0 or k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def decide(it, P, F):
    if it["pm"] is None:
        return None
    if it["pm"] >= P and it["pc"] >= P:
        return 1
    if it["pm"] <= F and it["pc"] <= F:
        return 0
    return None


def evaluate(items, P, F):
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    dec = fa = fr = npass = nfail = 0
    for it in items:
        d = decide(it, P, F)
        if d is None:
            continue
        dec += 1
        g = per[it["src"]]
        if d == 1:
            npass += 1; g[0] += 1
            if it["y"] == 0:
                fa += 1; g[1] += 1
        else:
            nfail += 1; g[2] += 1
            if it["y"] == 1:
                fr += 1; g[3] += 1
    cp, ce, cf, cr = (sum(min(CAP, g[i]) for g in per.values()) for i in range(4))
    return {"P": P, "F": F, "coverage": dec / max(1, len(items)), "n_pass": npass, "false_accepts": fa, "n_fail": nfail, "false_rejects": fr,
            "fa_ucb95": cp_upper(ce, cp) if cp else 1.0, "fr_ucb95": cp_upper(cr, cf) if cf else 1.0, "capped_pass": cp, "capped_fail": cf}


def fit_thresholds(items):
    """Fixed-sequence search: loosen P (then F) only while the capped 95% upper bound stays <= TARGET."""
    bestP, bestF = 1.01, -0.01
    for P in sorted(GRID_P, reverse=True):
        e = evaluate(items, P, -1.0)
        if cp_upper(0, e["capped_pass"]) > TARGET:
            continue
        if e["fa_ucb95"] > TARGET:
            break
        bestP = P
    for F in sorted(GRID_F):
        e = evaluate(items, 2.0, F)
        if cp_upper(0, e["capped_fail"]) > TARGET:
            continue
        if e["fr_ucb95"] > TARGET:
            break
        bestF = F
    return bestP, bestF


lg = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
sig = lambda z: 1 / (1 + np.exp(-z))


def fit_logistic(X, y, l2=1e-3, iters=50):
    X1 = np.column_stack([X, np.ones(len(X))]); w = np.zeros(X1.shape[1])
    for _ in range(iters):  # Newton / IRLS
        p = sig(X1 @ w)
        g = X1.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0])
        w -= np.linalg.solve(H, g)
    return w


def ece(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(nb) if (b == k).any())


def auroc(y, s):
    o = s.argsort(); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    npos = y.sum(); return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def fit(a):
    dev = [json.loads(l) for l in open(a.dev)]
    test = [json.loads(l) for l in open(a.test)]
    ds, ts = [r for r in dev if r["pm"] is not None], [r for r in test if r["pm"] is not None]
    keys = ["pm", "pc"] + (["pf"] if all(r.get("pf") is not None for r in ds + ts) else [])   # FactCG input when scored
    X = lambda R: np.column_stack([lg(np.array([r[k] for r in R])) for k in keys])
    w = fit_logistic(X(ds), np.array([r["y"] for r in ds]))
    names = {"pm": "w_mb", "pc": "w_mc", "pf": "w_fc"}
    print("calibration  " + "  ".join("%s %.6f" % (names[k], w[j]) for j, k in enumerate(keys)) + "  bias %.6f  (fit n=%d)" % (w[-1], len(ds)))
    for name, R in (("dev", ds), ("test", ts)):
        y = np.array([r["y"] for r in R]); pm = np.array([r["pm"] for r in R]); pc = np.array([r["pc"] for r in R])
        pcal = sig(np.column_stack([X(R), np.ones(len(R))]) @ w)
        print(f"  {name}: n={len(R)}  ECE raw-mean {ece((pm + pc) / 2, y):.4f} -> calibrated {ece(pcal, y):.4f}   AUROC {auroc(y, pcal):.4f}")
    P, F = fit_thresholds(dev)
    print(f"thresholds  pass_min {P}  fail_max {F}")
    for name, R in (("dev", dev), ("test", test)):
        e = evaluate(R, P, F)
        print(f"  {name}: coverage {e['coverage']:.3f}  false-accept ucb95 {e['fa_ucb95']:.3f}  false-reject ucb95 {e['fr_ucb95']:.3f}")
    if a.policy:
        cur = json.load(open(a.policy))["verify"]
        e = evaluate(test, cur["pass_min"], cur["fail_max"])
        print(f"  current policy {cur['pass_min']}/{cur['fail_max']} on test: coverage {e['coverage']:.3f}  fa_ucb95 {e['fa_ucb95']:.3f}  fr_ucb95 {e['fr_ucb95']:.3f}")
    cal = {names[k]: round(float(w[j]), 6) for j, k in enumerate(keys)}
    cal.update({"bias": round(float(w[-1]), 6), "clip": EPS})
    print(json.dumps({"pass_min": P, "fail_max": F, "calibration": cal}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("split", choices=["dev", "test"]); s.add_argument("--out", required=True)
    s.add_argument("--cap", type=int, default=400); s.add_argument("--url", default="http://127.0.0.1:8766"); s.add_argument("--token-file")
    s.add_argument("--parquet", help="local copy of the split instead of downloading it")
    f = sub.add_parser("fit"); f.add_argument("dev"); f.add_argument("test"); f.add_argument("--policy")
    a = ap.parse_args()
    score(a) if a.cmd == "score" else fit(a)
