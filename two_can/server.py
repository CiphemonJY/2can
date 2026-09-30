"""2Can: a small, non-generative claim checker with calibrated abstention.

Two independent encoders (an NLI model and a fact-checking model) must agree before a claim is
called SUPPORTED or UNSUPPORTED; anything else is NOUL ("no usable answer"). Also offers zero-shot
choose/score over a third encoder. Pure stdlib HTTP server; no text generation.

Configuration (all optional, via environment):
  TWOCAN_POLICY        "default", "light" (two models, v0.1 behaviour), or a path to a policy JSON
  TWOCAN_BIND          bind address                   (default: 127.0.0.1)
  TWOCAN_PORT          port                           (default: 8766)
  TWOCAN_TOKEN_FILE    file holding a bearer token    (required when binding a non-loopback address)
  TWOCAN_HEADS         directory of trained choose-head JSON files (default: none)
  TWOCAN_REQUEST_LOG   append one JSON line per request (decision counts + latency, no text)
  TWOCAN_DEVICE        cuda | cpu                     (default: cuda if available)
  TWOCAN_BATCH_WAIT_MS micro-batching window         (default: 4)
  TWOCAN_TOKEN_BUDGET  max tokens per forward batch   (default: 16384)

Without a token the server only listens on loopback, and it rejects requests whose Host header is
not a loopback name or whose POST body is not application/json, so a web page cannot drive it.
"""
import os, re, json, time, hmac, hashlib, queue, threading, collections, glob, ipaddress, socket
import numpy as np
import torch
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.environ.get("TWOCAN_POLICY", "default")
POLICY_PATH = {"default": os.path.join(HERE, "policy.json"), "light": os.path.join(HERE, "policy_light.json")}.get(POLICY_PATH, POLICY_PATH)
TOKEN_PATH = os.environ.get("TWOCAN_TOKEN_FILE")
REQ_LOG = os.environ.get("TWOCAN_REQUEST_LOG")
HEADS_DIR = os.environ.get("TWOCAN_HEADS")
BIND = os.environ.get("TWOCAN_BIND", "127.0.0.1")
PORT = int(os.environ.get("TWOCAN_PORT", "8766"))
MAXLEN = 512
MAX_BODY = 4 * 1024 * 1024
MAX_ITEMS = 64
BATCH_WAIT_S = float(os.environ.get("TWOCAN_BATCH_WAIT_MS", "4")) / 1000.0
TOKEN_BUDGET = int(os.environ.get("TWOCAN_TOKEN_BUDGET", "16384"))
dev = os.environ.get("TWOCAN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
GPU_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
STARTED = time.time()

with open(POLICY_PATH, "rb") as f:
    POLICY_BYTES = f.read()
POLICY = json.loads(POLICY_BYTES)
POLICY_SHA = hashlib.sha256(POLICY_BYTES).hexdigest()[:12]
VERSION = POLICY["version"]
CAL = POLICY.get("verify", {}).get("calibration")


USE_FC = bool(POLICY.get("verify", {}).get("use_factcg"))
if CAL and USE_FC != ("w_fc" in CAL):
    raise SystemExit("policy mismatch: use_factcg=%s but calibration %s w_fc; use the matching policy file" % (USE_FC, "has" if "w_fc" in CAL else "lacks"))


def p_supported(a, b, f=None):
    """Calibrated P(supported): logistic on the verify probabilities' log-odds (policy verify.calibration).
    With use_factcg the FactCG probability is a third input (weight w_fc)."""
    e = CAL["clip"]
    lg = lambda p: float(np.log(min(max(p, e), 1 - e) / (1 - min(max(p, e), 1 - e))))
    z = CAL["w_mb"] * lg(a) + CAL["w_mc"] * lg(b) + CAL["bias"]
    if f is not None:
        z += CAL["w_fc"] * lg(f)
    return round(1.0 / (1.0 + float(np.exp(-z))), 5)


API_TOKEN = None
if TOKEN_PATH:
    with open(TOKEN_PATH) as f:
        API_TOKEN = f.read().strip()
    if len(API_TOKEN) < 32:
        raise SystemExit("token file too short (need >= 32 chars)")


def _is_loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


if API_TOKEN is None and not _is_loopback(BIND):
    raise SystemExit("refusing to bind %s without TWOCAN_TOKEN_FILE; set a token or bind 127.0.0.1" % BIND)

MODEL_SPECS = {
    "mb": {"id": "tasksource/ModernBERT-large-nli", "dtype": torch.bfloat16, "pos_label": "entailment"},
    "mc": {"id": "lytang/MiniCheck-RoBERTa-Large", "dtype": torch.float16, "pos_label": "1"},
    "zs": {"id": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "dtype": torch.bfloat16, "pos_label": "entailment"},
}
if USE_FC:  # single-text prompt model: input is FC_TPL, index 1 = supported
    MODEL_SPECS["fc"] = {"id": "yaxili96/FactCG-DeBERTa-v3-Large", "dtype": torch.float16, "pos_index": 1, "single": True, "maxlen": 2048}
FC_TPL = "{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that \"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is "
FC_CHUNK_WORDS = 550


class Engine:
    def __init__(self, key, spec):
        self.key = key
        self.id = spec["id"]
        self.single = spec.get("single", False)
        self.maxlen = spec.get("maxlen", MAXLEN)
        self.tok = AutoTokenizer.from_pretrained(spec["id"])
        if self.single:
            self.tok.pad_token = self.tok.eos_token
        dtype = spec["dtype"] if dev == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(spec["id"], dtype=dtype).to(dev).eval()
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.pos = spec["pos_index"] if "pos_index" in spec else [i for i, l in id2label.items() if l == spec["pos_label"]][0]
        self.q = queue.Queue()
        self.stats = collections.Counter()
        threading.Thread(target=self._loop, daemon=True, name="engine-" + key).start()

    def windows(self, text, hyp_len):
        budget = max(64, MAXLEN - hyp_len - 8)
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            return [text]
        step = max(32, budget - 50)
        return [self.tok.decode(ids[i:i + budget]) for i in range(0, len(ids), step)]

    def hyp_len(self, hyp):
        return len(self.tok.encode(hyp, add_special_tokens=False))

    def run(self, pairs):
        if not pairs:
            return []
        slot = {"pairs": pairs, "ev": threading.Event(), "out": None, "err": None}
        self.q.put(slot)
        slot["ev"].wait()
        if slot["err"] is not None:
            raise slot["err"]
        return slot["out"]

    def _loop(self):
        while True:
            batch = [self.q.get()]
            n = len(batch[0]["pairs"])
            deadline = time.time() + BATCH_WAIT_S
            while n < 256:
                rem = deadline - time.time()
                if rem <= 0:
                    break
                try:
                    s = self.q.get(timeout=rem)
                except queue.Empty:
                    break
                batch.append(s)
                n += len(s["pairs"])
            flat = [p for s in batch for p in s["pairs"]]
            try:
                probs = self._forward(flat)
                i = 0
                for s in batch:
                    s["out"] = probs[i:i + len(s["pairs"])]
                    i += len(s["pairs"])
                self.stats["batches"] += 1
                self.stats["requests"] += len(batch)
                self.stats["pairs"] += len(flat)
            except Exception as e:
                for s in batch:
                    s["err"] = e
                self.stats["errors"] += 1
            for s in batch:
                s["ev"].set()

    def embed(self, texts, half=254):
        feats = []
        with GPU_LOCK, torch.inference_mode():
            for k in range(0, len(texts), 16):
                encs = []
                for t in texts[k:k + 16]:
                    ids = self.tok.encode(str(t), add_special_tokens=False)
                    if len(ids) > 2 * half:
                        ids = ids[:half] + ids[-half:]
                    encs.append([self.tok.cls_token_id] + ids + [self.tok.sep_token_id])
                L = max(len(e) for e in encs)
                pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
                input_ids = torch.tensor([e + [pad] * (L - len(e)) for e in encs], device=dev)
                attn = torch.tensor([[1] * len(e) + [0] * (L - len(e)) for e in encs], device=dev)
                hs = self.model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True).hidden_states[-1].float()
                m = attn.unsqueeze(-1).float()
                v = (hs * m).sum(1) / m.sum(1)
                v = torch.nn.functional.normalize(v, dim=-1)
                feats.extend(v.cpu().numpy().tolist())
        self.stats["embedded"] += len(texts)
        return feats

    def _forward(self, flat):
        lens = [len(p[0]) // 3 + len(p[1] or "") // 3 + 8 for p in flat]
        order = sorted(range(len(flat)), key=lambda i: lens[i])
        out = [0.0] * len(flat)
        groups, cur = [], []
        for i in order:
            est = min(self.maxlen, lens[i])
            if cur and est * (len(cur) + 1) > TOKEN_BUDGET:
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        with GPU_LOCK, torch.inference_mode():
            for g in groups:
                if self.single:
                    enc = self.tok([flat[i][0] for i in g], truncation=True, max_length=self.maxlen, padding=True, return_tensors="pt").to(dev)
                else:
                    enc = self.tok([flat[i][0] for i in g], [flat[i][1] for i in g], truncation="only_first", max_length=self.maxlen, padding=True, return_tensors="pt").to(dev)
                p = torch.softmax(self.model(**enc).logits.float(), -1)[:, self.pos].tolist()
                for i, v in zip(g, p):
                    out[i] = v
        return out


ENG = {}
HEADS = {}
JSON_FRAG = re.compile(r'^\s*(?:[\[\]{}]|"[^"]{1,60}"\s*:)')


def load():
    for k, spec in MODEL_SPECS.items():
        ENG[k] = Engine(k, spec)
        print("loaded", k, spec["id"], "on", dev, flush=True)
    for hp in sorted(glob.glob(os.path.join(HEADS_DIR, "*.json"))) if HEADS_DIR else []:
        with open(hp, "rb") as f:
            hb = f.read()
        h = json.loads(hb)
        h["_W"] = np.array(h["coef"], dtype=np.float64)
        h["_b"] = np.array(h["intercept"], dtype=np.float64)
        h["_mu"] = np.array(h["scaler_mean"], dtype=np.float64)
        h["_sd"] = np.array(h["scaler_scale"], dtype=np.float64)
        h["sha"] = hashlib.sha256(hb).hexdigest()[:12]
        HEADS[h["name"]] = h
        print("head", h["name"], "labels", len(h["labels"]), "threshold", h["threshold"], flush=True)


def score_pairs(eng, premises, hyps):
    pairs, owner = [], []
    nwin = 0
    for j, (prem, hyp) in enumerate(zip(premises, hyps)):
        ws = eng.windows(prem, eng.hyp_len(hyp))
        nwin = max(nwin, len(ws))
        for w in ws:
            pairs.append((w, hyp))
            owner.append(j)
    probs = eng.run(pairs)
    best = [0.0] * len(hyps)
    for j, p in zip(owner, probs):
        best[j] = max(best[j], p)
    return best, len(pairs), nwin


SENT_END = re.compile(r"(?<=[.!?])\s+|\n+")


def fc_chunks(src):
    """Sentence-packed chunks of at most FC_CHUNK_WORDS words (FactCG/MiniCheck-style document chunking)."""
    chunks, ch, size = [], [], 0
    sents = []
    for s in (x.strip() for x in SENT_END.split(src)):
        w = s.split()
        if len(w) > FC_CHUNK_WORDS:   # split only run-on "sentences"; leave normal ones byte-for-byte
            sents += [" ".join(w[i:i + FC_CHUNK_WORDS]) for i in range(0, len(w), FC_CHUNK_WORDS)]
        elif s:
            sents.append(s)
    for s in sents:
        n = len(s.split())
        if ch and size + n > FC_CHUNK_WORDS:
            chunks.append("\n".join(ch)); ch, size = [], 0
        ch.append(s); size += n
    if ch:
        chunks.append("\n".join(ch))
    return chunks or [src]


def fc_prompt(chunk, claim):
    """FC_TPL with only the premise truncated, so the claim at the end of the prompt always survives."""
    tok, room = ENG["fc"].tok, ENG["fc"].maxlen - 8
    budget = room - len(tok.encode(FC_TPL.format(text_a="", text_b=claim), add_special_tokens=False))
    ids = tok.encode(chunk, add_special_tokens=False)
    if len(ids) > budget:
        chunk = tok.decode(ids[:max(0, budget)])
    return FC_TPL.format(text_a=chunk, text_b=claim)


def score_fc(source, claims):
    """Max over chunks of FactCG P(supported) for each claim."""
    chunks = fc_chunks(source)
    pairs = [(fc_prompt(c, cl), None) for cl in claims for c in chunks]
    probs = ENG["fc"].run(pairs)
    k = len(chunks)
    return [max(probs[j * k:(j + 1) * k]) for j in range(len(claims))], len(pairs)


def noul(reason, **kw):
    d = {"decision": "NOUL", "noul_reason": reason, "confidence": None}
    d.update(kw)
    return d


def claim_sha(c):
    return hashlib.sha1(c.encode()).hexdigest()[:12]


def head_verify(source, claims):
    pv = POLICY["verify"]
    source = "" if source is None else str(source)
    claims = [str(c) for c in claims]
    res = [None] * len(claims)
    todo = []
    for i, c in enumerate(claims):
        cs = c.strip()
        if not source.strip():
            res[i] = noul("empty_source", claim_sha=claim_sha(c))
        elif not cs:
            res[i] = noul("empty_claim", claim_sha=claim_sha(c))
        elif len(cs) > pv["max_claim_chars"]:
            res[i] = noul("claim_too_long", claim_sha=claim_sha(c))
        elif JSON_FRAG.match(cs):
            res[i] = noul("json_fragment", claim_sha=claim_sha(c))
        elif len(re.findall(r"[A-Za-z]{2,}", cs)) < pv["min_claim_words"]:
            res[i] = noul("not_a_claim", claim_sha=claim_sha(c))
        else:
            todo.append(i)
    stats = {"pairs": 0, "windows": 0}
    if todo:
        tc = [claims[i].strip() for i in todo]
        nwin = len(ENG["mb"].windows(source, 64))
        if nwin > pv["max_windows"]:
            for i in todo:
                res[i] = noul("source_too_long", claim_sha=claim_sha(claims[i]), windows=nwin)
            todo = []
        else:
            pm, n1, w1 = score_pairs(ENG["mb"], [source] * len(tc), tc)
            pc, n2, w2 = score_pairs(ENG["mc"], [source] * len(tc), tc)
            pf, n3 = score_fc(source, tc) if USE_FC else ([None] * len(tc), 0)
            stats = {"pairs": n1 + n2 + n3, "windows": max(w1, w2, n3 // max(1, len(tc)))}
            P, F = pv["pass_min"], pv["fail_max"]
            for k, i in enumerate(todo):
                a, b = pm[k], pc[k]
                base = {"claim_sha": claim_sha(claims[i]), "p_modernbert": round(a, 5), "p_minicheck": round(b, 5)}
                if USE_FC:
                    base["p_factcg"] = round(pf[k], 5)
                if CAL:
                    base["p_supported"] = p_supported(a, b, pf[k])
                if a >= P and b >= P:
                    base.update({"decision": "SUPPORTED", "confidence": round(min(a, b), 5), "noul_reason": None})
                elif a <= F and b <= F:
                    base.update({"decision": "UNSUPPORTED", "confidence": round(1 - max(a, b), 5), "noul_reason": None})
                else:
                    base.update({"decision": "NOUL", "confidence": None, "noul_reason": "models_disagree" if (a >= 0.5) != (b >= 0.5) else "uncertain"})
                res[i] = base
    for i, r in enumerate(res):
        r["claim_idx"] = i
    return {"type": "verify", "results": res, "stats": stats}


def option_parts(options):
    labels, hyps_raw = [], []
    for o in options:
        if isinstance(o, dict):
            labels.append(str(o["label"]))
            hyps_raw.append(o.get("hypothesis"))
        else:
            labels.append(str(o))
            hyps_raw.append(None)
    return labels, hyps_raw


def head_choose(text, options, multi=False, template=None):
    pc = POLICY["choose"]
    text = "" if text is None else str(text)
    if not text.strip():
        return {"type": "choose", **noul("empty_text")}
    if not isinstance(options, list) or len(options) < (1 if multi else 2) or len(options) > MAX_ITEMS:
        return {"type": "choose", **noul("bad_options")}
    labels, hraw = option_parts(options)
    if len(set(labels)) != len(labels):
        return {"type": "choose", **noul("duplicate_options")}
    tpl = template or pc["default_template"]
    hyps = [h if h else tpl.replace("{}", l) for l, h in zip(labels, hraw)]
    nwin = len(ENG["zs"].windows(text, 64))
    if nwin > pc["max_windows"]:
        return {"type": "choose", **noul("text_too_long", windows=nwin)}
    pz, npairs, _ = score_pairs(ENG["zs"], [text] * len(hyps), hyps)
    probs = {l: round(p, 5) for l, p in zip(labels, pz)}
    ranked = sorted(zip(labels, pz), key=lambda x: -x[1])
    if multi:
        sel = [l for l, p in ranked if p >= pc["multi_pass"]]
        unc = [l for l, p in ranked if pc["multi_fail"] < p < pc["multi_pass"]]
        out = {"type": "choose", "multi": True, "selected": sel, "uncertain": unc, "probs": probs, "stats": {"pairs": npairs}}
        if unc:
            out.update({"decision": "NOUL", "noul_reason": "uncertain_options", "confidence": None})
        else:
            margins = [abs(p - 0.5) * 2 for _, p in ranked]
            out.update({"decision": "SELECTED", "noul_reason": None, "confidence": round(min(margins), 5)})
        return out
    top, second = ranked[0], ranked[1]
    out = {"type": "choose", "multi": False, "probs": probs, "ranking": [l for l, _ in ranked], "stats": {"pairs": npairs}}
    if top[1] < pc["min_top"]:
        out.update({"decision": "NOUL", "noul_reason": "no_option_fits", "label": None, "confidence": None})
    elif top[1] - second[1] < pc["min_margin"]:
        out.update({"decision": "NOUL", "noul_reason": "ambiguous", "label": None, "confidence": None})
    else:
        out.update({"decision": "CHOSEN", "label": top[0], "noul_reason": None, "confidence": round(top[1] - second[1], 5)})
    return out


def head_trained_choose(text, head):
    h = HEADS.get(str(head))
    if h is None:
        raise ValueError("unknown head " + str(head))
    text = "" if text is None else str(text)
    if not text.strip():
        return {"type": "choose", "head": h["name"], **noul("empty_text")}
    x = np.array(ENG[h["encoder"]].embed([text])[0], dtype=np.float64)
    z = h["_W"] @ ((x - h["_mu"]) / h["_sd"]) + h["_b"]
    z = z - z.max()
    p = np.exp(z) / np.exp(z).sum()
    order = np.argsort(-p)
    probs = {h["labels"][i]: round(float(p[i]), 5) for i in range(len(p))}
    out = {"type": "choose", "multi": False, "head": h["name"], "head_sha": h["sha"], "probs": probs, "ranking": [h["labels"][i] for i in order]}
    top = float(p[order[0]])
    if top >= h["threshold"]:
        out.update({"decision": "CHOSEN", "label": h["labels"][order[0]], "confidence": round(top, 5), "noul_reason": None})
    else:
        out.update({"decision": "NOUL", "label": None, "confidence": None, "noul_reason": "below_head_threshold"})
    return out


def head_score(text, criterion):
    ps = POLICY["score"]
    text = "" if text is None else str(text)
    criterion = "" if criterion is None else str(criterion)
    if not text.strip() or not criterion.strip():
        return {"type": "score", **noul("empty_input"), "score": None}
    nwin = len(ENG["zs"].windows(text, 64))
    if nwin > ps["max_windows"]:
        return {"type": "score", **noul("text_too_long", windows=nwin), "score": None}
    p, npairs, _ = score_pairs(ENG["zs"], [text], [criterion])
    s = p[0]
    band = "HIGH" if s >= ps["high"] else ("LOW" if s <= ps["low"] else "MID")
    out = {"type": "score", "score": round(s, 5), "band": band, "stats": {"pairs": npairs}}
    if band == "MID":
        out.update({"decision": "NOUL", "noul_reason": "mid_band", "confidence": None})
    else:
        out.update({"decision": band, "noul_reason": None, "confidence": round(abs(s - 0.5) * 2, 5)})
    return out


def dispatch(op, body):
    if op == "verify":
        claims = body.get("claims")
        if claims is None:
            claims = [body.get("claim", "")]
        if not isinstance(claims, list) or len(claims) > MAX_ITEMS:
            raise ValueError("claims must be a list of at most %d" % MAX_ITEMS)
        return head_verify(body.get("source"), claims)
    if op == "choose" and body.get("head"):
        return head_trained_choose(body.get("text"), body.get("head"))
    if op == "embed":
        texts = body.get("texts")
        if not isinstance(texts, list) or not texts or len(texts) > MAX_ITEMS:
            raise ValueError("texts must be a non-empty list of at most %d" % MAX_ITEMS)
        enc = body.get("encoder", "zs")
        if enc not in ENG or ENG[enc].single:
            raise ValueError("unknown encoder")
        return {"type": "embed", "encoder": enc, "embeddings": ENG[enc].embed(texts), "stats": {"pairs": len(texts)}}
    if op == "choose":
        return head_choose(body.get("text"), body.get("options"), bool(body.get("multi", False)), body.get("template"))
    if op == "score":
        return head_score(body.get("text"), body.get("criterion"))
    raise ValueError("unknown op " + str(op))


def envelope(op, out, t0):
    out["version"] = VERSION
    out["policy_sha"] = POLICY_SHA
    out["latency_ms"] = round((time.time() - t0) * 1000, 2)
    if REQ_LOG:
        decs = collections.Counter()
        if op == "verify":
            for r in out["results"]:
                decs[r["decision"]] += 1
        elif op == "embed":
            decs["EMBED"] += 1
        else:
            decs[out.get("decision")] += 1
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "op": op, "decisions": dict(decs), "latency_ms": out["latency_ms"], "version": VERSION, "policy_sha": POLICY_SHA, "pairs": out.get("stats", {}).get("pairs")}
        with LOG_LOCK, open(REQ_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return out


class GateHTTPServer(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True
    address_family = socket.AF_INET6 if ":" in BIND else socket.AF_INET


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class Handler(BaseHTTPRequestHandler):
    server_version = "2can/0.2"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authed(self):
        if API_TOKEN is None:
            host = self.headers.get("Host", "")
            host = host[1:host.find("]")] if host.startswith("[") else host.split(":")[0]   # strip port / IPv6 brackets
            return host in LOOPBACK_HOSTS
        h = self.headers.get("Authorization", "")
        return h.startswith("Bearer ") and hmac.compare_digest(h[7:].strip().encode(), API_TOKEN.encode())

    def do_GET(self):
        if API_TOKEN is None and not self._authed():   # no-token mode: loopback Host only
            return self._send(401, {"error": "unauthorized"})
        if self.path == "/v1/health":
            return self._send(200, {"ok": True, "version": VERSION, "policy_sha": POLICY_SHA, "uptime_s": round(time.time() - STARTED, 1), "device": dev, "models": {k: e.id for k, e in ENG.items()}, "engines": {k: dict(e.stats) for k, e in ENG.items()}, "heads": {k: {"sha": h["sha"], "labels": h["labels"], "threshold": h["threshold"]} for k, h in HEADS.items()}})
        if self.path == "/v1/policy":
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            return self._send(200, POLICY)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        t0 = time.time()
        m = re.match(r"^/v1/(verify|choose|score|embed)$", self.path)
        if not m:
            return self._send(404, {"error": "not found"})
        if not self._authed():
            return self._send(401, {"error": "unauthorized"})
        if API_TOKEN is None and self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            return self._send(415, {"error": "Content-Type must be application/json"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send(400, {"error": "bad content-length"})
        if n <= 0 or n > MAX_BODY:
            return self._send(413 if n > MAX_BODY else 400, {"error": "body size"})
        try:
            body = json.loads(self.rfile.read(n))
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            out = dispatch(m.group(1), body)
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, {"error": str(e)[:300]})
        except Exception as e:
            return self._send(500, {"error": type(e).__name__})
        return self._send(200, envelope(m.group(1), out, t0))


def warmup():
    t0 = time.time()
    src = "The Golden Gate Bridge opened to traffic in 1937."
    r = head_verify(src, [src, "The Golden Gate Bridge opened to traffic in 1952."])
    assert r["results"][0]["decision"] == "SUPPORTED" and r["results"][1]["decision"] != "SUPPORTED", r
    print("warmup ok", round((time.time() - t0) * 1000), "ms", [x["decision"] for x in r["results"]], flush=True)


def main():
    load()
    warmup()
    httpd = GateHTTPServer((BIND, PORT), Handler)
    print("2can listening", BIND, PORT, "version", VERSION, "policy", POLICY_SHA, "auth", "on" if API_TOKEN else "off (loopback only)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
