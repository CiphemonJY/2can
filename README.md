---
license: mit
language:
- en
pipeline_tag: text-classification
tags:
- fact-checking
- claim-verification
- grounding
- hallucination-detection
- abstention
- calibration
---

# 2Can

**Two models have to agree, or it says "I don't know."**

2Can checks whether a claim is supported by a source text. It runs two independent encoders, an
NLI model and a fact-checking model, and only answers when both agree strongly:

| decision | when |
|---|---|
| `SUPPORTED` | both models ≥ `pass_min` (0.90) |
| `UNSUPPORTED` | both models ≤ `fail_max` (0.03) |
| `NOUL` | anything else: the models disagree, or neither is confident |

Every scored claim also gets **`p_supported`**, a single calibrated probability. Since v0.2 it
combines the two verify models with a third fact-checker, FactCG, which raised held-out AUROC from
0.871 to 0.889 (see below). The decision rule still uses only the two verify models. 2Can only
classifies and never generates text, so it cannot invent content, although it can still be wrong
(see the error rates below).

It is small: about 1.2B parameters across the three models that score a claim (the two that decide,
plus FactCG). In our runs on an RTX 5060 Ti 16 GB, scoring the evaluation sample one request at a
time took about 175 ms per claim. FactCG reads up to 2,048 tokens per chunk, so memory use can
exceed 12 GB: our run reached the card's limit while sharing it with a 3 GB process. **Use a
16 GB GPU, or lower `TWOCAN_TOKEN_BUDGET`.** CPU is much slower. The lighter v0.1 behavior (two
models, about half the time and memory) is kept as
[`two_can/policy_light.json`](two_can/policy_light.json): run `TWOCAN_POLICY=light 2can-serve`.

This repo ships **no new weights**. 2Can is a policy plus a calibration layer over four published
models (not fine-tuned or merged), which are downloaded from the Hub on first run:

| role | model | license |
|---|---|---|
| verify (NLI) | [tasksource/ModernBERT-large-nli](https://huggingface.co/tasksource/ModernBERT-large-nli) | Apache-2.0 |
| verify (fact-check) | [lytang/MiniCheck-RoBERTa-Large](https://huggingface.co/lytang/MiniCheck-RoBERTa-Large) | MIT |
| calibration input (fact-check) | [yaxili96/FactCG-DeBERTa-v3-Large](https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large) | MIT |
| choose / score (zero-shot) | [MoritzLaurer/ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) | Apache-2.0 |

## Quick start

```bash
pip install -e .
2can-serve                      # binds 127.0.0.1:8766; GPU if available, else CPU
```

```bash
curl -s localhost:8766/v1/verify -H 'Content-Type: application/json' -d '{
  "source": "The Golden Gate Bridge opened to traffic in 1937.",
  "claims": ["The bridge opened in 1937.", "The bridge opened in 1952.", "The Golden Gate Bridge is in Paris."]}'
```

| claim | `decision` | `p_supported` |
|---|---|---|
| The bridge opened in 1937. | `SUPPORTED` | 0.907 |
| The bridge opened in 1952. | `NOUL` (uncertain) | 0.076 |
| The Golden Gate Bridge is in Paris. | `NOUL` (uncertain) | 0.041 |

`UNSUPPORTED` is deliberately strict: both models must put the claim at or below 0.03, which only
about 10% of LLM-AggreFact claims reach. **To reject claims, threshold `p_supported`.** Treat
`SUPPORTED` as the high-precision tier: about 6% of `SUPPORTED` claims were still false on the
held-out set.

Each result carries `decision`, `noul_reason`, `p_supported`, the raw model probabilities
(`p_modernbert`, `p_minicheck`, and `p_factcg` when enabled), and `confidence` (the weaker verify
model's margin on a decision).

Other endpoints: `POST /v1/choose` (zero-shot pick among options, or `NOUL` when none fits or two
are too close), `POST /v1/score` (0-1 score of text against a criterion, `NOUL` in the middle
band), `POST /v1/embed`, `GET /v1/health`, `GET /v1/policy`.

To bind a non-loopback address, set a bearer token: `TWOCAN_TOKEN_FILE=/path/to/token` (at least
32 characters). The server refuses to listen publicly without one. Without a token it accepts only
loopback `Host` headers and JSON bodies, so a web page in your browser can't drive it. The rest of
the configuration is described at the top of [`two_can/server.py`](two_can/server.py).

## How the numbers were fit

Everything in [`two_can/policy.json`](two_can/policy.json) was fit on **public human labels**, the
dev split of [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact), and evaluated on
its held-out test split. (The splits share 110 of 2,849 test source documents, which matters
little for a 4-parameter fit.) At most 400 rows were sampled per sub-dataset (seed 0), so no
single source dominates. LLM-AggreFact is gated and licensed CC BY-ND 4.0, so only the fitted
parameters are distributed here, never its rows.
[`scripts/fit_calibration.py`](scripts/fit_calibration.py) reproduces both fits.

**Thresholds** use a fixed-sequence search that keeps the 95% Clopper-Pearson upper bound on the
false-accept and false-reject rates at or below 10% on dev, counting at most 3 claims per source
document.

**Held-out results** on the LLM-AggreFact test sample (4,358 claims, of which 4,347 were scored;
the other 11 were too long or not a claim). The numbers come from the GPU path (bf16/fp16). The
CPU path runs fp32. Its probabilities typically differ by less than 0.01, and about 1% of decisions
that sit near a threshold can flip.

| | |
|---|---|
| claims decided (not `NOUL`) | 35.5% of 4,358 |
| false accepts among `SUPPORTED` | 68 / 1,114 (6.1%); 95% upper bound 9.5% (document-capped) |
| false rejects among `UNSUPPORTED` | 36 / 433 (8.3%); 95% upper bound 10.9% (document-capped), **just above the 10% target** |
| calibration error (ECE) of `p_supported` | **0.015** (v0.1, two inputs: 0.016; averaging the two verify probabilities: 0.101) |
| AUROC of `p_supported` | **0.889** (v0.1, two inputs: 0.871; +0.018, 95% document-clustered bootstrap CI +0.013 to +0.023) |

The decision rows are identical in v0.1 and v0.2, because FactCG only feeds `p_supported`. The
combination (FactCG as a third calibration input), its pass bar and the test read were
pre-registered before any number for the combination was computed; FactCG's standalone benchmark
scores were already known. It was then re-confirmed through 2Can's own server code. Review of that
code found a bug: a source with a very long run and no sentence breaks could push the claim out of
FactCG's 2,048-token prompt. It affected 16 of the 8,707 evaluation rows. Those rows were re-scored
with the fixed code, the calibration weights were refit on dev, and the test result was re-read;
the gain was unchanged at +0.018, and
[`tests/`](tests/) now guards the fix. It helped most on AggreFact-CNN (+0.062 AUROC), Wice,
AggreFact-XSum and RAGTruth, and slightly hurt ExpertQA (−0.005) and FactCheck-GPT (−0.005). Two
other ideas tested the same way did not clear their bars:
a numbers-in-the-claim-missing-from-the-source feature (+0.002 AUROC), and deciding directly on
`p_supported` (more coverage, but the false-reject bound rose to 13.8%).

## Limitations

- **Calibration depends on the text you feed it.** `p_supported` was fit on LLM-AggreFact. Across
  its sub-datasets the test ECE ranges from 0.03 to 0.29 (ExpertQA and Reveal are worst), and it
  can be worse again on text unlike LLM-AggreFact. If you have labelled data from your own domain,
  refit with `scripts/fit_calibration.py`.
- **It abstains a lot, on purpose.** About two thirds of LLM-AggreFact claims come back `NOUL`.
  The design pairs 2Can with a slower judge (a person or an LLM) for the claims it won't decide.
- **Contradictions usually come back `NOUL`, not `UNSUPPORTED`** (see the quick-start table),
  including a changed number or a plain falsehood. `p_supported` still ranks them low.
- **Long sources** of more than 32 windows of 512 tokens return `NOUL` (`source_too_long`).
  FactCG reads sources in chunks of up to 550 words. In unusually token-dense text (long numbers,
  code, non-English) a chunk can exceed its 2,048-token prompt, and then the end of that chunk is
  not seen by FactCG. The claim itself is never cut.
- **English only**, as far as the upstream models go.
- **Commercial use:** 2Can's own code and policy are MIT. The upstream models carry their own
  licenses (above), and some of them were trained on research datasets with non-commercial terms,
  so check the upstream model cards before commercial use.

## Citation

If you use 2Can, please cite the upstream models and LLM-AggreFact:
Tang, Laban & Durrett, *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents* (2024),
and Lei et al., *FactCG: Enhancing Fact Checkers with Graph-Based Multi-Hop Data* (NAACL 2025).

## License

MIT © 2026 James Yeung
