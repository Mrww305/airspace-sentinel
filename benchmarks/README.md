# Benchmarks

`results.json` is produced by the training notebook (`metrics.json`) and copied
here verbatim — never hand-edited. The console renders this file, so a claim on
the site is always downstream of a run artefact.

## Current

| Field | Value |
|---|---|
| Model | `sentinel-det-v1` — **pending training run** |
| Base | RFDETRNano (Apache-2.0) |
| mAP50 | — |
| mAP50-95 | — |
| AP-small | — |
| Recall@0.5 | — |
| **Bird FPR** | — |
| Latency p50 / p95 @640² CPU | — |
| Weights sha256 | — |
| Size (FP32 / INT8) | — |

## Acceptance gates

A model is only published on the demo when all four hold:

1. **mAP50 ≥ 0.55** — below that the demo misrepresents the state of the art.
2. **Bird FPR < 15%** — above that, a CLIP re-scoring stage is added before shipping.
3. **p95 CPU latency < 1500 ms @640²** — otherwise drop to 416² and re-measure.
4. **INT8 ΔmAP < 2 points** — otherwise ship FP32. Slower and honest beats fast and wrong.

Anything that fails a gate is published on the model card as a known limitation,
not quietly omitted.

## Reproduce

```bash
cd training
node build_notebook.mjs
# run airspace-sentinel-train.ipynb on Kaggle (T4 x1)
cp /kaggle/working/metrics.json ../benchmarks/results.json
```

Every entry records dataset licenses and the exact weight hash, so a number can
be traced to the graph that produced it.
