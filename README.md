# AIRSPACE SENTINEL

**CPU-only counter-UAS vision stack** — a fine-tuned drone detector, ONNX Runtime serving inside Node, and an audit chain that records what the model saw and why it escalated.

Live demo target: `https://vision.sajidafridi.com` · POC-04 of [sajidafridi.com](https://sajidafridi.com/lab)

---

## Why this exists

Most counter-UAS demos stop at drawing a box around a drone. That is the easy half.

The hard half is the part that decides whether a system is deployable:

| Question | What this repo reports |
|---|---|
| Does it find a 34×23 px drone at 200 m? | AP-small, recall@0.5 on held-out sequences |
| Does it scream at every bird? | **False-positive rate on Drone-vs-Bird** — the number that kills most pilots |
| Can it run without a GPU? | INT8 ONNX, single-digit ms budget documented per resolution |
| Can you audit a decision later? | Hash-chained record per inference: model id, weights sha256, input hash, policy verdict |
| Will it leak the drone operator's coordinates? | EXIF/GPS stripped server-side before anything is persisted |
| What is it *not* allowed to do? | Model card states known failure modes and out-of-scope use |

The thesis: **vision without governance is a liability.** This repo is the smallest complete system that shows both halves.

## Architecture

```
                    ┌──────────────────────────────────────────┐
  browser           │  apps/web  — operator console            │
  (image / webcam)  │  dropzone · canvas overlay · tracks      │
                    │  threat panel · audit trail · model card │
                    └───────────────────┬──────────────────────┘
                                        │  multipart, ≤8 frames, ≤2 MB each
                    ┌───────────────────▼──────────────────────┐
                    │  apps/api  — Node 24 · Fastify           │
                    │                                          │
                    │  1. validate  (magic bytes, size, count) │
                    │  2. strip EXIF/GPS  (before any write)   │
                    │  3. letterbox 640²  + normalize          │
                    │  4. ONNX Runtime  (INT8, 4 threads)      │
                    │  5. NMS (TS)  →  IoU tracker (SORT-lite) │
                    │  6. threat rules  →  NOMINAL/ELEVATED/CRITICAL
                    │  7. policy engine →  allow / deny / redact
                    │  8. append-only audit chain (sha256)      │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │  models/sentinel-det-v1.onnx  (sha256)   │
                    │  trained on Kaggle T4, exported INT8     │
                    └──────────────────────────────────────────┘
```

No Python in production. Training is offline; only the exported graph ships.

## Quickstart (mock mode — no weights required)

```bash
cd apps/api
npm approve-scripts sharp      # npm 11 blocks install scripts; sharp needs its binary
npm install
npm run dev
# → http://localhost:8080/health   { "model": "mock", "ok": true }
```

```bash
curl -s -X POST http://localhost:8080/api/v1/detect \
  -F "frame=@sample.jpg" -F 'config={"confidence":0.35}' | jq
```

Mock mode returns synthetic tracks so the whole pipeline — validation, tracking, threat rules, audit chain — is exercisable before the model exists. Swap in a real graph by setting `SENTINEL_MODEL_PATH`; the response contract does not change.

## Training

```bash
cd training
pip install -r requirements.txt
# open airspace-sentinel-train.ipynb on Kaggle (T4 x1 is enough)
```

The notebook is generated from `airspace_sentinel_train.py` (jupytext-style `# %%` cells):

```bash
node build_notebook.mjs      # regenerates airspace-sentinel-train.ipynb
```

Pipeline: dataset unification → COCO conversion → **ONNX export smoke test on 1 epoch** → full train → eval → INT8 quantize → model card.

Step 3 is a gate, not a formality: the export path is the one unproven link, so it gets proven in hour one, before any GPU hours are spent.

## Datasets and license provenance

| Dataset | Used for | License / terms | Redistributed? |
|---|---|---|---|
| [dasmehdixtr/drone-dataset-uav](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav) | train | see Kaggle dataset card | no — weights only |
| [muki2003/yolo-drone-detection-dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) | train | see Kaggle dataset card | no — weights only |
| [muhammadsaoodsarwar/drone-vs-bird](https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird) | bird FPR eval | see Kaggle dataset card | no — metrics only |
| Anti-UAV (anti-uav.github.io) | eval | research-only | no |

**We ship weights and metrics, never the data.** Every dataset is recorded with its license in `benchmarks/results.json` and in the model card at build time.

## Stack and license hygiene

The stack was chosen so a public, commercial-facing demo carries no copyleft surprise:

| Component | License | Note |
|---|---|---|
| [RF-DETR](https://github.com/roboflow/rf-detr) (`rfdetr`) | Apache-2.0 | detector, fine-tuned |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | serving |
| [supervision](https://github.com/roboflow/supervision) | MIT | eval utilities only |
| **Ultralytics YOLO** | **AGPL-3.0** | **deliberately not used for shipped weights** |

An AI-governance artefact that violates a dependency license is a contradiction. AGPL weights would attach to anything served from the subdomain, so the default path is Apache-2.0 end to end.

## Status

- [x] Repository scaffold, response contract, governance modules
- [x] Training pipeline + export smoke test (notebook)
- [x] Deployed to `vision.sajidfridi.com` **in mock mode** (honest label: `/health` reports `kind: "mock"`)
- [ ] Trained weights `sentinel-det-v1` (Kaggle run)
- [ ] INT8 quantized graph + accuracy delta
- [ ] Operator console (POC-04 UI)
- [ ] Published benchmark table (mAP50, AP-small, bird FPR, ms/frame)

Live endpoints: `GET https://vision.sajidafridi.com/health` · `POST /api/v1/detect` · `GET /api/v1/audit/recent`

> The subdomain currently runs the **mock detector**: the full pipeline (tracking,
> threat escalation, policy verdict, audit chain) is real, the detections are
> synthetic and every response says so. It flips to a real model when
> `SENTINEL_MODEL_PATH` points at trained weights.

## License

Code: Apache-2.0 (see `LICENSE`). Model weights: released under the same terms once the model card is published.

---

Built by [Sajid Afridi](https://sajidafridi.com) — AI Governance Architect.
