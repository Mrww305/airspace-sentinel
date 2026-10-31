# Running this on Google Colab

The Kaggle path is blocked: the account is not phone-verified, so Kaggle gives
the kernel neither network nor GPU (probe output in `../kaggle/README.md`).
Colab has no such gate.

## Open it

<https://colab.research.google.com/github/Mrww305/airspace-sentinel/blob/main/training/colab/airspace-sentinel-colab.ipynb>

The notebook lives in this repo, so that link always opens the committed
version — no upload, no drift between what ran and what is in git.

Regenerate it after editing `airspace_sentinel_train.py`:

```powershell
cd training
node build_notebook.mjs     # writes colab/ as well as kaggle/
```

## Two settings before running

1. **Runtime → Change runtime type → T4 GPU.** The first cell asserts CUDA and
   stops with a readable message if you forget.
2. **Kaggle token in Colab Secrets.** Click the key icon in the left sidebar,
   add a secret named `KAGGLE_API_TOKEN` with your `KGAT_…` token, and toggle
   "Notebook access" on. The notebook reads it with `google.colab.userdata`, so
   it never appears in a cell, in the saved notebook, or in git.

   The three datasets are public; the token is only there to satisfy the
   download API.

## What it does

Same pipeline as the Kaggle kernel: unify → dedupe → gate (1 epoch, ONNX export,
ONNX Runtime check) → full training → evaluate → INT8 → model card.

The gate is still the first hard stop. If `rfdetr.export()` is broken, it fails
in ~10 minutes instead of after the full run.

## Known Colab limitations

- **Session ceiling.** Free tier disconnects after ~12 h of runtime and on idle
  timeouts. A 60-epoch run on ~3,000 images at 640 px fits, but there is not
  much headroom. Checkpoints land in `runs/` if you need to resume manually.
- **No resumable storage.** `/content` is wiped between sessions. Run section 12
  before closing the tab — it zips the ONNX graphs, `metrics.json` and
  `model_card.md` and offers a single download.
- **Disk.** ~1.6 GB of datasets plus a copy of every image in COCO layout. Fits
  comfortably; not a concern until you add more sources.
- **Different torch than Kaggle.** Colab's image changes without notice, so a
  `metrics.json` produced here should record the versions it ran on if numbers
  are ever compared across platforms.

## Bringing the result back

Section 12 downloads `airspace-sentinel-model.zip`. Then, locally:

```powershell
# verify the graph loads before shipping it
.venv-gate\Scripts\python.exe -c "import onnxruntime as o; s=o.InferenceSession('sentinel-det-v1.onnx', providers=['CPUExecutionProvider']); print(s.get_inputs()[0].shape, [x.shape for x in s.get_outputs()])"

# pin it on the deployment (sha256 is printed by the notebook)
```

Set `SENTINEL_MODEL_PATH` and `SENTINEL_MODEL_SHA` on `vision.sajidafridi.com`
and redeploy; `/health` flips from `kind: "mock"` to `kind: "onnx"`.
