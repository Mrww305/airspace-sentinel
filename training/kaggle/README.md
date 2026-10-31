# Running the training kernel

Status as of the last attempt: **the account cannot run this kernel yet.** Not a
code problem, not a credential problem — Kaggle denies the container both
network and GPU because the account is not phone-verified.

## What the probe actually showed

A minimal kernel pushed to the same account, on `gcr.io/kaggle-gpu-images/python`
with `enable_gpu: true` and `enable_internet: true`, printed:

```
python: 3.13.15
dns: BLOCKED   [Errno -3] Temporary failure in name resolution
torch: 2.11.0+cpu   cuda: False
gpu: NONE
```

On a granted run that same image reports `torch 2.x+cu128` and a `Tesla T4`.
Getting the CPU build means the GPU was never allocated, so `enable_gpu` in
`kernel-metadata.json` is being honoured and then overridden by account policy.

Sources: `training/kaggle-probe/` (push it again any time with
`kaggle kernels push -p training/kaggle-probe`).

## The one action that unblocks it

Kaggle → **Settings** → *Phone Verification*. Both internet and GPU unlock
together; there is no API for this and no workaround that keeps a GPU in play.

After verifying, nothing needs editing:

```powershell
cd C:\Users\sajid\airspace-sentinel\training
node build_notebook.mjs
kaggle kernels push -p kaggle
```

## Auth, which already works

The CLI (2.2.5) reads `~/.kaggle/access_token` (a `KGAT_…` token from Settings →
API) or the legacy `kaggle.json`. Both are alternatives, not both required.
Verify with:

```powershell
kaggle competitions list -s "titanic"
```

## Watch it

```powershell
$slug = "sajidmrww305afridi/airspace-sentinel-counter-uas-detector"
kaggle kernels status $slug
kaggle kernels output $slug -p ..\runs\kaggle
```

`output` pulls `metrics.json`, `model_card.md` and the `.onnx` graphs. Expect
~2 h on T4. The gate cell fails in ~10 min if the ONNX path is broken, which is
the point of it existing.

## Gate without a GPU

`training/run_local_gate.ps1` runs sections 1–6 of the notebook on this machine
and stops before the 60-epoch run:

```powershell
powershell -ExecutionPolicy Bypass -File training\run_local_gate.ps1
```

It exports the model to ONNX and executes the graph through ONNX Runtime, which
is the only part of the pipeline that was never proven. Training is not viable
here — no CUDA device — so this is a correctness check, not a substitute for the
Kaggle run.

## Publish the weights

Upload the graph as a Kaggle Dataset (private is fine):

```powershell
kaggle datasets create -p .\kaggle-model --dir-mode zip
```

then set `SENTINEL_MODEL_PATH` and `SENTINEL_MODEL_SHA` on
`vision.sajidafridi.com` and redeploy.

## Why the CLI and not a browser upload

The kernel is generated from `airspace_sentinel_train.py`. Pushing from the CLI
means the notebook that ran is byte-identical to the notebook in the repo, and
`kernel-metadata.json` records the datasets and GPU setting that produced the
metrics — provenance, not a screenshot.
