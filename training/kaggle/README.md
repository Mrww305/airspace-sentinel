# Pushing the kernel from this machine

The Kaggle CLI is installed (`kaggle --version` → 2.2.5), but it has **no credentials** —
and it cannot, because the token is issued by your account. That is the only
step in this pipeline that needs you.

## 1 · Create the token (once, ~60 seconds)

Kaggle → avatar → **Settings** → *API* → **Create New Token**

It downloads `kaggle.json`. Put it here:

```
C:\Users\sajid\.kaggle\kaggle.json
```

Nothing else needs editing. Do not paste it into chat — it is a full account
credential.

> GPU runs require a phone-verified Kaggle account. Verify under Settings →
> Phone Verification, or the kernel will queue forever with no GPU slot.

## 2 · Verify the connection

```powershell
kaggle competitions list -s "titanic"     # or any list call
```

If that returns rows, auth works.

## 3 · Regenerate and push

```powershell
cd C:\Users\sajid\airspace-sentinel\training
node build_notebook.mjs                   # rebuilds .ipynb + kaggle/ staging
kaggle kernels push -p kaggle             # uploads, starts the run
```

The metadata attaches the three datasets and enables GPU + internet, so no
manual dataset clicking is needed.

## 4 · Watch it

```powershell
kaggle kernels status mrww305/airspace-sentinel-train
kaggle kernels output mrww305/airspace-sentinel-train -p ..\runs\kaggle
```

`output` pulls `metrics.json`, `model_card.md` and the `.onnx` graphs when the
run finishes. Expect ~2 h on T4; the export gate fails fast in ~10 min if the
ONNX path is broken.

## 5 · Publish the weights

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
