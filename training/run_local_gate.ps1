# Runs the training notebook's GATE section on this machine, CPU only.
#
# Purpose: the Kaggle kernel is blocked on account verification (no internet,
# no GPU), and the one genuinely unproven link in the pipeline is whether
# rfdetr's ONNX export produces a graph ONNX Runtime can execute. That link
# does not need a GPU to test.
#
# It stops after the gate: full training on CPU is not viable.
#
#   powershell -ExecutionPolicy Bypass -File training\run_local_gate.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv-gate\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Error "Missing .venv-gate. Create it with: uv venv .venv-gate --python 3.11; uv pip install --python .venv-gate\Scripts\python.exe rfdetr==1.11.2 onnxruntime onnx torchmetrics"
}

$cache = Join-Path $repo ".cache-gate"
$env:SENTINEL_ALLOW_CPU = "1"         # skip the CUDA assertion
$env:SENTINEL_STOP_AFTER_GATE = "1"   # do not start the 60-epoch run
$env:SENTINEL_DATA_ROOT = Join-Path $repo "data\raw"
$env:TORCH_HOME = $cache              # keep the pretrained weights out of ~/.cache
$env:HF_HOME = $cache
$env:KAGGLEHUB_CACHE = $cache

$work = Join-Path $repo "work"
New-Item -ItemType Directory -Force -Path $work | Out-Null
Set-Location $work

Write-Host "gate run: data=$($env:SENTINEL_DATA_ROOT)" -ForegroundColor Cyan
& $py (Join-Path $repo "training\airspace_sentinel_train.py")
exit $LASTEXITCODE
