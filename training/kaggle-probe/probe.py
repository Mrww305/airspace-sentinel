import os, socket, sys
print("python:", sys.version.split()[0])
try:
    socket.gethostbyname("pypi.org"); print("dns: ok")
except OSError as e:
    print("dns: BLOCKED", e)
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
except Exception as e:
    print("torch: FAILED", e)
p = "/kaggle/input"
print("input:", os.listdir(p) if os.path.exists(p) else "MISSING")
if os.path.exists(p):
    for root, dirs, files in os.walk(p):
        print("walk:", root, len(files))
        if root.count(os.sep) > 3: dirs[:] = []
pre = ["rfdetr","transformers","supervision","pycocotools","timm","torchmetrics","onnxruntime","onnx","kagglehub"]
import importlib
for m in pre:
    try:
        mod = importlib.import_module(m); print("have:", m, getattr(mod,"__version__","?"))
    except Exception:
        print("miss:", m)
