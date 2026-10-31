import os, socket, sys
print("python:", sys.version.split()[0])
try:
    socket.gethostbyname("pypi.org"); print("dns: ok")
except OSError as e:
    print("dns: BLOCKED", e)
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, round(p.total_memory/1e9, 1), "GB")
else:
    print("gpu: NONE", torch.version.cuda)
root = "/kaggle/input"
for dirpath, dirnames, filenames in os.walk(root):
    depth = dirpath[len(root):].count(os.sep)
    if depth <= 3:
        print("dir:", dirpath.replace(root, ""), "| files", len(filenames), "| subdirs", len(dirnames))
    if depth >= 3:
        dirnames[:] = []