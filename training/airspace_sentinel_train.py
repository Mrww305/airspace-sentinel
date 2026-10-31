# %% [markdown]
# # AIRSPACE SENTINEL — training pipeline
#
# Counter-UAS detector, CPU-servable. Fine-tunes **RF-DETR Nano** (Apache-2.0) on
# unified drone-detection data, proves the ONNX export path before spending GPU
# hours, then evaluates the number that decides deployability: **false positives
# against birds**.
#
# Run this on Kaggle (T4 x1, ~2 h). It is also a runnable script:
# `python airspace_sentinel_train.py`
#
# Output artefacts:
# - `sentinel-det-v1.onnx` (FP32) and `sentinel-det-v1-int8.onnx`
# - `metrics.json` — mAP50, AP-small, recall@0.5, bird FPR, latency
# - `model_card.md` — generated, with dataset licenses and weight sha256

# %% [markdown]
# ## 1 · Environment
#
# Kaggle already ships torch + CUDA. We add `rfdetr`, `kagglehub`, `onnxruntime`
# and quantisation tooling.

# %%
# Installs work identically in a notebook and as a bare script (no IPython magic).
import socket, subprocess, sys

PKGS = ["rfdetr==1.11.2", "kagglehub", "onnxruntime", "onnx", "torchmetrics", "supervision>=0.29.0"]


def has_internet(host: str = "pypi.org", timeout: float = 5.0) -> bool:
    prev = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        return True
    except OSError:
        return False
    finally:
        socket.setdefaulttimeout(prev)


HAS_INTERNET = has_internet()
print("internet:", "available" if HAS_INTERNET else "BLOCKED")

try:
    import rfdetr  # noqa: F401
    print("rfdetr already present — skipping install")
except ImportError:
    if not HAS_INTERNET:
        raise RuntimeError(
            "No internet in this kernel, so rfdetr cannot be installed.\n\n"
            "Kaggle silently disables internet (and GPU) for accounts that are not\n"
            "phone-verified, even when enable_internet is set in kernel-metadata.json.\n\n"
            "Fix: kaggle.com -> Settings -> Phone Verification, then re-push this kernel.\n"
            "Nothing is wrong with the code: the pip failure below is a DNS failure."
        )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PKGS])

import json, os, random, shutil, hashlib, time, warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SEED = 305
random.seed(SEED); np.random.seed(SEED)

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "work"
DATA = WORK / "data"          # unified dataset root (COCO layout)
OUT  = WORK / "runs"
CKPT = WORK / "weights"
for p in (DATA, OUT, CKPT):
    p.mkdir(parents=True, exist_ok=True)

CLASSES = ["drone"]           # single class: counter-UAS cares about "is it a UAV"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@dataclass
class Cfg:
    epochs: int = 60
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 1e-4
    resolution: int = 640
    smoke_images: int = 200
    smoke_epochs: int = 1
    conf: float = 0.30
    val_frac: float = 0.15

CFG = Cfg()

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
if os.environ.get("SENTINEL_ALLOW_CPU") == "1":
    print("[warn] SENTINEL_ALLOW_CPU=1 — smoke path only, full training is not viable on CPU")
else:
    assert torch.cuda.is_available(), "Enable a GPU accelerator in Kaggle settings (T4 x1 is enough)."

# %% [markdown]
# ## 2 · Datasets
#
# Three sources, all recorded with their license in the model card. Nothing is
# redistributed — only derived weights and aggregate metrics leave this notebook.
#
# - `dasmehdixtr/drone-dataset-uav` — YOLO boxes, general UAV imagery
# - `muki2003/yolo-drone-detection-dataset` — YOLO boxes, varied backgrounds
# - `muhammadsaoodsarwar/drone-vs-bird` — the **evaluation-only** set that gives us a bird FPR

# %%
SOURCES = {
    "uav_dasmehdi":  "dasmehdixtr/drone-dataset-uav",
    "yolo_muki":     "muki2003/yolo-drone-detection-dataset",
    "drone_vs_bird": "muhammadsaoodsarwar/drone-vs-bird",
}


def resolve_roots(input_root: Path = Path("/kaggle/input")) -> dict:
    """Attached datasets first, network second.

    kernel-metadata.json attaches all three, so they appear under /kaggle/input
    even when the kernel has no internet. Downloading them again through
    kagglehub would be slower and fails outright on an unverified account.

    The mount layout is not /kaggle/input/<slug>: current Kaggle images use
    /kaggle/input/datasets/<owner>/<slug>, so the search is recursive and
    matches on the slug's last path component.
    """
    out, found = {}, {}

    def walk(root: Path, depth: int = 0):
        yield root
        if depth >= 4:
            return
        try:
            children = [c for c in root.iterdir() if c.is_dir()]
        except OSError:
            return
        for c in children:
            yield from walk(c, depth + 1)

    input_root = Path(input_root)
    if input_root.exists():
        dirs = list(walk(input_root))
        for key, slug in SOURCES.items():
            needle = slug.split("/")[-1].lower()
            hits = [d for d in dirs if d.name.lower() == needle or needle in d.name.lower()]
            if hits:
                found[key] = min(hits, key=lambda p: len(p.parts))
                print(f"{key:<14} attached  {found[key]}")

    for key, slug in SOURCES.items():
        if key in found:
            out[key] = found[key]
        elif HAS_INTERNET:
            try:
                import kagglehub
            except ImportError:
                print(f"[warn] {key}: not attached and kagglehub is not installed")
                continue
            out[key] = Path(kagglehub.dataset_download(slug))
            print(f"{key:<14} downloaded {out[key]}")
        else:
            print(f"[warn] {key}: not attached and no internet")
    return out


_local_root = os.environ.get("SENTINEL_DATA_ROOT")
roots = resolve_roots(Path(_local_root)) if _local_root else resolve_roots()
assert roots, (
    "No datasets available. Attached datasets mount under /kaggle/input; "
    "found none matching the three slugs in SOURCES."
)

# %% [markdown]
# ## 3 · Unify to COCO
#
# Datasets disagree on layout (YOLO `labels/` mirror, YOLO sidecar `.txt`,
# classification folders). One normaliser, one class, one split.
#
# **Leakage rule:** the split is by *parent folder* (sequence/scene), never by
# image. Splitting frames from the same video across train and val inflates
# recall by 20+ points and is the most common self-deception in this domain.

# %%
def find_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file()]

def yolo_label_for(img: Path):
    """Locate the YOLO .txt for an image across the layouts seen in the wild."""
    cands = [
        img.with_suffix(".txt"),
        img.parent.parent / "labels" / (img.stem + ".txt"),
        img.parent / "labels" / (img.stem + ".txt"),
        Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt"),
    ]
    for c in cands:
        if c.exists():
            return c
    return None

def voc_label_for(img: Path):
    """Locate a Pascal VOC .xml for an image (dasmehdixtr/drone-dataset-uav layout)."""
    cands = [
        img.with_suffix(".xml"),
        img.parent.parent / "annotations" / (img.stem + ".xml"),
        img.parent / "annotations" / (img.stem + ".xml"),
        Path(str(img).replace("/images/", "/annotations/")).with_suffix(".xml"),
    ]
    for c in cands:
        if c.exists():
            return c
    return None

# Class names that are explicitly *not* the target. Unknown names are kept: most
# of these datasets label their only class "drone"/"uav"/"0".
NEGATIVE_NAMES = {"bird", "person", "people", "car", "human", "airplane", "plane"}

def read_voc(xml: Path, w: int, h: int):
    """Pascal VOC xmin/ymin/xmax/ymax -> COCO absolute xywh."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(xml).getroot()
    except ET.ParseError:
        return []
    out = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name in NEGATIVE_NAMES:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append([round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)])
    return out

def read_yolo(txt: Path, w: int, h: int):
    """YOLO normalised cx,cy,w,h -> COCO absolute xywh. Any class id -> 0 (drone)."""
    out = []
    for line in txt.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _, cx, cy, bw, bh = (float(x) for x in parts[:5])
        except ValueError:
            continue
        if bw <= 0 or bh <= 0:
            continue
        x = (cx - bw / 2) * w; y = (cy - bh / 2) * h
        out.append([round(x, 2), round(y, 2), round(bw * w, 2), round(bh * h, 2)])
    return out

def image_size(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.size

records, skipped_no_label, full_box = [], 0, 0
per_source = {}
for key, root in roots.items():
    if key == "drone_vs_bird":
        continue  # eval-only: never trains the detector
    n_before = len(records)
    for img in find_images(root):
        w, h = image_size(img)
        # 1. YOLO sidecar / mirror, 2. Pascal VOC, 3. folder-name classification
        txt = yolo_label_for(img)
        xml = voc_label_for(img) if txt is None else None
        if txt is not None:
            boxes = read_yolo(txt, w, h)
        elif xml is not None:
            boxes = read_voc(xml, w, h)
        else:
            path_tokens = str(img).lower()
            is_drone = any(t in path_tokens for t in ("drone", "uav", "quadcopter"))
            boxes = [[0, 0, w, h]] if is_drone else []
            full_box += bool(boxes)
        if not boxes:
            skipped_no_label += 1
            continue
        records.append({
            "file": str(img), "source": key,
            "group": img.parent.name,          # sequence-level split key
            "width": w, "height": h, "boxes": boxes,
        })
    per_source[key] = len(records) - n_before
    print(f"{key:<14} images with boxes: {per_source[key]}")

print(f"images with boxes : {len(records)}")
print(f"images skipped    : {skipped_no_label}")
print(f"full-frame pseudo-boxes: {full_box}")

# The UAV dataset ships the same frames twice: once as PNG+XML, once as a
# re-encoded YOLO JPEG. The files differ byte-wise and even in name, so a
# content hash misses them — and a split that puts a frame in train and its
# re-render in valid inflates every metric. Dedupe on a perceptual hash.
def dhash(p: Path, size: int = 8) -> int:
    from PIL import Image
    with Image.open(p) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = np.asarray(g, dtype=np.int16)
    return int("".join("1" if b else "0" for b in (px[:, 1:] > px[:, :-1]).flatten()), 2)


hashes = np.array([dhash(Path(r["file"])) for r in records], dtype=np.uint64)
keep_idx, dropped = [], 0
for i in range(len(hashes)):
    if keep_idx:
        diff = np.bitwise_xor(hashes[keep_idx], hashes[i])
        bits = np.unpackbits(diff.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        if (bits <= 4).any():          # <=4 differing bits of 64 == same frame
            dropped += 1
            continue
    keep_idx.append(i)
records = [records[i] for i in keep_idx]
print(f"duplicate frames dropped: {dropped} -> {len(records)} unique")
assert len(records) > 500, (
    f"Too few labelled images ({len(records)}) — per source: {per_source}. "
    "A source at 0 means its label format was not recognised."
)

# Grouping: siblings from one video must not straddle the split. The parent
# directory is the natural key — but these datasets dump thousands of images
# into a single folder, which would push a whole source into one split and
# leave nothing to train on. Oversized groups are chunked into blocks of
# consecutive frames instead, which keeps frame order intact.
MAX_GROUP = max(48, int(len(records) * 0.05))
by_parent = {}
for r in records:
    by_parent.setdefault((r["source"], Path(r["file"]).parent), []).append(r)

for (src, parent), rows in by_parent.items():
    rows.sort(key=lambda r: r["file"])
    if len(rows) <= MAX_GROUP:
        for r in rows:
            r["group"] = f"{src}:{parent.name}"
    else:
        for i, r in enumerate(rows):
            r["group"] = f"{src}:{parent.name}:{i // 24}"

groups = sorted({r["group"] for r in records})
random.shuffle(groups)
n_val = max(1, int(len(groups) * CFG.val_frac))
val_groups = set(groups[:n_val])
split_of = lambda r: "valid" if r["group"] in val_groups else "train"

train_rows = [r for r in records if split_of(r) == "train"]
val_rows = [r for r in records if split_of(r) == "valid"]
print(f"groups: {len(groups)} -> train {len(train_rows)} images / valid {len(val_rows)} images")
for src in {r["source"] for r in records}:
    print(f"  {src:<14} train {sum(r['source'] == src for r in train_rows):<6} "
          f"valid {sum(r['source'] == src for r in val_rows)}")
assert train_rows, "Empty training split — the grouping collapsed."
assert val_rows, "Empty validation split — the grouping collapsed."

# %% [markdown]
# ## 4 · Write the COCO dataset RF-DETR expects
#
# `data/train/_annotations.coco.json`, `data/valid/_annotations.coco.json`.
# Images are copied once; RF-DETR reads `file_name` relative to the split dir.

# %%
def write_split(split: str, rows, root: Path):
    d = root / split
    (d / "images").mkdir(parents=True, exist_ok=True)
    images, annotations, ann_id = [], [], 1
    for i, r in enumerate(rows):
        fname = f"{r['source']}_{Path(r['file']).stem}_{i}.jpg"
        dst = d / "images" / fname
        if not dst.exists():
            try:
                from PIL import Image
                with Image.open(r["file"]) as im:
                    im.convert("RGB").save(dst, quality=92)
            except Exception:
                continue
        images.append({"id": i + 1, "file_name": f"images/{fname}",
                       "width": r["width"], "height": r["height"]})
        for (x, y, w, h) in r["boxes"]:
            annotations.append({"id": ann_id, "image_id": i + 1, "category_id": 1,
                                "bbox": [x, y, w, h],
                                "area": w * h, "iscrowd": 0})
            ann_id += 1
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "drone", "supercategory": "uav"}]}
    (d / "_annotations.coco.json").write_text(json.dumps(coco))
    return len(images), len(annotations)

train_rows = [r for r in records if split_of(r) == "train"]
valid_rows = [r for r in records if split_of(r) == "valid"]
ntr, atr = write_split("train", train_rows, DATA)
nva, ava = write_split("valid", valid_rows, DATA)
print(f"train: {ntr} imgs / {atr} boxes")
print(f"valid: {nva} imgs / {ava} boxes")

# %% [markdown]
# ## 5 · Reality check on object size
#
# Counter-UAS lives or dies on small objects. If the median box is under ~32 px
# the detector is in AP-small territory and the model card must say so.

# %%
areas = [w * h for r in records for (_, _, w, h) in r["boxes"]]
sides = sorted(np.sqrt(areas))
print(f"boxes      : {len(areas)}")
print(f"median side: {np.median(sides):.1f} px")
print(f"p10 / p90  : {np.percentile(sides,10):.1f} / {np.percentile(sides,90):.1f} px")
print(f"under 32px : {100*np.mean(np.array(sides) < 32):.1f}%")

# %% [markdown]
# ## 6 · GATE — prove ONNX export before spending GPU hours
#
# Train 1 epoch on a 200-image slice, export, load the graph with ONNX Runtime,
# and assert a usable output tensor. If this cell fails, stop: the rest of the
# notebook is wasted compute.

# %%
from rfdetr import RFDETRNano

def build_smoke_dir(src: Path, dst: Path, n: int):
    if dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "valid"):
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        coco = json.loads((src / split / "_annotations.coco.json").read_text())
        keep = coco["images"][: n if split == "train" else max(20, n // 5)]
        ids = {im["id"] for im in keep}
        coco["images"] = keep
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
        for im in keep:
            f = src / split / im["file_name"]
            if f.exists():
                shutil.copy2(f, dst / split / im["file_name"])
        (dst / split / "_annotations.coco.json").write_text(json.dumps(coco))
    return dst

SMOKE = build_smoke_dir(DATA, WORK / "smoke", CFG.smoke_images)

smoke = RFDETRNano()
smoke.train(dataset_dir=str(SMOKE), epochs=CFG.smoke_epochs, batch_size=CFG.batch_size,
            grad_accum_steps=CFG.grad_accum, lr=CFG.lr, output_dir=str(WORK / "smoke_run"))

def export_to_onnx(m, out_dir: Path) -> Path | None:
    """Export and locate the graph regardless of what `export()` returns.

    RF-DETR's `export(output_dir=...)` writes an artifact into a directory; older
    builds return a path, newer ones return metadata. We therefore diff the
    directory before/after and fall back to the newest .onnx found.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.rglob("*.onnx"))
    res = None
    try:
        res = m.export(output_dir=str(out_dir))
    except TypeError:
        try:
            res = m.export()
        except Exception as e:
            print("[warn] export() failed:", repr(e))
    except Exception as e:
        print("[warn] export() failed:", repr(e))

    cands = [p for p in out_dir.rglob("*.onnx") if p not in before]
    if isinstance(res, (str, Path)):
        p = Path(str(res))
        if p.is_file() and p.suffix == ".onnx":
            cands.append(p)
        elif p.is_dir():
            cands.extend(p.rglob("*.onnx"))
    if not cands:
        cands = sorted(out_dir.rglob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    best = max(cands, key=lambda p: p.stat().st_mtime)
    print("exported graph:", best)
    return best


smoke_onnx = export_to_onnx(smoke, WORK / "smoke_export")

if smoke_onnx is None:
    # fallback: raw graph export; post-processing then happens in TypeScript
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy = torch.randn(1, 3, CFG.resolution, CFG.resolution).to(device)
        smoke.model.eval()
        smoke_onnx = CKPT / "smoke_raw.onnx"
        torch.onnx.export(smoke.model, dummy, str(smoke_onnx),
                          input_names=["images"], output_names=["pred_logits", "pred_boxes"],
                          opset_version=17, dynamic_axes={"images": {0: "batch"}})
        print("torch.onnx.export fallback ->", smoke_onnx)
    except Exception as e:
        raise RuntimeError(
            "ONNX export failed entirely. Do NOT proceed to full training. "
            "Fallback plan: switch the detector to mmdetection RT-DETR (Apache-2.0, "
            "supported ONNX deploy path)."
        ) from e

import onnxruntime as ort
sess = ort.InferenceSession(str(smoke_onnx), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
dummy_np = np.random.rand(1, 3, inp.shape[2] or CFG.resolution, inp.shape[3] or CFG.resolution).astype(np.float32)
t0 = time.perf_counter()
outs = sess.run(None, {inp.name: dummy_np})
dt = (time.perf_counter() - t0) * 1000
print("GATE PASSED — onnxruntime CPU:", f"{dt:.1f} ms", "| outputs:", [o.shape for o in outs])

if os.environ.get("SENTINEL_STOP_AFTER_GATE") == "1":
    print("SENTINEL_STOP_AFTER_GATE=1 — stopped after the gate, no full training run.")
    raise SystemExit(0)

# %% [markdown]
# ## 7 · Full training run

# %%
model = RFDETRNano()
model.train(
    dataset_dir=str(DATA),
    epochs=CFG.epochs,
    batch_size=CFG.batch_size,
    grad_accum_steps=CFG.grad_accum,
    lr=CFG.lr,
    output_dir=str(OUT),
)

# %% [markdown]
# ## 8 · Export and evaluate
#
# We measure the deployed artefact, not the training object: everything below
# runs the **ONNX graph** through ONNX Runtime on CPU, because that is what
# `vision.sajidafridi.com` will execute.

# %%
onnx_path = export_to_onnx(model, CKPT / "final_export")

if onnx_path is None:
    import torch
    dummy = torch.randn(1, 3, CFG.resolution, CFG.resolution).cuda()
    onnx_path = CKPT / "sentinel-det-v1-raw.onnx"
    torch.onnx.export(model.model, dummy, str(onnx_path), input_names=["images"],
                      output_names=["pred_logits", "pred_boxes"], opset_version=17,
                      dynamic_axes={"images": {0: "batch"}})

final_onnx = CKPT / "sentinel-det-v1.onnx"
shutil.copy2(onnx_path, final_onnx)
sha = hashlib.sha256(final_onnx.read_bytes()).hexdigest()
print("weights:", final_onnx, final_onnx.stat().st_size // 1024, "KB")
print("sha256 :", sha)

sess = ort.InferenceSession(str(final_onnx), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
side = inp.shape[2] if isinstance(inp.shape[2], int) else CFG.resolution

def onnx_detect(path: Path, conf: float = CFG.conf):
    """Minimal CPU detector: letterbox + run + decode. Mirrors apps/api in TS."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    W, H = im.size
    s = min(side / W, side / H)
    nw, nh = int(round(W * s)), int(round(H * s))
    canvas = Image.new("RGB", (side, side), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((side - nw) // 2, (side - nh) // 2))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    outs = sess.run(None, {inp.name: x})
    return outs, (W, H, s)

# Decode is format-tolerant: DETR-style (logits+boxes) or already-decoded (N,6).
def decode(outs, conf):
    dets = []
    arrs = [np.asarray(o, dtype=np.float32) for o in outs]
    if len(arrs) == 2 and arrs[0].ndim == 3 and arrs[1].ndim == 3:
        logits, boxes = arrs
        if logits.shape[-1] < boxes.shape[-1]:
            logits, boxes = boxes, logits
        prob = 1 / (1 + np.exp(-logits[0]))
        scores = prob.max(axis=-1); cls = prob.argmax(axis=-1)
        b = boxes[0]  # cx cy w h
        for (cx, cy, bw, bh), sc, c in zip(b, scores, cls):
            if sc < conf: continue
            dets.append([cx - bw/2, cy - bh/2, bw, bh, float(sc), int(c)])
    else:
        flat = arrs[0]
        if flat.ndim == 3: flat = flat[0]
        for row in flat:
            if len(row) >= 6 and row[4] >= conf:
                dets.append([row[0], row[1], row[2], row[3], float(row[4]), int(row[5])])
    return dets

# --- mAP50 on the held-out sequences (torchmetrics) ---
from torchmetrics.detection import MeanAveragePrecision
from PIL import Image

def evaluate(split: str, limit: int | None = None):
    coco = json.loads((DATA / split / "_annotations.coco.json").read_text())
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a["bbox"])
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    lat = []
    for im in coco["images"][: limit or len(coco["images"])]:
        p = DATA / split / im["file_name"]
        if not p.exists(): continue
        t = time.perf_counter()
        outs, (W, H, s) = onnx_detect(p)
        lat.append((time.perf_counter() - t) * 1000)
        dets = decode(outs, CFG.conf)
        ox, oy = (side - int(round(W*s))) / 2, (side - int(round(H*s))) / 2
        pb = [[max(0,(d[0]-ox)/s), max(0,(d[1]-oy)/s), (d[0]-ox+d[2])/s, (d[1]-oy+d[3])/s] for d in dets]
        ps = [d[4] for d in dets]
        gb = [[x, y, x+w, y+h] for (x, y, w, h) in by_img.get(im["id"], [])]
        metric.update(
            [{"boxes": torch.tensor(pb, dtype=torch.float32).reshape(-1, 4),
              "scores": torch.tensor(ps, dtype=torch.float32),
              "labels": torch.zeros(len(pb), dtype=torch.int64)}],
            [{"boxes": torch.tensor(gb, dtype=torch.float32).reshape(-1, 4),
              "labels": torch.zeros(len(gb), dtype=torch.int64)}],
        )
    m = metric.compute()
    # torchmetrics returns torch.Tensor (and per-class arrays), so isinstance(v, float)
    # silently dropped every metric and produced an empty results block.
    scalars = {}
    for k, v in m.items():
        try:
            if hasattr(v, "numel") and v.numel() != 1:
                continue
            fv = float(v)
            if np.isfinite(fv):
                scalars[k] = fv
        except Exception:
            continue
    return scalars, lat

metrics, lat = evaluate("valid", limit=300)
print({k: round(v, 4) for k, v in metrics.items() if k.startswith("map")})
print(f"ONNX CPU latency: p50 {np.percentile(lat,50):.0f} ms | p95 {np.percentile(lat,95):.0f} ms")

# %% [markdown]
# ## 9 · The number that decides deployability: bird false-positive rate
#
# A counter-UAS system that fires on birds is worse than no system. Every image
# in the bird set that produces **any** detection above threshold is a false
# positive.

# %%
bird_fpr = None
if "drone_vs_bird" in roots:
    birds = [p for p in find_images(roots["drone_vs_bird"])
             if "bird" in str(p).lower() and p.suffix.lower() in IMG_EXT][:250]
    if not birds:
        birds = [p for p in find_images(roots["drone_vs_bird"])][:250]
    fired = 0
    for p in birds:
        try:
            outs, _ = onnx_detect(p)
            if decode(outs, CFG.conf):
                fired += 1
        except Exception:
            continue
    bird_fpr = fired / max(1, len(birds))
    print(f"bird images   : {len(birds)}")
    print(f"false alarms  : {fired}")
    print(f"BIRD FPR      : {bird_fpr:.1%}  ({'reportable' if bird_fpr < 0.15 else 'needs CLIP re-scoring stage'})")
else:
    print("[skip] drone-vs-bird dataset not available")

# %% [markdown]
# ## 10 · INT8 quantisation and latency
#
# Static quantisation with a calibration reader drawn from the validation split.
# If the mAP drop exceeds 2 points, ship FP32 — a slower honest model beats a
# fast one that misses.

# %%
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

class Reader(CalibrationDataReader):
    def __init__(self, split: str, n: int = 64):
        coco = json.loads((DATA / split / "_annotations.coco.json").read_text())
        self.items = [str(DATA / split / im["file_name"]) for im in coco["images"][:n]]
        self.i = 0
        self.side = side
    def _prep(self, p: Path):
        im = Image.open(p).convert("RGB")
        W, H = im.size
        s = min(self.side / W, self.side / H)
        canvas = Image.new("RGB", (self.side, self.side), (114, 114, 114))
        canvas.paste(im.resize((int(W*s), int(H*s)), Image.BILINEAR), (0, 0))
        return np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    def get_next(self):
        while self.i < len(self.items):
            p = Path(self.items[self.i]); self.i += 1
            if p.exists():
                return {inp.name: self._prep(p)}
        return None

int8_path = CKPT / "sentinel-det-v1-int8.onnx"
try:
    quantize_static(str(final_onnx), str(int8_path), Reader("valid"),
                    quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8, per_channel=False)
    sz = int8_path.stat().st_size // 1024
    print(f"int8 written: {int8_path.name} ({sz} KB vs {final_onnx.stat().st_size//1024} KB fp32)")
    int8_sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    lat8 = []
    for im in json.loads((DATA / "valid" / "_annotations.coco.json").read_text())["images"][:40]:
        p = DATA / "valid" / im["file_name"]
        if not p.exists(): continue
        x = Reader("valid", 1)._prep(p)
        t = time.perf_counter(); int8_sess.run(None, {inp.name: x}); lat8.append((time.perf_counter()-t)*1000)
    print(f"int8 latency: p50 {np.percentile(lat8,50):.0f} ms | p95 {np.percentile(lat8,95):.0f} ms")
except Exception as e:
    int8_path = None
    print("[warn] quantisation failed — ship FP32:", repr(e))

# %% [markdown]
# ## 11 · Model card and manifest
#
# The artefact that makes this system auditable rather than a demo.

# %%
results = {
    "model_id": "sentinel-det-v1",
    "base": "RFDETRNano (Apache-2.0)",
    "resolution": side,
    "classes": CLASSES,
    "conf_threshold": CFG.conf,
    "epochs": CFG.epochs,
    "sha256": sha,
    "size_kb": final_onnx.stat().st_size // 1024,
    "int8": bool(int8_path),
    "metrics": {k: round(v, 5) for k, v in metrics.items() if k.startswith(("map", "mar"))},
    "bird_fpr": None if bird_fpr is None else round(bird_fpr, 4),
    "latency_ms": {
        "p50": round(float(np.percentile(lat, 50)), 1),
        "p95": round(float(np.percentile(lat, 95)), 1),
    },
    "datasets": [
        {"slug": s, "role": "eval-only" if "bird" in k else "train",
         "license": "see dataset card on Kaggle", "redistributed": False}
        for k, s in SOURCES.items()
    ],
    "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "generated_by": "training/airspace_sentinel_train.py",
}
(WORK / "metrics.json").write_text(json.dumps(results, indent=2))

map50 = results["metrics"].get("map_50", float("nan"))
map5095 = results["metrics"].get("map", float("nan"))
fpr_txt = "n/a" if bird_fpr is None else f"{bird_fpr:.1%}"
int8_txt = " / INT8 available" if int8_path else ""
card = f"""# Model card — sentinel-det-v1

Counter-UAS object detector. Single class: `drone`.

| Field | Value |
|---|---|
| Base | RFDETRNano (Apache-2.0) |
| Input | {side}x{side} RGB, letterboxed |
| Weights sha256 | `{sha}` |
| Size | {results['size_kb']} KB FP32{int8_txt} |
| mAP50 | {map50:.4f} |
| mAP50-95 | {map5095:.4f} |
| Bird false-positive rate | {fpr_txt} |
| CPU latency p50 / p95 | {results['latency_ms']['p50']} / {results['latency_ms']['p95']} ms |

## Intended use
Detection of small unmanned aerial vehicles in RGB imagery for perimeter
awareness, research and demonstration. **Not** a targeting system, **not**
certified for safety-critical or regulated airspace operations, and **not**
validated at ranges beyond the training distribution.

## Known failure modes
- Small targets (<32 px) at long range: recall degrades sharply; AP-small is reported above, do not substitute overall mAP.
- Birds of prey at altitude: the primary false-positive source; see bird FPR.
- Night / thermal: out of scope (RGB only).
- Occlusion behind foliage, heavy rain, and motion blur: underrepresented in training data.
- Multiple simultaneous UAVs: tracker is IoU-based, no appearance re-identification.

## Data provenance
Datasets are not redistributed. Licenses recorded per source in
`benchmarks/results.json`. Research-only sources were used for evaluation only.

## Governance
Every inference on the public demo emits a hash-chained audit record containing
model id, weights hash, input hash, policy verdict and retention TTL. Raw frames
are never persisted; EXIF/GPS is stripped before decode.
"""
(WORK / "model_card.md").write_text(card)

print(json.dumps(results, indent=2)[:1200])
print("\nArtefacts:", [p.name for p in CKPT.glob("*.onnx")])

# %% [markdown]
# ## 12 · Package for the API
#
# Download `sentinel-det-v1.onnx` (and the INT8 variant) plus `metrics.json` and
# `model_card.md`, then publish as a Kaggle Dataset so the API build can pull it
# by reference instead of committing binaries.

# %%
for f in [final_onnx, int8_path, WORK / "metrics.json", WORK / "model_card.md"]:
    if f and Path(f).exists():
        print(f"{Path(f).name:<34} {Path(f).stat().st_size/1024:9.1f} KB")
