
import os, sys, json, random, shutil, time, warnings
from copy import deepcopy
from pathlib import Path
from IPython.display import display

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").exists()
WORKDIR = Path("/content/dfire_buoi4" if IN_COLAB else "/kaggle/working/dfire_buoi4" if IN_KAGGLE else "./dfire_buoi4_work").resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(WORKDIR / "yolo_config")

RUN_MODE = "full"  # @param ["smoke_test", "demo", "full"]
RUN_TRAIN = True  # @param {type:"boolean"}
RUN_COCO_SIZE_EVAL = True  # @param {type:"boolean"}
EXPERIMENTS_TO_RUN = ["baseline", "multiscale", "attention", "combined"]
SEEDS = [42]
IMGSZ = 640
BATCH = 16
WORKERS = 2

MODES = {
    "smoke_test": dict(epochs=1, fraction=0.02, patience=1),
    "demo": dict(epochs=12, fraction=0.20, patience=5),
    "full": dict(epochs=80, fraction=1.00, patience=20),
}
CFG = MODES[RUN_MODE]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch, yaml, ultralytics
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

random.seed(SEEDS[0]); np.random.seed(SEEDS[0]); torch.manual_seed(SEEDS[0])
DEVICE = 0 if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    warnings.warn("Không có GPU. Hãy bật GPU trong phần Accelerator.")
else:
    capability = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = torch.cuda.get_arch_list()
    if compiled_arches and required_arch not in compiled_arches:
        raise RuntimeError(
            f"PyTorch {torch.__version__} không hỗ trợ GPU {torch.cuda.get_device_name(0)} ({required_arch}). "
            "Trên Kaggle hãy chọn Settings > Accelerator > GPU T4 x2 rồi Restart Session."
        )
platform_name = "colab" if IN_COLAB else "kaggle" if IN_KAGGLE else "local"
print({"platform": platform_name, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
       "gpu": torch.cuda.get_device_name(0) if DEVICE != "cpu" else None,
       "device": str(DEVICE), "mode": RUN_MODE, **CFG})

DATASET_SLUG = "sayedgamal99/smoke-fire-detection-yolo"
MANUAL_DATA_ROOT = ""  # @param {type:"string"}
VAL_RATIO_IF_MISSING = 0.18
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if MANUAL_DATA_ROOT:
    RAW_ROOT = Path(MANUAL_DATA_ROOT).expanduser().resolve()
else:
    dataset_name = DATASET_SLUG.split("/")[-1]
    kaggle_input_dir = Path(f"/kaggle/input/{dataset_name}")
    
    if IN_KAGGLE and kaggle_input_dir.exists():
        RAW_ROOT = kaggle_input_dir.resolve()
    else:
        import kagglehub
        try:
            if IN_COLAB:
                from google.colab import userdata
                token = userdata.get("KAGGLE_API_TOKEN")
                if token:
                    os.environ["KAGGLE_API_TOKEN"] = token
        except Exception:
            pass
            
        try:
            RAW_ROOT = Path(kagglehub.dataset_download(DATASET_SLUG)).resolve()
        except Exception as e:
            if IN_KAGGLE:
                raise RuntimeError(
                    f"Kagglehub lỗi: {e}\n\n"
                    f"-> TRÊN KAGGLE: Bạn phải ấn nút 'Add Data' (hoặc 'Add Input') ở thanh bên phải, "
                    f"tìm dataset '{DATASET_SLUG}' và thêm vào notebook trước khi chạy (đặc biệt khi Save & Run All)."
                ) from e
            raise

def find_split(root, aliases):
    aliases = {x.lower() for x in aliases}
    candidates = []
    for images_dir in root.rglob("images"):
        if images_dir.parent.name.lower() in aliases and (images_dir.parent / "labels").is_dir():
            candidates.append(images_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda p: sum(1 for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS))

train_dir = find_split(RAW_ROOT, {"train", "training"})
val_dir = find_split(RAW_ROOT, {"val", "valid", "validation"})
test_dir = find_split(RAW_ROOT, {"test", "testing"})
if train_dir is None or test_dir is None:
    raise FileNotFoundError(f"Không tìm thấy train/test images + labels trong {RAW_ROOT}")

def sanitize_label_text(text, source):
    output, changed, removed = [], 0, 0
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{source}:{line_no} phải có 5 cột")
        try:
            cls_value, x, y, w, h = map(float, parts)
        except ValueError as exc:
            raise ValueError(f"{source}:{line_no} chứa giá trị không phải số") from exc
        if not all(np.isfinite(z) for z in (cls_value, x, y, w, h)):
            raise ValueError(f"{source}:{line_no} chứa NaN/Inf")
        cls = int(cls_value)
        if cls_value != cls or cls not in (0, 1):
            raise ValueError(f"{source}:{line_no} có class id không hợp lệ: {parts[0]}")
        if w <= 0 or h <= 0:
            changed += 1; removed += 1
            continue
        if all(0.0 <= z <= 1.0 for z in (x, y, w, h)):
            cleaned = line.strip() if parts[0] == str(cls) else f"{cls} {x:.12g} {y:.12g} {w:.12g} {h:.12g}"
        else:
            xmin, ymin = max(0.0, x-w/2), max(0.0, y-h/2)
            xmax, ymax = min(1.0, x+w/2), min(1.0, y+h/2)
            w, h = xmax-xmin, ymax-ymin
            if w <= 0 or h <= 0:
                changed += 1; removed += 1
                continue
            x, y = xmin+w/2, ymin+h/2
            cleaned = f"{cls} {x:.12g} {y:.12g} {w:.12g} {h:.12g}"
        changed += int(cleaned != line.strip())
        output.append(cleaned)
    return "\n".join(output) + ("\n" if output else ""), changed, removed

sanitize_rows = []
def prepare_clean_split(source_images, split_name):
    source_images = source_images.resolve()
    source_labels = source_images.parent / "labels"
    target_split = WORKDIR / "data" / split_name
    target_images, target_labels = target_split / "images", target_split / "labels"
    target_split.mkdir(parents=True, exist_ok=True)
    if target_images.is_symlink() and target_images.resolve() != source_images:
        target_images.unlink()
    link_mode = "symlink"
    if not target_images.exists():
        try:
            target_images.symlink_to(source_images, target_is_directory=True)
        except OSError:
            target_images.mkdir(parents=True, exist_ok=True)
    if not target_images.is_symlink():
        link_mode = "hardlink"
        for source_image in source_images.iterdir():
            if not source_image.is_file() or source_image.suffix.lower() not in IMAGE_EXTS:
                continue
            target_image = target_images / source_image.name
            if target_image.exists():
                continue
            try:
                os.link(source_image, target_image)
            except OSError:
                shutil.copy2(source_image, target_image); link_mode = "copy"
    target_labels.mkdir(parents=True, exist_ok=True)
    files_changed = lines_changed = lines_removed = 0
    for source_label in sorted(source_labels.glob("*.txt")):
        original = source_label.read_text(encoding="utf-8")
        cleaned, changed, removed = sanitize_label_text(original, source_label)
        (target_labels / source_label.name).write_text(cleaned, encoding="utf-8")
        files_changed += int(changed > 0); lines_changed += changed; lines_removed += removed
    sanitize_rows.append({"split": split_name, "image_mode": link_mode, "files_changed": files_changed,
                          "lines_changed": lines_changed, "lines_removed": lines_removed})
    return target_images

def image_paths(images_dir):
    if images_dir is None:
        return []
    return sorted(p.absolute() for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

clean_train_dir = prepare_clean_split(train_dir, "train")
clean_val_dir = prepare_clean_split(val_dir, "val") if val_dir is not None else None
clean_test_dir = prepare_clean_split(test_dir, "test")
display(pd.DataFrame(sanitize_rows))

all_train = image_paths(clean_train_dir)
test_paths = image_paths(clean_test_dir)
if val_dir is not None:
    train_paths, val_paths = all_train, image_paths(clean_val_dir)
else:
    shuffled = all_train.copy()
    random.Random(SEEDS[0]).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * VAL_RATIO_IF_MISSING))
    val_paths, train_paths = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])

split_dir = WORKDIR / "splits"
split_dir.mkdir(parents=True, exist_ok=True)
for name, paths in {"train": train_paths, "val": val_paths, "test": test_paths}.items():
    (split_dir / f"{name}.txt").write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")

data_yaml = WORKDIR / "dfire.yaml"
data_yaml.write_text(yaml.safe_dump({
    "path": str(WORKDIR),
    "train": str(split_dir / "train.txt"),
    "val": str(split_dir / "val.txt"),
    "test": str(split_dir / "test.txt"),
    "names": {0: "smoke", 1: "fire"},
}, sort_keys=False), encoding="utf-8")

assert set(train_paths).isdisjoint(val_paths)
assert set(train_paths).isdisjoint(test_paths)
assert set(val_paths).isdisjoint(test_paths)
print({"root": str(RAW_ROOT), "train": len(train_paths), "val": len(val_paths), "test": len(test_paths)})
print(data_yaml.read_text())

CLASS_NAMES = {0: "smoke", 1: "fire"}

def label_path(image_path):
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"

def read_labels(image_path):
    path = label_path(image_path)
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no} phải có 5 cột")
        cls, x, y, w, h = map(float, parts)
        if int(cls) not in CLASS_NAMES or not all(0 <= z <= 1 for z in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Nhãn không hợp lệ tại {path}:{line_no}")
        rows.append((int(cls), x, y, w, h))
    return rows

records, empty_counts = [], {}
for split_name, paths in {"train": train_paths, "val": val_paths, "test": test_paths}.items():
    empty_counts[split_name] = 0
    for image_path in paths:
        labels = read_labels(image_path)
        empty_counts[split_name] += int(not labels)
        for cls, x, y, w, h in labels:
            area_at_640 = w * h * IMGSZ * IMGSZ
            size = "small" if area_at_640 < 32**2 else "medium" if area_at_640 < 96**2 else "large"
            records.append({"split": split_name, "image": str(image_path), "class_id": cls,
                            "class": CLASS_NAMES[cls], "x": x, "y": y, "w": w, "h": h,
                            "area_at_640": area_at_640, "size_at_640": size})
boxes = pd.DataFrame(records)
display(pd.crosstab(boxes["split"], boxes["class"]).join(pd.Series(empty_counts, name="empty_images")))
display(pd.crosstab(boxes["class"], boxes["size_at_640"]).reindex(columns=["small", "medium", "large"], fill_value=0))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=boxes, x="class", hue="split", ax=axes[0])
sns.countplot(data=boxes, x="size_at_640", hue="class", order=["small", "medium", "large"], ax=axes[1])
axes[0].set_title("Phân bố lớp"); axes[1].set_title("Kích thước bbox quy đổi tại 640")
plt.tight_layout(); plt.show()

def draw_ground_truth(image_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    W, H = image.size
    colors = {0: "deepskyblue", 1: "red"}
    for cls, x, y, w, h in read_labels(image_path):
        x1, y1, x2, y2 = (x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H
        draw.rectangle((x1, y1, x2, y2), outline=colors[cls], width=max(2, W//300))
        draw.text((x1+3, y1+3), CLASS_NAMES[cls], fill=colors[cls])
    return image

sample_pool = boxes.query("split == 'train' and size_at_640 == 'small'")["image"].drop_duplicates().tolist()
sample_paths = random.Random(SEEDS[0]).sample(sample_pool, min(6, len(sample_pool)))
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, path in zip(axes.flat, sample_paths):
    ax.imshow(draw_ground_truth(Path(path))); ax.axis("off")
for ax in axes.flat[len(sample_paths):]: ax.axis("off")
plt.tight_layout(); plt.show()

BACKBONE = [
    [-1, 1, "Conv", [64, 3, 2]],
    [-1, 1, "Conv", [128, 3, 2]],
    [-1, 2, "C3k2", [256, False, 0.25]],
    [-1, 1, "Conv", [256, 3, 2]],
    [-1, 2, "C3k2", [512, False, 0.25]],
    [-1, 1, "Conv", [512, 3, 2]],
    [-1, 2, "C3k2", [512, True]],
    [-1, 1, "Conv", [1024, 3, 2]],
    [-1, 2, "C3k2", [1024, True]],
    [-1, 1, "SPPF", [1024, 5]],
    [-1, 2, "C2PSA", [1024]],
]

HEAD_P3 = [
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
    [[-1, 6], 1, "Concat", [1]],
    [-1, 2, "C3k2", [512, False]],
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
    [[-1, 4], 1, "Concat", [1]],
    [-1, 2, "C3k2", [256, False]],
    [-1, 1, "Conv", [256, 3, 2]],
    [[-1, 13], 1, "Concat", [1]],
    [-1, 2, "C3k2", [512, False]],
    [-1, 1, "Conv", [512, 3, 2]],
    [[-1, 10], 1, "Concat", [1]],
    [-1, 2, "C3k2", [1024, True]],
    [[16, 19, 22], 1, "Detect", [2]],
]

HEAD_P2 = [
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
    [[-1, 6], 1, "Concat", [1]],
    [-1, 2, "C3k2", [512, False]],
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
    [[-1, 4], 1, "Concat", [1]],
    [-1, 2, "C3k2", [256, False]],
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
    [[-1, 2], 1, "Concat", [1]],
    [-1, 2, "C3k2", [128, False]],
    [-1, 1, "Conv", [128, 3, 2]],
    [[-1, 16], 1, "Concat", [1]],
    [-1, 2, "C3k2", [256, False]],
    [-1, 1, "Conv", [256, 3, 2]],
    [[-1, 13], 1, "Concat", [1]],
    [-1, 2, "C3k2", [512, False]],
    [-1, 1, "Conv", [512, 3, 2]],
    [[-1, 10], 1, "Concat", [1]],
    [-1, 2, "C3k2", [1024, True]],
    [[19, 22, 25, 28], 1, "Detect", [2]],
]

def make_model_config(use_p2, use_attention):
    backbone = deepcopy(BACKBONE)
    if not use_attention:
        backbone[10] = [-1, 2, "C3k2", [1024, True]]
    return {"nc": 2, "scale": "n", "scales": {"n": [0.50, 0.25, 1024]},
            "backbone": backbone, "head": deepcopy(HEAD_P2 if use_p2 else HEAD_P3)}

MODEL_DIR = WORKDIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS = {
    "baseline":   dict(use_p2=False, use_attention=False),
    "multiscale": dict(use_p2=True,  use_attention=False),
    "attention":  dict(use_p2=False, use_attention=True),
    "combined":   dict(use_p2=True,  use_attention=True),
}
for name, factors in EXPERIMENTS.items():
    path = MODEL_DIR / f"yolo11n_{name}.yaml"
    path.write_text(yaml.safe_dump(make_model_config(**factors), sort_keys=False), encoding="utf-8")
    factors["yaml"] = path

architecture_rows = []
for name, cfg in EXPERIMENTS.items():
    model = YOLO(str(cfg["yaml"]))
    params = sum(p.numel() for p in model.model.parameters())
    architecture_rows.append({"experiment": name, "P2": cfg["use_p2"],
                              "PSA": cfg["use_attention"],
                              "strides": model.model.model[-1].stride.tolist(),
                              "params_M": params / 1e6, "GFLOPs": get_flops(model.model, imgsz=IMGSZ)})
    del model
architecture_df = pd.DataFrame(architecture_rows)
display(architecture_df)

RUNS_DIR = WORKDIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
run_manifest = []

common_train_args = dict(
    data=str(data_yaml), imgsz=IMGSZ, epochs=CFG["epochs"], fraction=CFG["fraction"],
    batch=BATCH, device=DEVICE, workers=WORKERS, optimizer="AdamW", lr0=0.001, lrf=0.01,
    cos_lr=True, patience=CFG["patience"], amp=True, deterministic=True,
    mosaic=1.0, close_mosaic=min(5, max(0, CFG["epochs"] // 4)),
    degrees=5.0, translate=0.10, scale=0.40, fliplr=0.5, multi_scale=False,
    cache=False, plots=True, val=True, pretrained=False, project=str(RUNS_DIR), exist_ok=True,
)

if RUN_TRAIN:
    for experiment in EXPERIMENTS_TO_RUN:
        for seed in SEEDS:
            run_name = f"{experiment}_{RUN_MODE}_seed{seed}"
            best_path = RUNS_DIR / run_name / "weights" / "best.pt"
            started = time.time()
            if not best_path.exists():
                model = YOLO(str(EXPERIMENTS[experiment]["yaml"]))
                model.train(name=run_name, seed=seed, **common_train_args)
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            if not best_path.exists():
                raise FileNotFoundError(f"Thiếu checkpoint: {best_path}")
            run_manifest.append({"experiment": experiment, "seed": seed,
                                 "weights": str(best_path), "train_minutes_this_session": (time.time()-started)/60})
else:
    for experiment in EXPERIMENTS_TO_RUN:
        for seed in SEEDS:
            best_path = RUNS_DIR / f"{experiment}_{RUN_MODE}_seed{seed}" / "weights" / "best.pt"
            if best_path.exists():
                run_manifest.append({"experiment": experiment, "seed": seed,
                                     "weights": str(best_path), "train_minutes_this_session": 0.0})

manifest_df = pd.DataFrame(run_manifest)
if manifest_df.empty:
    raise RuntimeError("Không có checkpoint. Bật RUN_TRAIN hoặc chép weights vào đúng thư mục runs.")
display(manifest_df)

eval_rows, class_rows, val_dirs = [], [], {}
for row in run_manifest:
    experiment, seed = row["experiment"], row["seed"]
    model = YOLO(row["weights"])
    metrics = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH,
                        device=DEVICE, workers=WORKERS, conf=0.001, iou=0.7, max_det=300,
                        plots=True, project=str(RUNS_DIR / "test_eval"),
                        name=f"{experiment}_{RUN_MODE}_seed{seed}", exist_ok=True, verbose=False)
    params = sum(p.numel() for p in model.model.parameters())
    val_dirs[(experiment, seed)] = Path(metrics.save_dir)
    eval_rows.append({
        "experiment": experiment, "seed": seed, "precision": metrics.box.mp,
        "recall": metrics.box.mr, "mAP50": metrics.box.map50,
        "mAP75": metrics.box.map75, "mAP50_95": metrics.box.map,
        "inference_ms": metrics.speed.get("inference", np.nan),
        "params_M": params / 1e6, "GFLOPs": get_flops(model.model, imgsz=IMGSZ),
    })
    for cls_id, cls_map in enumerate(metrics.box.maps):
        class_rows.append({"experiment": experiment, "seed": seed,
                           "class": CLASS_NAMES[cls_id], "mAP50_95": cls_map})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

eval_df = pd.DataFrame(eval_rows)
class_df = pd.DataFrame(class_rows)
display(eval_df.sort_values("mAP50_95", ascending=False).style.format(precision=4))
display(class_df.pivot_table(index=["experiment", "seed"], columns="class", values="mAP50_95").style.format(precision=4))

import contextlib, io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def build_coco_ground_truth(paths):
    dataset = {"info": {}, "licenses": [], "images": [], "annotations": [],
               "categories": [{"id": i+1, "name": name} for i, name in CLASS_NAMES.items()]}
    path_to_id, ann_id = {}, 1
    for image_id, path in enumerate(paths, 1):
        with Image.open(path) as im: W, H = im.size
        path_to_id[str(path.resolve())] = image_id
        dataset["images"].append({"id": image_id, "file_name": str(path), "width": W, "height": H})
        for cls, x, y, w, h in read_labels(path):
            bw, bh = w*W, h*H
            dataset["annotations"].append({"id": ann_id, "image_id": image_id,
                "category_id": cls+1, "bbox": [(x-w/2)*W, (y-h/2)*H, bw, bh],
                "area": bw*bh, "iscrowd": 0})
            ann_id += 1
    coco = COCO()
    coco.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()): coco.createIndex()
    return coco, path_to_id

def predict_coco(model, paths, path_to_id):
    predictions = []

    CHUNK_SIZE = 128

    for start in range(0, len(paths), CHUNK_SIZE):
        chunk_paths = paths[start:start + CHUNK_SIZE]

        stream = model.predict(
            source=[str(p) for p in chunk_paths],
            stream=True,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            conf=0.001,
            iou=0.7,
            max_det=300,
            verbose=False
        )

        for result in stream:
            image_id = path_to_id[str(Path(result.path).resolve())]

            if result.boxes is None:
                continue

            for xyxy, score, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy()
            ):
                x1, y1, x2, y2 = xyxy.tolist()

                predictions.append({
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "score": float(score)
                })

        del stream

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return predictions

def coco_metrics(coco_gt, predictions, cat_ids=None):
    if not predictions:
        return {k: 0.0 for k in ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR100"]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        if cat_ids is not None: evaluator.params.catIds = cat_ids
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    s = evaluator.stats
    return dict(AP=s[0], AP50=s[1], AP75=s[2], AP_small=s[3], AP_medium=s[4], AP_large=s[5], AR100=s[8])

size_rows = []
if RUN_COCO_SIZE_EVAL:
    coco_gt, path_to_id = build_coco_ground_truth(test_paths)
    pred_dir = WORKDIR / "coco_predictions"
    pred_dir.mkdir(exist_ok=True)
    for row in run_manifest:
        experiment, seed = row["experiment"], row["seed"]
        pred_file = pred_dir / f"{experiment}_{RUN_MODE}_seed{seed}.json"
        if pred_file.exists():
            predictions = json.loads(pred_file.read_text(encoding="utf-8"))
        else:
            predictions = predict_coco(YOLO(row["weights"]), test_paths, path_to_id)
            pred_file.write_text(json.dumps(predictions), encoding="utf-8")
        size_rows.append({"experiment": experiment, "seed": seed, "class": "all",
                          **coco_metrics(coco_gt, predictions)})
        for cls_id, cls_name in CLASS_NAMES.items():
            size_rows.append({"experiment": experiment, "seed": seed, "class": cls_name,
                              **coco_metrics(coco_gt, predictions, [cls_id+1])})
size_df = pd.DataFrame(size_rows)
if not size_df.empty:
    display(size_df.query("`class` == 'all'").sort_values("AP", ascending=False).style.format(precision=4))

order = ["baseline", "multiscale", "attention", "combined"]
summary = eval_df.groupby("experiment", as_index=False).agg(
    mAP50_95=("mAP50_95", "mean"), mAP50=("mAP50", "mean"),
    precision=("precision", "mean"), recall=("recall", "mean"),
    inference_ms=("inference_ms", "mean"), params_M=("params_M", "mean"), GFLOPs=("GFLOPs", "mean"))
if not size_df.empty:
    size_summary = size_df.query("`class` == 'all'").groupby("experiment", as_index=False)[["AP_small", "AP_medium", "AP_large"]].mean()
    summary = summary.merge(size_summary, on="experiment", how="left")
summary["experiment"] = pd.Categorical(summary["experiment"], order, ordered=True)
summary = summary.sort_values("experiment").reset_index(drop=True)
base = summary.loc[summary["experiment"] == "baseline", "mAP50_95"]
summary["delta_mAP50_95"] = summary["mAP50_95"] - (base.iloc[0] if len(base) else np.nan)
display(summary.style.format(precision=4))

metric_candidates = [c for c in ["mAP50_95", "mAP50", "AP_small"] if c in summary and summary[c].notna().any()]
lookup = summary.set_index("experiment")
effects = []
if set(order).issubset(set(lookup.index.astype(str))):
    for metric in metric_candidates:
        b, m, a, c = [float(lookup.loc[x, metric]) for x in order]
        effects.append({"metric": metric, "P2 effect": m-b, "PSA effect": a-b,
                        "combined effect": c-b, "interaction": c-m-a+b})
effects_df = pd.DataFrame(effects)
display(effects_df.style.format(precision=4))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.barplot(data=summary, x="experiment", y="mAP50_95", ax=axes[0])
if "AP_small" in summary: sns.barplot(data=summary, x="experiment", y="AP_small", ax=axes[1])
sns.barplot(data=summary, x="experiment", y="inference_ms", ax=axes[2])
axes[0].set_title("mAP50-95"); axes[1].set_title("COCO AP small"); axes[2].set_title("Inference ms/ảnh")
for ax in axes: ax.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.show()

best_experiment = str(summary.sort_values("mAP50_95", ascending=False).iloc[0]["experiment"])
best_row = next(r for r in run_manifest if r["experiment"] == best_experiment)
best_model = YOLO(best_row["weights"])

hard_pool = boxes.query("split == 'test' and (size_at_640 == 'small' or `class` == 'smoke')")["image"].drop_duplicates().tolist()
hard_samples = random.Random(SEEDS[0]).sample(hard_pool, min(9, len(hard_pool)))
predictions = best_model.predict(hard_samples, imgsz=IMGSZ, conf=0.25, iou=0.7, device=DEVICE, verbose=False)
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for ax, result in zip(axes.flat, predictions):
    ax.imshow(result.plot()[:, :, ::-1]); ax.axis("off"); ax.set_title(Path(result.path).name)
for ax in axes.flat[len(predictions):]: ax.axis("off")
plt.suptitle(f"Best: {best_experiment}"); plt.tight_layout(); plt.show()

plot_names = ["confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"]
plot_dir = val_dirs[(best_experiment, best_row["seed"])]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, name in zip(axes, plot_names):
    path = plot_dir / name
    if path.exists(): ax.imshow(Image.open(path))
    ax.axis("off"); ax.set_title(name)
plt.tight_layout(); plt.show()

SAVE_TO_DRIVE = False  # @param {type:"boolean"}
DOWNLOAD_ZIP = False  # @param {type:"boolean"}

REPORT_DIR = WORKDIR / "report"
REPORT_DIR.mkdir(exist_ok=True)
architecture_df.to_csv(REPORT_DIR / "architectures.csv", index=False)
manifest_df.to_csv(REPORT_DIR / "run_manifest.csv", index=False)
eval_df.to_csv(REPORT_DIR / "test_metrics.csv", index=False)
class_df.to_csv(REPORT_DIR / "per_class_metrics.csv", index=False)
summary.to_csv(REPORT_DIR / "ablation_summary.csv", index=False)
effects_df.to_csv(REPORT_DIR / "factorial_effects.csv", index=False)
if not size_df.empty: size_df.to_csv(REPORT_DIR / "coco_size_metrics.csv", index=False)

best_copy = REPORT_DIR / f"best_{best_experiment}.pt"
shutil.copy2(best_row["weights"], best_copy)
archive = shutil.make_archive(str(WORKDIR / "dfire_buoi4_results"), "zip", root_dir=WORKDIR,
                              base_dir="report")
print("Báo cáo:", REPORT_DIR)
print("Checkpoint tốt nhất:", best_copy)
print("ZIP:", archive)

if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    target = Path("/content/drive/MyDrive/DFire_Buoi4")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, target / Path(archive).name)
    shutil.copy2(best_copy, target / best_copy.name)
if IN_COLAB and DOWNLOAD_ZIP:
    from google.colab import files
    files.download(archive)

