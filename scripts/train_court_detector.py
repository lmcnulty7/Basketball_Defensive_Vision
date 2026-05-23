"""
scripts/train_court_detector.py

Train a YOLOv8n bounding-box detector to locate the basketball court in
broadcast frames.  Uses the 850-image Roboflow detection dataset.

The trained model is used by CourtDetector (src/court/court_detector.py) to
crop frames before running ClassicalKeyDetector, improving keypoint hit rate.

Usage
─────
  python scripts/train_court_detector.py
  python scripts/train_court_detector.py --epochs 60 --batch 16
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_ROOT  = ROOT / "data" / "export_dataset_ YOLOv8" / "basketball_court_dataset"
DATASET_YAML  = ROOT / "configs" / "court_detect_train.yaml"
WEIGHTS_OUT   = ROOT / "models" / "checkpoints" / "court_detector_yolov8n.pt"
BASE_WEIGHTS  = "yolov8n.pt"
VAL_FRAC      = 0.15


def make_val_split():
    """Move 15% of training images/labels into a val/ split."""
    train_img_dir = DATASET_ROOT / "train" / "images"
    train_lbl_dir = DATASET_ROOT / "train" / "labels"
    val_img_dir   = DATASET_ROOT / "val"   / "images"
    val_lbl_dir   = DATASET_ROOT / "val"   / "labels"

    if val_img_dir.exists() and any(val_img_dir.iterdir()):
        print(f"Val split already exists ({len(list(val_img_dir.glob('*')))} images) — skipping.")
        return

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(train_img_dir.glob("*.jpg")) + sorted(train_img_dir.glob("*.png"))
    random.shuffle(images)
    n_val = max(1, int(len(images) * VAL_FRAC))
    val_images = images[:n_val]

    for img in val_images:
        lbl = train_lbl_dir / (img.stem + ".txt")
        shutil.move(str(img), val_img_dir / img.name)
        if lbl.exists():
            shutil.move(str(lbl), val_lbl_dir / lbl.name)

    print(f"Val split: moved {n_val}/{len(images)} images to val/")


def train(args):
    make_val_split()

    from ultralytics import YOLO
    model = YOLO(BASE_WEIGHTS)

    results = model.train(
        data      = str(DATASET_YAML),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        patience  = 15,
        project   = str(ROOT / "models" / "runs"),
        name      = "court_detector",
        # Standard detection augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, translate=0.1, scale=0.5, mosaic=1.0,
    )

    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, WEIGHTS_OUT)
        print(f"\nWeights saved to {WEIGHTS_OUT}")
        print("CourtDetector will automatically use these on the next pipeline run.")
    else:
        print(f"WARNING: best.pt not found at {best_pt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int,   default=50)
    ap.add_argument("--imgsz",  type=int,   default=640)
    ap.add_argument("--batch",  type=int,   default=16)
    ap.add_argument("--device", type=str,   default="mps")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
