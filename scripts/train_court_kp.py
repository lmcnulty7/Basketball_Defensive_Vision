"""
scripts/train_court_kp.py

Fine-tune yolov8n-pose on labeled basketball court keypoint images.

Prerequisites
─────────────
  1. Run extract_court_frames.py to pull candidate images.
  2. Label them in Roboflow (Pose Estimation project, 14 keypoints in order).
  3. Export as "YOLOv8 Pose" format and unzip to data/court_kp_dataset/ so:
       data/court_kp_dataset/
         images/train/*.jpg
         images/val/*.jpg
         labels/train/*.txt
         labels/val/*.txt

Usage
─────
  python scripts/train_court_kp.py
  python scripts/train_court_kp.py --epochs 80 --batch 8 --imgsz 1280
  python scripts/train_court_kp.py --resume   # resume from last checkpoint
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_YAML   = ROOT / "configs" / "court_kp_train.yaml"
DATASET_DIR    = ROOT / "data" / "court_kp_dataset"
WEIGHTS_OUT    = ROOT / "models" / "checkpoints" / "court_kp_yolov8n.pt"
BASE_WEIGHTS   = "yolov8n-pose.pt"   # downloaded automatically on first run


def check_dataset():
    """Verify labeled data exists before starting training."""
    train_imgs  = list((DATASET_DIR / "images" / "train").glob("*.jpg"))
    train_lbls  = list((DATASET_DIR / "labels" / "train").glob("*.txt"))
    val_imgs    = list((DATASET_DIR / "images" / "val").glob("*.jpg"))
    val_lbls    = list((DATASET_DIR / "labels" / "val").glob("*.txt"))

    ok = True
    if not train_imgs:
        print("ERROR: No training images found in data/court_kp_dataset/images/train/")
        print("       Run extract_court_frames.py first, then label with Roboflow.")
        ok = False
    if not train_lbls:
        print("ERROR: No training labels found in data/court_kp_dataset/labels/train/")
        print("       Export from Roboflow as 'YOLOv8 Pose' and unzip to data/court_kp_dataset/")
        ok = False
    if not val_imgs or not val_lbls:
        print("WARNING: No validation images/labels found — training will proceed without validation.")

    if ok:
        print(f"Dataset: {len(train_imgs)} train images, {len(train_lbls)} labels")
        print(f"         {len(val_imgs)} val images,   {len(val_lbls)} labels")
    return ok


def train(args):
    from ultralytics import YOLO

    if not args.resume and not check_dataset():
        sys.exit(1)

    model = YOLO(BASE_WEIGHTS)

    results = model.train(
        data      = str(DATASET_YAML),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        patience  = args.patience,
        resume    = args.resume,
        project   = str(ROOT / "models" / "runs"),
        name      = "court_kp",
        # Augmentation — conservative to preserve keypoint accuracy
        hsv_h     = 0.01,
        hsv_s     = 0.4,
        hsv_v     = 0.3,
        degrees   = 0.0,    # no rotation (court orientation matters)
        translate = 0.05,
        scale     = 0.3,
        fliplr    = 0.5,    # horizontal flip uses flip_idx from yaml
        mosaic    = 0.5,
        copy_paste= 0.0,
        # Pose-specific
        pose      = 12.0,   # keypoint loss weight
        kobj      = 2.0,    # keypoint objectness weight
    )

    # Copy best weights to the canonical path the pipeline expects
    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, WEIGHTS_OUT)
        print(f"\nWeights saved to {WEIGHTS_OUT}")
        print("The pipeline will automatically use NeuralKeyDetector on the next run.")
    else:
        print(f"\nWARNING: best.pt not found at {best_pt}")
        print(f"         Manually copy the best checkpoint to {WEIGHTS_OUT}")

    return results


def validate(args):
    """Quick validation pass to check mAP on val set."""
    from ultralytics import YOLO
    if not WEIGHTS_OUT.exists():
        print(f"No weights at {WEIGHTS_OUT} — train first.")
        sys.exit(1)
    model = YOLO(str(WEIGHTS_OUT))
    metrics = model.val(data=str(DATASET_YAML), device=args.device)
    print(metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",   type=int,   default=100)
    ap.add_argument("--imgsz",    type=int,   default=1280,
                    help="Input resolution (1280 recommended for court detail)")
    ap.add_argument("--batch",    type=int,   default=4,
                    help="Batch size (4-8 fits comfortably on 16 GB unified memory)")
    ap.add_argument("--device",   type=str,   default="mps",
                    help="'mps' for Apple Silicon, 'cpu', or '0' for CUDA GPU")
    ap.add_argument("--patience", type=int,   default=20,
                    help="Early-stopping patience (epochs without improvement)")
    ap.add_argument("--resume",   action="store_true",
                    help="Resume from last checkpoint in models/runs/court_kp/")
    ap.add_argument("--validate", action="store_true",
                    help="Skip training; just run validation on existing weights")
    args = ap.parse_args()

    if args.validate:
        validate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
