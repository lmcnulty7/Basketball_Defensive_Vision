"""
scripts/train_ball_detector.py

Fine-tune YOLOv8s on the ball dataset built by build_ball_dataset.py.

Usage
─────
    # RTX 4070 (recommended — ~4-6 hrs):
    python scripts/train_ball_detector.py --device cuda --batch 128

    # Apple Silicon (~2-3 days):
    PYTORCH_ENABLE_MPS_FALLBACK=1 \\
        python scripts/train_ball_detector.py --device mps --batch 32

    # Resume interrupted run:
    python scripts/train_ball_detector.py --device cuda --batch 128 --resume

Output
──────
    runs/ball_detector/train/weights/best.pt  ← auto-copied to models/checkpoints/ball_yolo.pt
    runs/ball_detector/train/results.csv      ← per-epoch mAP50, losses
    runs/ball_detector/train/*.png            ← training curves, confusion matrix
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DST = Path("models/checkpoints/ball_yolo.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8s ball detector")
    parser.add_argument("--data",    default="data/ball_dataset_500/dataset.yaml",
                        help="Path to dataset.yaml (default: data/ball_dataset_500/dataset.yaml)")
    parser.add_argument("--model",   default="yolov8s.pt",
                        help="Base model weights (default: yolov8s.pt — better small-object detection than n)")
    parser.add_argument("--device",  default="cuda",
                        help="Training device: cuda | mps | cpu (default: cuda)")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=64,
                        help="Batch size — use 128 for RTX 4070, 32 for MPS (default: 64)")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from runs/ball_detector/train/weights/last.pt")
    parser.add_argument("--no-copy", action="store_true",
                        help="Skip auto-copying best.pt to models/checkpoints/")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset YAML not found: %s", data_path)
        logger.error("Run build_ball_dataset.py first.")
        raise SystemExit(1)

    from ultralytics import YOLO

    if args.resume:
        last_ckpt = Path("runs/ball_detector/train/weights/last.pt")
        if not last_ckpt.exists():
            logger.error("No checkpoint to resume from: %s", last_ckpt)
            raise SystemExit(1)
        model = YOLO(str(last_ckpt))
        logger.info("Resuming from %s", last_ckpt)
    else:
        model = YOLO(args.model)
        logger.info("Starting from %s (ImageNet pretrained)", args.model)

    logger.info(
        "Training on %s  |  epochs=%d  device=%s  batch=%d  imgsz=%d",
        data_path, args.epochs, args.device, args.batch, args.imgsz,
    )

    project_dir = Path("runs/ball_detector").resolve()

    model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name="train",
        exist_ok=args.resume,
        workers=4,

        # ── Augmentation — tuned for broadcast basketball ─────────────────────
        # Basketball at broadcast distance spans 8–40 px in a 1280×720 frame.
        # Standard augmentation for small objects:
        mosaic=1.0,            # mosaic on (helps small-object detection)
        close_mosaic=10,       # disable mosaic in last 10 epochs for fine-tuning stability
        mixup=0.05,            # light mixup across arena types
        scale=0.4,             # scale jitter ±40% — ball shouldn't shrink to noise
        fliplr=0.5,            # horizontal flip valid (courts are symmetric)
        flipud=0.0,            # never flip vertically — camera always above court
        degrees=0.0,           # no rotation — broadcast cameras don't tilt
        translate=0.1,         # mild translation
        perspective=0.0003,    # very mild perspective warp (simulates slight angle changes)
        hsv_h=0.015,           # hue jitter — accounts for arena lighting color differences
        hsv_s=0.7,             # saturation jitter — TV vs. streaming exposure
        hsv_v=0.4,             # brightness jitter — night vs. day arena lighting

        # ── Single-class settings ─────────────────────────────────────────────
        single_cls=True,       # one class (ball) — simplifies classification head

        # ── Training stability ────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,              # final lr = lr0 * lrf (cosine schedule)
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        patience=20,           # early stopping if val mAP doesn't improve for 20 epochs

        # ── Checkpointing / validation ────────────────────────────────────────
        save_period=10,        # save checkpoint every 10 epochs
        val=True,
        plots=True,            # save training curves + confusion matrix
    )

    best = project_dir / "train" / "weights" / "best.pt"
    if not args.no_copy and best.exists():
        CHECKPOINT_DST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, CHECKPOINT_DST)
        logger.info("─" * 60)
        logger.info("Best weights saved → %s", CHECKPOINT_DST)
        logger.info("Pipeline will automatically load these on the next run.")
        logger.info("Check val mAP50 in runs/ball_detector/train/results.csv")
        logger.info("Target: mAP50 > 0.40 before using in production.")
    elif not best.exists():
        logger.warning("best.pt not found — training may have failed or been interrupted.")
        logger.warning("To resume: python scripts/train_ball_detector.py --resume --device %s", args.device)


if __name__ == "__main__":
    main()
