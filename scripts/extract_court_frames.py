"""
scripts/extract_court_frames.py

Extract candidate frames from existing raw clips for court-keypoint labeling.

The script samples frames at a fixed interval, then uses the ClassicalKeyDetector
as a quick pre-filter to keep only frames where ≥2 court lines are visible.
This increases the "hit rate" when you go to manually label in Roboflow/CVAT.

Output layout
─────────────
  data/court_kp_dataset/
    images/
      train/   ← 85% of frames (random split)
      val/     ← 15% of frames
    labels/    ← empty until you export from your labeling tool

Usage
─────
  python scripts/extract_court_frames.py
  python scripts/extract_court_frames.py --clips data/raw/clip_10m00_18m00.mp4
  python scripts/extract_court_frames.py --every 60 --max-per-clip 40
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.court.keypoint_detector import ClassicalKeyDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR     = ROOT / "data" / "raw"
OUT_BASE    = ROOT / "data" / "court_kp_dataset"
TRAIN_DIR   = OUT_BASE / "images" / "train"
VAL_DIR     = OUT_BASE / "images" / "val"
LABEL_TRAIN = OUT_BASE / "labels" / "train"
LABEL_VAL   = OUT_BASE / "labels" / "val"


def setup_dirs():
    for d in [TRAIN_DIR, VAL_DIR, LABEL_TRAIN, LABEL_VAL]:
        d.mkdir(parents=True, exist_ok=True)


def extract_from_clip(
    clip_path: Path,
    every_n: int,
    max_frames: int,
    detector: ClassicalKeyDetector,
    min_lines: int,
) -> list[np.ndarray]:
    """Return a list of BGR frames that pass the line-visibility filter."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        log.warning("Cannot open %s — skipping", clip_path.name)
        return []

    good = []
    frame_idx = 0
    try:
        while cap.isOpened() and len(good) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n == 0:
                kps = detector.detect(frame)
                n = sum(1 for v in kps.values() if v is not None)
                if n >= min_lines:
                    good.append(frame)
            frame_idx += 1
    finally:
        cap.release()

    log.info("%s: sampled %d candidate frames", clip_path.name, len(good))
    return good


def save_frames(frames: list[np.ndarray], stem: str, val_frac: float = 0.15):
    """Write frames to train/ or val/ with a numbered suffix."""
    random.shuffle(frames)
    n_val = max(1, int(len(frames) * val_frac))
    for i, frame in enumerate(frames):
        dest = VAL_DIR if i < n_val else TRAIN_DIR
        name = f"{stem}_{i:04d}.jpg"
        cv2.imwrite(str(dest / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", help="Specific clip paths (default: all in data/raw/)")
    ap.add_argument("--every", type=int, default=45, help="Sample every N frames (default 45 ≈ 1.5s at 30fps)")
    ap.add_argument("--max-per-clip", type=int, default=35, help="Max frames to keep per clip")
    ap.add_argument("--min-lines", type=int, default=2, help="Min ClassicalKeyDetector detections to keep a frame")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    setup_dirs()

    detector = ClassicalKeyDetector()

    clips = (
        [Path(p) for p in args.clips]
        if args.clips
        else sorted(RAW_DIR.glob("*.mp4"))
    )

    if not clips:
        log.error("No .mp4 files found in %s", RAW_DIR)
        sys.exit(1)

    total = 0
    for clip in clips:
        frames = extract_from_clip(
            clip,
            every_n=args.every,
            max_frames=args.max_per_clip,
            detector=detector,
            min_lines=args.min_lines,
        )
        save_frames(frames, clip.stem, val_frac=args.val_frac)
        total += len(frames)

    n_train = len(list(TRAIN_DIR.glob("*.jpg")))
    n_val   = len(list(VAL_DIR.glob("*.jpg")))
    log.info(
        "Done. %d total frames → %d train / %d val in %s",
        total, n_train, n_val, OUT_BASE,
    )
    log.info("")
    log.info("Next steps:")
    log.info("  1. Upload data/court_kp_dataset/images/ to Roboflow (Pose Estimation project)")
    log.info("     https://app.roboflow.com  →  New Project → Pose Estimation → 14 keypoints")
    log.info("  2. Label each image with the 14 keypoints in this order:")
    log.info("     0 home_paint_bl  1 home_paint_tl  2 home_paint_br  3 home_paint_tr")
    log.info("     4 away_paint_bl  5 away_paint_tl  6 away_paint_br  7 away_paint_tr")
    log.info("     8 half_bot       9 half_top")
    log.info("    10 home_three_bl 11 home_three_tl 12 away_three_bl 13 away_three_tl")
    log.info("  3. Export → YOLOv8 Pose format → download zip → unzip to data/court_kp_dataset/")
    log.info("  4. python scripts/train_court_kp.py")


if __name__ == "__main__":
    main()
