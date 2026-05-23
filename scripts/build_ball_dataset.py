"""
scripts/build_ball_dataset.py

Build a YOLO-format ball detector training dataset from a YouTube playlist.

For each video in the playlist:
  1. Download a short segment (yt-dlp, deleted after processing)
  2. Sample every Nth frame
  3. Run HSV _color_detect on each frame
     - circularity >= AUTO_ACCEPT  → auto-labeled (image + YOLO label saved)
     - AUTO_ACCEPT > circ >= REVIEW → saved to review/ for manual correction
     - no detection                 → saved as a negative (no label file)
  4. Delete the raw video segment

Output layout:
  data/ball_dataset/
    images/train/        ← accepted + negatives
    labels/train/        ← YOLO labels for accepted frames (absent = negative)
    review/images/       ← flagged frames for manual correction
    review/labels/       ← corresponding auto-labels (starting point for correction)
    dataset.yaml         ← ready for `yolo detect train`

Usage
─────
    python scripts/build_ball_dataset.py --playlist <url> [options]

    --playlist URL      YouTube playlist (or single video) URL
    --segment-sec N     Seconds to download per video (default: 300 = 5 min)
    --start-sec N       Start offset within each video in seconds (default: 300)
    --frame-stride N    Sample every Nth frame (default: 3 = ~10fps from 30fps)
    --neg-ratio F       Fraction of no-detection frames to keep as negatives (default: 0.05)
    --max-videos N      Stop after N videos (default: unlimited)
    --out-dir PATH      Dataset root (default: data/ball_dataset)
"""

from __future__ import annotations

import os, sys, types
os.environ.setdefault("OMP_NUM_THREADS",             "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",        "1")
os.environ.setdefault("MKL_NUM_THREADS",             "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK",        "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as _np
from scipy.optimize import linear_sum_assignment as _lsa
def _lapjv(cost, **kw):
    r, c = _lsa(cost.astype(float))
    x = _np.full(cost.shape[0], -1, dtype=_np.int32)
    y = _np.full(cost.shape[1], -1, dtype=_np.int32)
    for ri, ci in zip(r, c): x[ri] = ci; y[ci] = ri
    return 0.0, x, y
_lap = types.ModuleType("lap"); _lap.lapjv = _lapjv; sys.modules["lap"] = _lap

import argparse
import json
import logging
import random
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.ball_detector import BallDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

AUTO_ACCEPT = 0.65   # circularity threshold for auto-labeling
REVIEW_MIN  = 0.55   # minimum circularity to flag for review (below = ignore)


# ── yt-dlp helpers ────────────────────────────────────────────────────────────

def get_playlist_urls(playlist_url: str) -> list[str]:
    """Return individual video URLs from a playlist using yt-dlp."""
    logger.info("Fetching playlist video URLs...")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "-j", "--no-warnings", playlist_url],
        capture_output=True, text=True,
    )
    urls = []
    for line in result.stdout.strip().splitlines():
        try:
            entry = json.loads(line)
            vid_id = entry.get("id") or entry.get("url")
            if vid_id:
                urls.append(f"https://www.youtube.com/watch?v={vid_id}" if not vid_id.startswith("http") else vid_id)
        except json.JSONDecodeError:
            continue
    logger.info("Found %d videos in playlist", len(urls))
    return urls


def download_segment(url: str, start_sec: int, duration_sec: int, out_path: Path) -> bool:
    """Download a time segment of a YouTube video to out_path."""
    cmd = [
        "yt-dlp",
        "--download-sections", f"*{start_sec}-{start_sec + duration_sec}",
        "--force-keyframes-at-cuts",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", str(out_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("yt-dlp failed for %s: %s", url, result.stderr[-300:])
        return False
    return out_path.exists()


# ── Frame processing ──────────────────────────────────────────────────────────

def _yolo_label(det, frame_w: int, frame_h: int) -> str:
    """Convert a Detection bbox to a YOLO label line (class=0)."""
    x1, y1, x2, y2 = det.bbox
    cx = ((x1 + x2) / 2) / frame_w
    cy = ((y1 + y2) / 2) / frame_h
    w  = (x2 - x1) / frame_w
    h  = (y2 - y1) / frame_h
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process_video(
    video_path: Path,
    detector: BallDetector,
    img_train_dir: Path,
    lbl_train_dir: Path,
    rev_img_dir: Path,
    rev_lbl_dir: Path,
    frame_stride: int,
    neg_ratio: float,
    video_tag: str,
) -> dict:
    """Extract frames from video_path, auto-label, and write to dataset dirs."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s", video_path)
        return {}

    counts = {"accepted": 0, "review": 0, "negative": 0, "skipped": 0}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        det = detector._color_detect(frame, w, h)

        stem = f"{video_tag}_f{frame_idx:07d}"

        if det is not None and det.confidence >= AUTO_ACCEPT:
            # Auto-accept
            cv2.imwrite(str(img_train_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (lbl_train_dir / f"{stem}.txt").write_text(_yolo_label(det, w, h))
            counts["accepted"] += 1

        elif det is not None and det.confidence >= REVIEW_MIN:
            # Flag for manual review
            cv2.imwrite(str(rev_img_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (rev_lbl_dir / f"{stem}.txt").write_text(_yolo_label(det, w, h))
            counts["review"] += 1

        else:
            # No detection — keep as negative at neg_ratio sampling rate
            if random.random() < neg_ratio:
                cv2.imwrite(str(img_train_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                # No label file = YOLO treats as negative
                counts["negative"] += 1
            else:
                counts["skipped"] += 1

        frame_idx += 1

    cap.release()
    return counts


# ── Dataset YAML ──────────────────────────────────────────────────────────────

def write_dataset_yaml(out_dir: Path) -> None:
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['ball']\n"
    )
    logger.info("Wrote %s", yaml_path)


def split_val(out_dir: Path, val_frac: float = 0.15) -> None:
    """
    Move val_frac of *labeled* train images (those with a matching .txt) to val.

    Only moves frames that have a ball label, so val mAP is meaningful.
    Safe to call multiple times — already-moved files won't be double-counted.
    """
    img_train = out_dir / "images" / "train"
    lbl_train = out_dir / "labels" / "train"
    img_val   = out_dir / "images" / "val"
    lbl_val   = out_dir / "labels" / "val"
    lbl_val.mkdir(parents=True, exist_ok=True)
    img_val.mkdir(parents=True, exist_ok=True)

    labeled = [p for p in img_train.glob("*.jpg") if (lbl_train / p.with_suffix(".txt").name).exists()]
    random.shuffle(labeled)
    n_move = max(1, int(len(labeled) * val_frac))
    for img_p in labeled[:n_move]:
        lbl_p = lbl_train / img_p.with_suffix(".txt").name
        img_p.rename(img_val / img_p.name)
        lbl_p.rename(lbl_val / lbl_p.name)
    logger.info("Val split: moved %d labeled images to images/val (from %d total labeled)", n_move, len(labeled))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build ball detector training dataset")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--playlist",  help="YouTube playlist or video URL")
    src.add_argument("--local-dir", help="Directory of local .mp4 files to process (skips download)")
    parser.add_argument("--segment-sec", type=int, default=300,  help="[YouTube] Seconds to download per video (default: 300)")
    parser.add_argument("--start-sec",   type=int, default=300,  help="[YouTube] Start offset in seconds (default: 300)")
    parser.add_argument("--frame-stride",type=int, default=3,    help="Sample every Nth frame (default: 3)")
    parser.add_argument("--neg-ratio",   type=float, default=0.05, help="Fraction of no-detection frames to keep as negatives (default: 0.05)")
    parser.add_argument("--max-videos",  type=int, default=0,    help="Stop after N videos (0 = unlimited)")
    parser.add_argument("--out-dir",     default="data/ball_dataset", help="Dataset root directory")
    args = parser.parse_args()

    out_dir      = Path(args.out_dir)
    img_train    = out_dir / "images" / "train"
    lbl_train    = out_dir / "labels" / "train"
    img_val      = out_dir / "images" / "val"
    rev_img      = out_dir / "review"  / "images"
    rev_lbl      = out_dir / "review"  / "labels"

    for d in (img_train, lbl_train, img_val, rev_img, rev_lbl):
        d.mkdir(parents=True, exist_ok=True)

    write_dataset_yaml(out_dir)

    detector = BallDetector.__new__(BallDetector)
    detector._mode   = "color_based"
    detector._model  = None
    detector._MIN_DIM_PX    = BallDetector._MIN_DIM_PX
    detector._MAX_FRAME_PCT = BallDetector._MAX_FRAME_PCT
    detector._MIN_ASPECT    = BallDetector._MIN_ASPECT
    detector._is_plausible  = BallDetector._is_plausible.__get__(detector, BallDetector)

    total = {"accepted": 0, "review": 0, "negative": 0, "skipped": 0}

    if args.local_dir:
        # ── Local .mp4 files ─────────────────────────────────────────────────
        local_paths = sorted(Path(args.local_dir).glob("*.mp4"))
        if not local_paths:
            logger.error("No .mp4 files found in %s", args.local_dir)
            return
        if args.max_videos > 0:
            local_paths = local_paths[:args.max_videos]
        logger.info("Processing %d local clips from %s", len(local_paths), args.local_dir)

        for i, video_path in enumerate(local_paths):
            video_tag = f"local_{video_path.stem}"
            logger.info("[%d/%d] Processing %s", i + 1, len(local_paths), video_path.name)
            counts = process_video(
                video_path, detector,
                img_train, lbl_train,
                rev_img, rev_lbl,
                args.frame_stride, args.neg_ratio,
                video_tag,
            )
            for k in total:
                total[k] += counts.get(k, 0)
            logger.info(
                "  %s: accepted=%d review=%d negative=%d",
                video_path.name, counts.get("accepted", 0),
                counts.get("review", 0), counts.get("negative", 0),
            )

    else:
        # ── YouTube playlist ──────────────────────────────────────────────────
        urls = get_playlist_urls(args.playlist)
        if args.max_videos > 0:
            urls = urls[:args.max_videos]

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, url in enumerate(urls):
                video_tag = f"v{i:04d}"
                tmp_mp4   = Path(tmpdir) / f"{video_tag}.mp4"

                logger.info("[%d/%d] Downloading segment from %s", i + 1, len(urls), url)
                ok = download_segment(url, args.start_sec, args.segment_sec, tmp_mp4)
                if not ok:
                    logger.warning("Skipping %s — download failed", url)
                    continue

                logger.info("Processing %s...", tmp_mp4.name)
                counts = process_video(
                    tmp_mp4, detector,
                    img_train, lbl_train,
                    rev_img, rev_lbl,
                    args.frame_stride, args.neg_ratio,
                    video_tag,
                )
                tmp_mp4.unlink(missing_ok=True)

                for k in total:
                    total[k] += counts.get(k, 0)

                logger.info(
                    "  video %s: accepted=%d review=%d negative=%d",
                    video_tag, counts.get("accepted", 0),
                    counts.get("review", 0), counts.get("negative", 0),
                )

    split_val(out_dir)

    logger.info("─" * 60)
    logger.info("Dataset complete:")
    logger.info("  Auto-labeled (train):  %d", total["accepted"])
    logger.info("  Flagged for review:    %d", total["review"])
    logger.info("  Negatives (train):     %d", total["negative"])
    logger.info("  Output: %s", out_dir.resolve())
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review flagged frames in %s", rev_img)
    logger.info("     Recommended: label-studio start  (pip install label-studio)")
    logger.info("  2. Move corrected labels → images/train + labels/train")
    logger.info("  3. Train:")
    logger.info("     python scripts/train_ball_detector.py --data %s/dataset.yaml --model yolov8s.pt --device cuda --epochs 100 --batch 128", out_dir)


if __name__ == "__main__":
    main()
