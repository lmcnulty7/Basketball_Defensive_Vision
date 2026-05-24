"""
scripts/build_tracknet_dataset.py

Build a TrackNet-format dataset from NBA broadcast footage.

Two source modes
────────────────
  --playlist URL     Download 5-min segments from a YouTube playlist with yt-dlp,
                     process each, then delete.  Disk stays ~flat (one clip at a time).
  --raw-dir PATH     Process .mp4 files already present locally (default: data/raw).

Per-frame pipeline
──────────────────
  Gate 1 — Court visibility: CLIP zero-shot classifier (openai/clip-vit-base-patch32).
            Generalizes across all NBA eras, arenas, and broadcast styles.
            Skips commercials, studio segments, halftime shows.
  Label  — HSV orange detector: visibility=1 + (cx, cy) when ball found,
            visibility=0 otherwise.  This is a label, not a gate — invisible-ball
            frames are saved so the model learns to output zero heatmaps.

Every court-visible frame is saved (not just ball-visible triplets from the old
version), giving proper temporal continuity for 8-frame sequence training.

Output
──────
  data/tracknet_dataset/
    train/<clip_name>/frames/000001.jpg  000002.jpg ...
    train/<clip_name>/labels.csv
    val/<clip_name>/...

labels.csv columns: frame_id, visibility, cx, cy, orig_frame_idx
  - cx / cy : ball center in 288×512 pixel space (0.0 when visibility=0)
  - orig_frame_idx : raw video frame number — used by the DataLoader to detect
                     camera cuts and skip windows that span gaps

Usage
─────
  # A100 playlist mode (500 clips, downloads incrementally):
  python scripts/build_tracknet_dataset.py \\
      --playlist  "https://www.youtube.com/playlist?list=<ID>" \\
      --out-dir   data/tracknet_dataset \\
      --max-videos 500 \\
      --segment-sec 300 \\
      --start-sec   300 \\
      --device      cuda

  # Local raw-dir mode:
  python scripts/build_tracknet_dataset.py \\
      --raw-dir data/raw \\
      --out-dir data/tracknet_dataset \\
      --device  cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TRACKNET_H  = 288
TRACKNET_W  = 512
COURT_CONF  = 0.35   # min CLIP court probability (4-way softmax; random baseline = 0.25)
HSV_CIRC    = 0.65   # min contour circularity for ball
BALL_MIN_PX = 8      # min ball width at original resolution
BALL_MAX_PX = 50     # max ball width at original resolution
FRAME_STRIDE = 3     # sample every Nth raw video frame
VAL_FRAC    = 0.10   # fraction of clips reserved for validation


# ── HSV ball labeler (single frame) ───────────────────────────────────────────

def _hsv_ball_center(
    bgr: np.ndarray,
    min_px: int = BALL_MIN_PX,
    max_px: int = BALL_MAX_PX,
    min_circ: float = HSV_CIRC,
) -> Optional[Tuple[float, float]]:
    """
    Return (cx, cy) of the most circular orange blob in original frame space,
    or None.  Suppresses the bottom 15% (scoreboard) to avoid false positives
    on circular ESPN broadcast graphics.
    """
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo   = np.array([5,  150, 120], dtype=np.uint8)
    hi   = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    # Suppress scoreboard band and outer 3% side margins
    mask[int(h * 0.85):, :]          = 0
    mask[:, :int(w * 0.03)]          = 0
    mask[:, int(w * 0.97):]          = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score, best_center = -1.0, None
    for cnt in contours:
        area  = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if area < 20 or perim == 0:
            continue
        circ = 4.0 * np.pi * area / (perim ** 2)
        if circ < min_circ:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        ball_px = max(bw, bh)
        if ball_px < min_px or ball_px > max_px:
            continue
        if circ > best_score:
            best_score  = circ
            best_center = (x + bw / 2.0, y + bh / 2.0)

    return best_center


# ── Court gate (CLIP zero-shot) ────────────────────────────────────────────────

class _CLIPCourtGate:
    """
    Zero-shot court detector using CLIP text-image similarity.
    Generalizes across all NBA eras, arenas, and broadcast qualities
    without any task-specific training.
    """
    _TEXTS = [
        "an NBA basketball court during a game",
        "a television commercial or advertisement",
        "a sports news studio or analyst desk",
        "an arena crowd with no court visible",
    ]

    def __init__(self, device: str) -> None:
        from transformers import CLIPModel, CLIPProcessor
        self._device = device
        self._model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device).eval()
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("CLIP court gate ready (zero-shot, device=%s)", device)

    def confidence(self, bgr: np.ndarray) -> float:
        """Return probability [0, 1] that frame shows an NBA basketball court."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self._processor(
            text=self._TEXTS, images=pil_img,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            probs = out.logits_per_image.softmax(dim=1)[0]
        return float(probs[0])


def _load_court_detector(weights_path: str, device: str) -> _CLIPCourtGate:
    return _CLIPCourtGate(device)


def _court_confidence(model: _CLIPCourtGate, bgr: np.ndarray, device: str) -> float:
    return model.confidence(bgr)


# ── YouTube download helpers ───────────────────────────────────────────────────

def _get_playlist_urls(
    playlist_url: str,
    max_videos: Optional[int] = None,
) -> List[str]:
    """Return list of video URLs from a YouTube playlist using yt-dlp."""
    cmd = ["yt-dlp", "--flat-playlist", "--dump-json", "--quiet", playlist_url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        logger.error("yt-dlp not found — install with:  pip install yt-dlp")
        raise SystemExit(1)
    except subprocess.TimeoutExpired:
        logger.error("Playlist fetch timed out after 180s")
        raise SystemExit(1)

    urls: List[str] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            vid_id = entry.get("id", "")
            url = entry.get("webpage_url") or (
                f"https://www.youtube.com/watch?v={vid_id}" if vid_id else None
            )
            if url:
                urls.append(url)
        except (json.JSONDecodeError, KeyError):
            continue
        if max_videos and len(urls) >= max_videos:
            break

    logger.info("Playlist: found %d video URLs", len(urls))
    return urls


def _download_segment(
    url: str,
    start_sec: int = 300,
    segment_sec: int = 300,
) -> Optional[Path]:
    """
    Download one segment of a YouTube video into a temp directory.
    Returns path to the downloaded file, or None on failure.
    Caller is responsible for deleting the temp directory.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="tracknet_dl_"))
    out_tmpl = str(tmpdir / "clip.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--quiet",
        "--format",
        "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]"
        "/best[ext=mp4][height<=720]/best",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start_sec}-{start_sec + segment_sec}",
        "--force-keyframes-at-cuts",
        "-o", out_tmpl,
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        logger.warning("Download timed out: %s", url)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None
    except FileNotFoundError:
        logger.error("yt-dlp not found — install with:  pip install yt-dlp")
        raise SystemExit(1)

    if result.returncode != 0:
        err = result.stderr[-300:].decode(errors="replace").strip()
        logger.warning("Download failed (%s): %s", url, err)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    candidates = sorted(tmpdir.iterdir())
    if not candidates:
        logger.warning("No output file after download: %s", url)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    return candidates[0]   # caller cleans up tmpdir


# ── Per-clip processing ────────────────────────────────────────────────────────

def _process_clip(
    video_path: Path,
    out_dir: Path,
    court_model,
    device: str,
    max_frames: int,
) -> int:
    """
    Extract every court-visible frame from one clip.
    Labels each with HSV ball detection (visibility 0 or 1).
    Returns number of frames saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s — skipping", video_path.name)
        return 0

    clip_name  = video_path.stem
    frames_dir = out_dir / clip_name / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / clip_name / "labels.csv"

    orig_h = orig_w = None
    raw_frame_idx = 0   # absolute raw video frame counter
    saved_id      = 0   # index of saved frame (0-based)
    rows: List[Tuple] = []

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        raw_frame_idx += 1
        if raw_frame_idx % FRAME_STRIDE != 0:
            continue

        if orig_h is None:
            orig_h, orig_w = bgr.shape[:2]

        # Gate 1 — court must be visible
        if _court_confidence(court_model, bgr, device) < COURT_CONF:
            continue

        # Label — HSV ball detection (visibility 0 or 1, not a filter)
        center   = _hsv_ball_center(bgr)
        bgr_small = cv2.resize(bgr, (TRACKNET_W, TRACKNET_H))

        if center is not None:
            cx_s = round(center[0] * TRACKNET_W / orig_w, 2)
            cy_s = round(center[1] * TRACKNET_H / orig_h, 2)
            vis  = 1
        else:
            cx_s = cy_s = 0.0
            vis  = 0

        fname = f"{saved_id:06d}.jpg"
        cv2.imwrite(
            str(frames_dir / fname), bgr_small,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        # orig_frame_idx lets the DataLoader detect camera-cut gaps
        rows.append((saved_id, vis, cx_s, cy_s, raw_frame_idx))
        saved_id += 1

        if saved_id % 200 == 0:
            logger.info(
                "  %s: %d frames saved (raw=%d, vis=%d%%)",
                clip_name, saved_id, raw_frame_idx,
                int(100 * sum(r[1] for r in rows) / saved_id),
            )

        if saved_id >= max_frames:
            break

    cap.release()

    if not rows:
        shutil.rmtree(out_dir / clip_name, ignore_errors=True)
        return 0

    with open(label_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "visibility", "cx", "cy", "orig_frame_idx"])
        writer.writerows(rows)

    n_vis = sum(r[1] for r in rows)
    logger.info(
        "  %s → %d frames  (visible=%d  %.0f%%)",
        clip_name, saved_id, n_vis, 100 * n_vis / saved_id,
    )
    return saved_id


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TrackNet basketball dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--playlist",    default=None,
                     help="YouTube playlist URL (yt-dlp download mode)")
    src.add_argument("--raw-dir",     default="data/raw",
                     help="Directory of existing .mp4 clips")

    # Download options (playlist mode only)
    parser.add_argument("--start-sec",   type=int, default=300,
                        help="Seconds into each video to start the segment")
    parser.add_argument("--segment-sec", type=int, default=300,
                        help="Segment length in seconds per video")
    parser.add_argument("--max-videos",  type=int, default=None,
                        help="Stop after N videos (playlist mode)")

    # Output
    parser.add_argument("--out-dir",    default="data/tracknet_dataset")
    parser.add_argument("--max-frames", type=int, default=3000,
                        help="Max frames saved per clip (caps storage per video)")

    # Model / hardware
    parser.add_argument("--court-weights", default=None,
                        help="Ignored — court gate now uses CLIP zero-shot")
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--seed",    type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    court_model = _load_court_detector("", args.device)
    out_dir     = Path(args.out_dir)
    total       = 0

    if args.playlist:
        # ── Playlist mode: download → process → delete ─────────────────────
        urls = _get_playlist_urls(args.playlist, args.max_videos)
        random.shuffle(urls)
        n_val   = max(1, int(len(urls) * VAL_FRAC))
        val_set = set(urls[:n_val])

        for i, url in enumerate(urls, 1):
            split = "val" if url in val_set else "train"
            logger.info("[%d/%d] %s → %s", i, len(urls), url, split)

            video_path = _download_segment(url, args.start_sec, args.segment_sec)
            if video_path is None:
                logger.warning("  Skipped (download failed)")
                continue

            try:
                n = _process_clip(
                    video_path, out_dir / split,
                    court_model, args.device, args.max_frames,
                )
                total += n
            finally:
                shutil.rmtree(video_path.parent, ignore_errors=True)

            logger.info("  Running total: %d frames", total)

    else:
        # ── Raw-dir mode: process existing .mp4 files ──────────────────────
        clips = sorted(Path(args.raw_dir).glob("*.mp4"))
        if not clips:
            logger.error("No .mp4 files found in %s", args.raw_dir)
            raise SystemExit(1)

        random.shuffle(clips)
        n_val   = max(1, int(len(clips) * VAL_FRAC))
        val_set = set(c.name for c in clips[:n_val])

        for clip in clips:
            split = "val" if clip.name in val_set else "train"
            n = _process_clip(
                clip, out_dir / split,
                court_model, args.device, args.max_frames,
            )
            total += n

    logger.info("─" * 60)
    logger.info("Done.  Total frames: %d", total)
    logger.info("Dataset written to: %s", out_dir)
    logger.info("Next step: python scripts/train_tracknet.py --data-dir %s --device %s",
                out_dir, args.device)


if __name__ == "__main__":
    main()
