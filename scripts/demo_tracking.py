"""
scripts/demo_tracking.py

End-to-end demo of the tracking pipeline: MultiTracker + TrackManager.

Simulates a 15-frame clip by panning the bus.jpg sample image 3px/frame
(mimicking a slow broadcast camera pan).  ByteTrack assigns persistent IDs
to each person and keeps them consistent as the boxes shift across frames.

Output
──────
  data/samples/tracking_frame_01.jpg  — annotated frame 1 (first assignment)
  data/samples/tracking_frame_08.jpg  — mid-clip (IDs should match frame 1)
  data/samples/tracking_frame_15.jpg  — last frame (IDs still consistent)
  data/samples/tracking_strip.jpg     — three frames side-by-side for comparison

Run:
    python scripts/demo_tracking.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics.utils import ASSETS

from src.tracking.multi_tracker import MultiTracker, TrackingResult
from src.tracking.track_manager import TrackManager

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_IMAGE  = ASSETS / "bus.jpg"
OUT_DIR       = ROOT / "data" / "samples"
N_FRAMES      = 15
PAN_PX        = 3       # pixels to shift right each frame (simulates camera pan)
CONF_THRESH   = 0.40
DEVICE        = "cpu"

# Each unique track_id gets a distinct, consistent color
_PALETTE = [
    (0,   200,   0),   # green
    (0,   100, 255),   # orange
    (220,   0, 220),   # magenta
    (0,   220, 220),   # cyan
    (255,  50,  50),   # blue
    (50,  255, 255),   # yellow
]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def track_color(track_id: int) -> tuple:
    return _PALETTE[track_id % len(_PALETTE)]


def pan_frame(bgr: np.ndarray, dx: int) -> np.ndarray:
    """Shift image dx pixels to the right, filling left edge with gray."""
    h, w = bgr.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(bgr, M, (w, h), borderValue=(114, 114, 114))


def annotate(bgr: np.ndarray, tracks, frame_idx: int, manager: TrackManager) -> np.ndarray:
    canvas = bgr.copy()

    for t in tracks:
        color = track_color(t.track_id)
        x1, y1, x2, y2 = [int(v) for v in t.bbox]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        vel = manager.get_velocity(t.track_id, n_frames=5)
        if vel is not None:
            vx, vy = vel
            vel_str = f" v=({vx:+.1f},{vy:+.1f})"
        else:
            vel_str = ""

        label = f"ID:{t.track_id} {t.confidence:.2f}{vel_str}"
        (lw, lh), bl = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - lh - bl - 4), (x1 + lw, y1), color, -1)
        cv2.putText(canvas, label, (x1, y1 - bl - 2), FONT, 0.45, (255, 255, 255), 1)

        # Draw velocity arrow from center
        if vel is not None and np.linalg.norm(vel) > 0.5:
            cx, cy = int(t.center[0]), int(t.center[1])
            tip_x = int(cx + vel[0] * 10)
            tip_y = int(cy + vel[1] * 10)
            cv2.arrowedLine(canvas, (cx, cy), (tip_x, tip_y), color, 2, tipLength=0.4)

    # Banner
    banner = (
        f"Frame {frame_idx:02d}/{N_FRAMES}  |  "
        f"Active tracks: {len(tracks)}  |  "
        f"History depth: {len(manager.get_history(tracks[0].track_id)) if tracks else 0}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(canvas, banner, (8, 18), FONT, 0.50, (255, 255, 255), 1)

    return canvas


def main():
    print("=" * 65)
    print("Basketball Defensive Vision — Tracking Demo")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bgr_orig = cv2.imread(str(SAMPLE_IMAGE))
    assert bgr_orig is not None, f"Could not load {SAMPLE_IMAGE}"
    h, w = bgr_orig.shape[:2]
    print(f"\nSource image : {SAMPLE_IMAGE.name}  ({w}×{h} px)")
    print(f"Simulating   : {N_FRAMES} frames, {PAN_PX}px/frame right pan\n")

    tracker = MultiTracker(
        weights_path="yolov8m.pt",
        conf_thresh=CONF_THRESH,
        device=DEVICE,
    )
    manager = TrackManager(history_len=30)

    saved_frames = {}
    save_at = {1, 8, 15}

    for i in range(1, N_FRAMES + 1):
        bgr = pan_frame(bgr_orig, dx=(i - 1) * PAN_PX)
        fps = 30.0
        frame_idx = (i - 1) * 3        # stride=3 → frame indices 0, 3, 6, ...
        timestamp  = frame_idx / fps

        tracks = tracker.update(bgr, frame_idx=frame_idx, timestamp_sec=timestamp)
        manager.update(tracks, ball_track=None, frame_idx=frame_idx, timestamp_sec=timestamp)

        # Print per-frame summary
        id_str = " ".join(f"#{t.track_id}" for t in tracks)
        print(f"  Frame {i:02d} (idx={frame_idx:3d})  "
              f"tracks=[{id_str}]  count={len(tracks)}")

        if i in save_at:
            ann = annotate(bgr, tracks, i, manager)
            saved_frames[i] = ann
            path = OUT_DIR / f"tracking_frame_{i:02d}.jpg"
            cv2.imwrite(str(path), ann)

    # ── Stitch into a comparison strip ───────────────────────────────────────
    strip_frames = [saved_frames[k] for k in sorted(saved_frames)]
    target_h = min(f.shape[0] for f in strip_frames)
    resized = []
    for f in strip_frames:
        scale = target_h / f.shape[0]
        rw = int(f.shape[1] * scale)
        resized.append(cv2.resize(f, (rw, target_h)))

    divider = np.full((target_h, 6, 3), 255, dtype=np.uint8)
    strip = resized[0]
    for r in resized[1:]:
        strip = np.hstack([strip, divider, r])

    strip_path = OUT_DIR / "tracking_strip.jpg"
    cv2.imwrite(str(strip_path), strip)

    # ── Per-track history summary ─────────────────────────────────────────────
    print("\n── Track history summary ──────────────────────────────────────")
    final_tracks = tracker.update(
        pan_frame(bgr_orig, dx=(N_FRAMES - 1) * PAN_PX),
        frame_idx=N_FRAMES * 3,
        timestamp_sec=N_FRAMES * 3 / 30.0,
    )
    for t in sorted(manager.get_active_tracks().values(), key=lambda x: x.track_id):
        hist  = manager.get_history(t.track_id)
        vel   = manager.get_velocity(t.track_id, n_frames=8)
        alive = manager.frames_active(t.track_id)
        vel_s = f"({vel[0]:+.1f},{vel[1]:+.1f}) px/frame" if vel is not None else "n/a"
        print(f"  ID {t.track_id:>2}  history={len(hist):>2} frames  "
              f"alive={alive:>2} frames  velocity={vel_s}")

    print(f"\n── Output files ───────────────────────────────────────────────")
    for k in sorted(saved_frames):
        p = OUT_DIR / f"tracking_frame_{k:02d}.jpg"
        print(f"  {p}")
    print(f"  {strip_path}")
    print(f"\n  ↑ tracking_strip.jpg shows frames 1 | 8 | 15 side-by-side.")
    print(f"  Same track IDs (same colors) across all three = tracker working.\n")


if __name__ == "__main__":
    main()
