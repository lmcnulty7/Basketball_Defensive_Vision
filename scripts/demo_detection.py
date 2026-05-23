"""
scripts/demo_detection.py

End-to-end smoke test of the ingestion + detection pipeline.

Uses the ultralytics bundled sample image (bus.jpg) so no footage is required.
Outputs an annotated image to data/samples/demo_output.jpg.

Run:
    python scripts/demo_detection.py
"""

import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics.utils import ASSETS

from src.ingestion.video_reader import Frame
from src.ingestion.preprocessor import Preprocessor
from src.detection.player_detector import PlayerDetector
from src.detection.ball_detector import BallDetector, KalmanBallFilter
from src.detection.postprocess import DetectionResult, filter_by_confidence


# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_IMAGE = ASSETS / "bus.jpg"       # ships with ultralytics — people on a bus
OUTPUT_PATH  = ROOT / "data" / "samples" / "demo_output.jpg"
CONF_THRESH  = 0.40
DEVICE       = "cpu"


# ── Colors ────────────────────────────────────────────────────────────────────

COLOR_PLAYER = (0, 200, 0)    # green
COLOR_TEXT   = (255, 255, 255)
FONT         = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(bgr: np.ndarray, result: DetectionResult) -> np.ndarray:
    canvas = bgr.copy()

    for det in result.players:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_PLAYER, 2)

        label = f"person {det.confidence:.2f}"
        (lw, lh), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), COLOR_PLAYER, -1)
        cv2.putText(canvas, label, (x1, y1 - baseline - 2), FONT, 0.5, COLOR_TEXT, 1)

    # Summary banner at top
    summary = (
        f"Frame {result.frame_idx}  |  "
        f"Players detected: {len(result.players)}  |  "
        f"Ball: {'predicted' if result.ball_is_predicted else ('detected' if result.ball else 'not found')}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(canvas, summary, (8, 20), FONT, 0.55, COLOR_TEXT, 1)

    return canvas


def main():
    print("=" * 60)
    print("Basketball Defensive Vision — Pipeline Demo")
    print("=" * 60)

    # ── 1. Load sample image as a Frame ──────────────────────────────────────
    print(f"\n[1/4] Loading sample image: {SAMPLE_IMAGE.name}")
    bgr = cv2.imread(str(SAMPLE_IMAGE))
    if bgr is None:
        raise FileNotFoundError(f"Could not read {SAMPLE_IMAGE}")
    h, w = bgr.shape[:2]
    print(f"      Image size: {w}×{h} px")

    frame = Frame(data=bgr, frame_idx=0, timestamp_sec=0.0)

    # ── 2. Ingestion — Preprocessor ───────────────────────────────────────────
    print("\n[2/4] Running Preprocessor (letterbox resize → tensor)...")
    prep = Preprocessor(input_size=640, device=DEVICE)
    processed = prep.process(frame)
    print(f"      Tensor shape : {tuple(processed.tensor.shape)}")
    print(f"      Scale factor : {processed.scale:.4f}")
    print(f"      Padding (L,T): {processed.pad}")

    # ── 3. Detection — PlayerDetector ────────────────────────────────────────
    print("\n[3/4] Running PlayerDetector (YOLOv8m)...")
    print("      (downloading yolov8m.pt on first run — ~50 MB)")
    detector = PlayerDetector(
        weights_path="yolov8m.pt",
        conf_thresh=CONF_THRESH,
        device=DEVICE,
    )
    players = detector.detect(bgr)
    print(f"      Detections: {len(players)}")
    for i, d in enumerate(players):
        cx, cy = d.center
        print(
            f"        [{i+1}] conf={d.confidence:.3f}  "
            f"bbox=[{d.bbox[0]:.0f},{d.bbox[1]:.0f},{d.bbox[2]:.0f},{d.bbox[3]:.0f}]  "
            f"center=({cx:.0f},{cy:.0f})  "
            f"size={d.width:.0f}×{d.height:.0f}px"
        )

    # ── 4. Ball detector (Kalman only — no ball in bus.jpg) ───────────────────
    print("\n[4/4] Running BallDetector (Kalman filter demo)...")
    kf = KalmanBallFilter()
    kf.update(320.0, 240.0)   # simulate ball seen at frame 0
    kf.update(340.0, 230.0)   # frame 1
    kf.update(360.0, 220.0)   # frame 2
    state = kf.predict()       # frame 3 — occluded, predict from velocity
    print(f"      Observed:  (320,240), (340,230), (360,220)")
    print(f"      Predicted: ({state[0]:.1f}, {state[1]:.1f})  "
          f"velocity=({state[2]:.1f}, {state[3]:.1f}) px/frame")
    print(f"      Frames since last observation: {kf.frames_since_update}")

    # ── Assemble DetectionResult ──────────────────────────────────────────────
    result = DetectionResult(
        frame_idx=0,
        timestamp_sec=0.0,
        players=players,
        ball=None,
        ball_is_predicted=False,
        raw_bgr=bgr,
    )

    # ── Save annotated output ─────────────────────────────────────────────────
    annotated = draw_detections(bgr, result)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), annotated)

    print(f"\n{'=' * 60}")
    print(f"Annotated output saved → {OUTPUT_PATH}")
    print(f"{'=' * 60}")
    print("\nPipeline stage summary:")
    print(f"  VideoReader   ✓  Frame(idx=0, size={w}×{h})")
    print(f"  Preprocessor  ✓  Tensor {tuple(processed.tensor.shape)}, scale={processed.scale:.3f}")
    print(f"  PlayerDetect  ✓  {len(players)} person(s) at conf ≥ {CONF_THRESH}")
    print(f"  KalmanFilter  ✓  Predicted ball at ({state[0]:.1f}, {state[1]:.1f})")
    print()


if __name__ == "__main__":
    main()
