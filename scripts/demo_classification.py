"""
scripts/demo_classification.py

Demonstrates TeamClassifier on the bus.jpg sample (4 real people).

The 4 people wear noticeably different clothing.  The classifier:
  1. Extracts the torso jersey color from each person crop.
  2. Accumulates color samples across 20 simulated frames.
  3. Fits KMeans(k=2) to split them into two "teams."
  4. Annotates each person with their assigned team color and
     a small jersey-color swatch.

The output proves the full accumulate→fit→predict pipeline works on real
pixel data.  On actual basketball footage the same code runs unchanged.

Output: data/samples/classification_demo.jpg
Run:    python scripts/demo_classification.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics.utils import ASSETS

from src.tracking.multi_tracker import MultiTracker
from src.classification.team_classifier import (
    TeamClassifier, TEAM_HOME, TEAM_AWAY, TEAM_REFEREE, TEAM_UNKNOWN, TEAM_BGR
)

OUT_PATH = ROOT / "data" / "samples" / "classification_demo.jpg"
FONT     = cv2.FONT_HERSHEY_SIMPLEX
N_SIMULATED_FRAMES = 20


def main():
    print("=" * 65)
    print("Basketball Defensive Vision — Team Classification Demo")
    print("=" * 65)

    bgr_orig = cv2.imread(str(ASSETS / "bus.jpg"))
    assert bgr_orig is not None

    # ── 1. Detect + track people ──────────────────────────────────────────────
    print("\n[1/3] Detecting and tracking players (YOLOv8m)...")
    tracker = MultiTracker(weights_path="yolov8m.pt", conf_thresh=0.40, device="cpu")
    tracks  = tracker.update(bgr_orig, frame_idx=0, timestamp_sec=0.0)
    print(f"      {len(tracks)} tracks found: IDs = {[t.track_id for t in tracks]}")

    # ── 2. Accumulate jersey colors over N simulated frames ───────────────────
    print(f"\n[2/3] Accumulating jersey colors over {N_SIMULATED_FRAMES} frames...")
    clf = TeamClassifier(
        min_samples_to_fit=len(tracks) * 5,
        auto_fit=True,
        sat_thresh=45,
    )

    for i in range(N_SIMULATED_FRAMES):
        assignments = clf.process_frame(tracks, bgr_orig, frame_idx=i)

    print(f"      Samples collected : {clf.n_samples_collected}")
    print(f"      Classifier fitted : {clf.is_fitted}")

    if not clf.is_fitted:
        clf.fit()

    # ── 3. Retrieve assignments + cluster colors ──────────────────────────────
    team_colors_bgr = clf.get_cluster_colors_bgr()
    print(f"\n[3/3] Team assignments:")

    LABEL = {TEAM_HOME: "Home (0)", TEAM_AWAY: "Away (1)",
             TEAM_REFEREE: "Referee", TEAM_UNKNOWN: "Unknown"}

    for track in tracks:
        tid  = track.track_id
        team = clf.get_assignment(tid)
        col  = team_colors_bgr.get(team, (128, 128, 128))
        print(f"      Track #{tid}  →  {LABEL[team]}  "
              f"cluster_color=BGR{col}")

    # ── Annotated output ──────────────────────────────────────────────────────
    canvas = bgr_orig.copy()

    for track in tracks:
        tid  = track.track_id
        team = clf.get_assignment(tid)

        # Bounding box in team color
        box_color = TEAM_BGR[team]
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)

        # Jersey color swatch (what the classifier extracted)
        swatch_color = team_colors_bgr.get(team, (128, 128, 128))
        sx1, sy1 = x1, max(0, y1 - 28)
        sx2, sy2 = x1 + 28, max(0, y1 - 2)
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), swatch_color, -1)
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (255,255,255), 1)

        # Label
        label = f"#{tid} {LABEL[team]}"
        (lw, lh), bl = cv2.getTextSize(label, FONT, 0.42, 1)
        cv2.rectangle(canvas, (x1+30, max(0,y1-28)), (x1+30+lw, max(0,y1)-2), box_color, -1)
        cv2.putText(canvas, label, (x1+32, max(0,y1)-bl-2), FONT, 0.42, (255,255,255), 1)

        # Jersey color extracted from torso — draw torso region outline
        h = y2 - y1; w_ = x2 - x1
        tr1 = (x1 + int(w_*0.20), y1 + int(h*0.25))
        tr2 = (x1 + int(w_*0.80), y1 + int(h*0.65))
        cv2.rectangle(canvas, tr1, tr2, (0, 255, 255), 1)

    # Legend
    ly = 20
    for team_id, label in LABEL.items():
        if team_id == TEAM_UNKNOWN:
            continue
        color = TEAM_BGR[team_id]
        cv2.circle(canvas, (15, ly), 8, color, -1)
        cv2.putText(canvas, label, (28, ly + 4), FONT, 0.40, (255,255,255), 1)
        if team_id in team_colors_bgr:
            cv2.rectangle(canvas, (130, ly-8), (150, ly+8), team_colors_bgr[team_id], -1)
            cv2.rectangle(canvas, (130, ly-8), (150, ly+8), (200,200,200), 1)
        ly += 22

    # Summary banner
    banner = (f"Tracks: {len(tracks)}  |  "
              f"Samples: {clf.n_samples_collected}  |  "
              f"Fitted: {clf.is_fitted}  |  "
              f"Yellow box = torso region analyzed")
    cv2.rectangle(canvas, (0, canvas.shape[0]-28), (canvas.shape[1], canvas.shape[0]),
                  (20,20,20), -1)
    cv2.putText(canvas, banner, (8, canvas.shape[0]-10), FONT, 0.42, (255,255,255), 1)

    cv2.imwrite(str(OUT_PATH), canvas)
    print(f"\n  Output → {OUT_PATH}")
    print("  (yellow boxes = torso regions sampled for jersey color)")


if __name__ == "__main__":
    main()
