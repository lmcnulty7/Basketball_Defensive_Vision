"""Probe exactly which line crashes by printing before each step."""
import os, sys, types
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# lap stub
import numpy as _np
from scipy.optimize import linear_sum_assignment as _lsa
def _lapjv(cost, extend_cost=False, cost_limit=float("inf"), return_cost=True):
    c = cost.astype(float)
    row_ind, col_ind = _lsa(c)
    n_rows, n_cols = cost.shape
    x = _np.full(n_rows, -1, dtype=_np.int32)
    y = _np.full(n_cols, -1, dtype=_np.int32)
    for r, c_ in zip(row_ind, col_ind):
        if r < n_rows and c_ < n_cols:
            x[r] = c_; y[c_] = r
    return 0.0, x, y
_lap = types.ModuleType("lap"); _lap.lapjv = _lapjv; sys.modules["lap"] = _lap

sys.path.insert(0, ".")
import cv2
import numpy as np

print("step 1: open video")
cap = cv2.VideoCapture("data/raw/curry_classic_clip.mp4")
ret, frame = cap.read()
print(f"step 2: got frame ret={ret} shape={frame.shape if ret else None}")

print("step 3: court visibility check")
from src.pipeline.runner import PipelineRunner
r = PipelineRunner.__new__(PipelineRunner)
visible = r._is_court_visible(frame)
print(f"step 4: court visible = {visible}")

print("step 5: create MultiTracker")
from src.tracking.multi_tracker import MultiTracker
tracker = MultiTracker("models/checkpoints/yolov8m.pt", device="cpu",
                       tracker_config="botsort.yaml")

print("step 6: call tracker.update() — KEEP VideoCapture OPEN")
# Note: cap is still open here, same as full pipeline
tracks = tracker.update(frame, frame_idx=0, timestamp_sec=0.0)
print(f"step 7: tracks = {len(tracks)}")

print("step 8: read second frame via VideoCapture")
ret2, frame2 = cap.read()
cap.release()
print(f"step 9: second frame ret={ret2}")

print("ALL DONE")
