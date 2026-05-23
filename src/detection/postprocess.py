"""
src/detection/postprocess.py

Shared data structures and post-processing utilities for detection outputs.

All bounding boxes are in ORIGINAL pixel coordinates (raw frame space, before
any letterboxing or model-internal resizing).  The ultralytics YOLO API
returns boxes in original pixel space automatically, so no manual unprojection
is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# COCO class names referenced by this pipeline
COCO_NAMES = {
    0:  "person",
    32: "sports_ball",
}

BALL_CLASS_ID = 100  # internal pseudo-class for the basketball (normalized from any model)


# ── Core data structures ───────────────────────────────────────────────────────

@dataclass
class Detection:
    """
    A single object detection.

    Attributes
    ──────────
    bbox        : [x1, y1, x2, y2] in original pixel coordinates (float32).
    confidence  : Model confidence score in [0, 1].
    class_id    : COCO class index, or BALL_CLASS_ID (100) for ball detections.
    class_name  : Human-readable label string.
    track_id    : Assigned by the tracker module; None until tracking runs.
    """
    bbox: np.ndarray       # shape (4,) float32, [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def center(self) -> np.ndarray:
        """[cx, cy] center of the bounding box."""
        return np.array(
            [(self.bbox[0] + self.bbox[2]) / 2.0,
             (self.bbox[1] + self.bbox[3]) / 2.0],
            dtype=np.float32,
        )

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.tolist(),
            "confidence": round(float(self.confidence), 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


@dataclass
class DetectionResult:
    """
    All detections for a single video frame.

    Attributes
    ──────────
    frame_idx        : Absolute frame index from the source video.
    timestamp_sec    : Frame timestamp in seconds.
    players          : All person detections, confidence-sorted descending.
    ball             : Ball detection, or None if completely lost.
    ball_is_predicted: True when ball position is a Kalman extrapolation.
    raw_bgr          : Original frame — used for visualization and downstream modules.
    """
    frame_idx: int
    timestamp_sec: float
    players: List[Detection]
    ball: Optional[Detection]
    ball_is_predicted: bool
    raw_bgr: np.ndarray

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "players": [p.to_dict() for p in self.players],
            "ball": self.ball.to_dict() if self.ball else None,
            "ball_is_predicted": self.ball_is_predicted,
        }


# ── Utility functions ──────────────────────────────────────────────────────────

def filter_by_confidence(
    detections: List[Detection],
    min_conf: float,
) -> List[Detection]:
    """Keep only detections at or above the confidence threshold."""
    return [d for d in detections if d.confidence >= min_conf]


def filter_by_area(
    detections: List[Detection],
    min_area: float = 100.0,
    max_area: float = 1e7,
) -> List[Detection]:
    """Remove detections whose bounding box area is outside [min_area, max_area]."""
    return [d for d in detections if min_area <= d.area <= max_area]


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection-over-Union for two boxes in [x1, y1, x2, y2] format.

    Used by event detectors to check spatial overlap (e.g. charge detection).
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0
