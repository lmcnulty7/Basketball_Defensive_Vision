"""
src/court/court_detector.py

Detects the basketball court bounding box in a broadcast frame using a
fine-tuned YOLOv8n model.  Used to crop frames before running
ClassicalKeyDetector, reducing noise from crowd and sideline areas.

Usage
─────
  detector = CourtDetector()
  bbox = detector.detect(frame)              # [x1, y1, x2, y2] or None
  crop, x_off, y_off = detector.crop(frame) # cropped frame + offsets
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "court_detector_yolov8n.pt"
MARGIN_FRAC = 0.02  # expand bbox by 2% on each side to avoid clipping court lines


class CourtDetector:
    """
    Wraps a YOLOv8n court bounding-box detector.

    Falls back gracefully (returns full frame) when weights are absent,
    so the rest of the pipeline keeps running unchanged.
    """

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        conf_thresh: float = 0.4,
        device: str = "cpu",
    ) -> None:
        self.conf_thresh = conf_thresh
        self.device = device
        self._model = None

        weights_path = Path(weights_path)
        if weights_path.exists():
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(weights_path))
                logger.info("CourtDetector: loaded %s", weights_path)
            except Exception as e:
                logger.warning("CourtDetector: failed to load model (%s) — returning full frame", e)
        else:
            logger.info(
                "CourtDetector: weights not found at %s — run train_court_detector.py first",
                weights_path,
            )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def detect(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Return the best court bounding box as [x1, y1, x2, y2] in pixels,
        or None if no court detected or model not loaded.
        """
        if self._model is None:
            return None
        try:
            results = self._model.predict(bgr, conf=self.conf_thresh, verbose=False)
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                # Take highest-confidence detection
                best_idx = int(r.boxes.conf.argmax())
                box = r.boxes.xyxy[best_idx].cpu().numpy().astype(np.float32)
                return box  # [x1, y1, x2, y2]
        except Exception as e:
            logger.debug("CourtDetector.detect failed: %s", e)
        return None

    def crop(self, bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Crop the frame to the detected court region.

        Returns
        ───────
        (cropped_frame, x_offset, y_offset)

        x_offset / y_offset must be added back to any pixel coordinates
        produced by a detector run on the cropped frame to convert them
        to full-frame coordinates.

        If no court is detected, returns (full_frame, 0, 0).
        """
        h, w = bgr.shape[:2]
        bbox = self.detect(bgr)

        if bbox is None:
            return bgr, 0, 0

        x1, y1, x2, y2 = bbox

        # Expand by margin to avoid clipping court lines at edges
        margin_x = (x2 - x1) * MARGIN_FRAC
        margin_y = (y2 - y1) * MARGIN_FRAC
        x1 = max(0, int(x1 - margin_x))
        y1 = max(0, int(y1 - margin_y))
        x2 = min(w, int(x2 + margin_x))
        y2 = min(h, int(y2 + margin_y))

        cropped = bgr[y1:y2, x1:x2]
        return cropped, x1, y1
