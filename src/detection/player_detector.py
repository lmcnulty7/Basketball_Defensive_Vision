"""
src/detection/player_detector.py

Detects players in each frame using YOLOv8.

Uses the ultralytics YOLO API, which handles letterbox preprocessing
internally and returns bounding boxes in original pixel coordinates —
no manual unprojection needed here.

Filters to COCO class 0 (person) only.  Referee vs. player separation
happens later in team_classifier, which has jersey-color information.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.detection.postprocess import COCO_NAMES, Detection

logger = logging.getLogger(__name__)

_PERSON_CLASS = 0


class PlayerDetector:
    """
    YOLOv8-based person detector.

    Parameters
    ──────────
    weights_path : Path to .pt weights file.  ultralytics auto-downloads
                   named weights (e.g. "yolov8m.pt") when not found locally.
    conf_thresh  : Minimum detection confidence.
    iou_thresh   : NMS IoU threshold (applied inside ultralytics).
    device       : "cpu" | "cuda" | "mps"
    """

    def __init__(
        self,
        weights_path: str | Path = "yolov8m.pt",
        conf_thresh: float = 0.40,
        iou_thresh: float = 0.45,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO  # lazy import — only fails if ultralytics missing

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device

        logger.info("Loading PlayerDetector weights: %s", weights_path)
        self._model = YOLO(str(weights_path))
        logger.info("PlayerDetector ready (device=%s, conf=%.2f)", device, conf_thresh)

    @classmethod
    def from_config(cls, cfg: Dict, device: str = "cpu") -> "PlayerDetector":
        """Construct from the `detection` section of models.yaml."""
        return cls(
            weights_path=cfg["player_weights"],
            conf_thresh=cfg["player_conf"],
            iou_thresh=cfg["iou_threshold"],
            device=device,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        """
        Detect players in a single BGR frame.

        Parameters
        ──────────
        bgr : Raw frame from VideoReader, shape (H, W, 3) uint8.

        Returns
        ───────
        Detections for COCO class 0 (person), sorted by confidence descending.
        Boxes are in original pixel coordinates.
        """
        results = self._model.predict(
            bgr,
            classes=[_PERSON_CLASS],
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def detect_batch(self, bgr_frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect players in a list of frames in one forward pass.

        Higher GPU utilization than calling detect() in a loop.
        Returns one Detection list per input frame, preserving order.
        """
        results = self._model.predict(
            bgr_frames,
            classes=[_PERSON_CLASS],
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )
        return [self._parse_results([r]) for r in results]

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_results(results) -> List[Detection]:
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes   = r.boxes.xyxy.cpu().numpy()              # (N, 4)
            confs   = r.boxes.conf.cpu().numpy()              # (N,)
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)   # (N,)

            for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                detections.append(Detection(
                    bbox=bbox.astype(np.float32),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=COCO_NAMES.get(int(cls_id), str(cls_id)),
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.debug("PlayerDetector: %d detections", len(detections))
        return detections
