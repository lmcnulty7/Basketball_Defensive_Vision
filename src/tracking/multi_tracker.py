"""
src/tracking/multi_tracker.py

Assigns persistent track IDs to players across frames using ByteTrack.

Uses ultralytics' built-in ByteTrack implementation (model.track()), which
runs detection + tracking in a single forward pass.  persist=True keeps the
ByteTrack state alive between calls so IDs are consistent frame-to-frame.

Architecture note
─────────────────
MultiTracker is the streaming replacement for PlayerDetector in the main
pipeline loop.  PlayerDetector.detect() is still useful for single-frame
analysis (event snapshots, gallery building for ReID), but MultiTracker
is what runs on every frame.

Camera cuts
───────────
ByteTrack maintains state indefinitely.  Call reset() on detected camera
cuts (identified by >80% of tracks simultaneously vanishing) to clear all
IDs and start fresh with the new broadcast angle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

BALL_TRACK_ID = -1   # reserved ID for the ball across the entire codebase


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Track:
    """
    A single tracked object at one point in time.

    Attributes
    ──────────
    track_id      : Persistent integer ID assigned by ByteTrack.  Stays
                    consistent across occlusions within the same camera angle.
    bbox          : [x1, y1, x2, y2] in original pixel coords, float32.
    center        : [cx, cy] of the bounding box, float32.
    confidence    : YOLO detection confidence for this frame.
    class_id      : COCO class (0 = person, BALL_TRACK_ID = ball).
    frame_idx     : Absolute frame index from the source video.
    timestamp_sec : Frame time in seconds.
    team_id       : 0 = home, 1 = away, 2 = referee; None until classified.
    court_pos     : [x_ft, y_ft] real court coordinates; None until homography runs.
    """
    track_id: int
    bbox: np.ndarray        # shape (4,) float32
    center: np.ndarray      # shape (2,) float32
    confidence: float
    class_id: int
    frame_idx: int
    timestamp_sec: float
    team_id: Optional[int] = None
    court_pos: Optional[np.ndarray] = None
    jersey_number: Optional[str] = None    # read by JerseyReader (SmolVLM2)
    player_name: Optional[str]   = None    # resolved from roster DB

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    def to_dict(self) -> dict:
        return {
            "track_id":      self.track_id,
            "bbox":          self.bbox.tolist(),
            "center":        self.center.tolist(),
            "confidence":    round(float(self.confidence), 4),
            "class_id":      self.class_id,
            "frame_idx":     self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "team_id":       self.team_id,
            "court_pos":     self.court_pos.tolist() if self.court_pos is not None else None,
        }


@dataclass
class TrackingResult:
    """
    All tracking outputs for a single video frame.

    Attributes
    ──────────
    player_tracks     : Active player tracks this frame, sorted by track_id.
    ball_track        : Ball track (BALL_TRACK_ID = -1), or None if lost.
    ball_is_predicted : True if ball position is a Kalman extrapolation.
    ball_possessor_id : track_id of the player holding the ball, or None.
    raw_bgr           : Original frame for visualization and downstream use.
    """
    frame_idx: int
    timestamp_sec: float
    player_tracks: List[Track]
    ball_track: Optional[Track]
    ball_is_predicted: bool
    ball_possessor_id: Optional[int]
    raw_bgr: np.ndarray

    def to_dict(self) -> dict:
        return {
            "frame_idx":        self.frame_idx,
            "timestamp_sec":    round(self.timestamp_sec, 4),
            "player_tracks":    [t.to_dict() for t in self.player_tracks],
            "ball_track":       self.ball_track.to_dict() if self.ball_track else None,
            "ball_is_predicted": self.ball_is_predicted,
            "ball_possessor_id": self.ball_possessor_id,
        }


# ── MultiTracker ──────────────────────────────────────────────────────────────

class MultiTracker:
    """
    Player tracker using ultralytics' built-in ByteTrack.

    Parameters
    ──────────
    weights_path    : YOLOv8 .pt weights.  Auto-downloaded if not found locally.
    conf_thresh     : Minimum detection confidence (should match bytetrack.yaml
                      track_high_thresh or lower).
    iou_thresh      : NMS IoU threshold for detection.
    device          : "cpu" | "cuda" | "mps"
    tracker_config  : Path to a bytetrack.yaml config file, or the string
                      "bytetrack.yaml" to use ultralytics' bundled defaults.

    Example
    ───────
        tracker = MultiTracker("yolov8m.pt", device="mps")
        for frame in video_reader:
            tracks = tracker.update(frame.data, frame.frame_idx, frame.timestamp_sec)
    """

    def __init__(
        self,
        weights_path: str | Path = "yolov8m.pt",
        conf_thresh: float = 0.40,
        iou_thresh: float = 0.45,
        device: str = "cpu",
        tracker_config: str | Path = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLO  # lazy import

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self._tracker_config = str(tracker_config)
        self._model = YOLO(str(weights_path))
        logger.info(
            "MultiTracker ready (device=%s, conf=%.2f, tracker=%s)",
            device, conf_thresh, tracker_config,
        )

    @classmethod
    def from_config(
        cls,
        player_cfg: Dict,
        device: str = "cpu",
        tracker_config: str | Path = "bytetrack.yaml",
    ) -> "MultiTracker":
        """Construct from the `detection` section of models.yaml."""
        return cls(
            weights_path=player_cfg["player_weights"],
            conf_thresh=player_cfg["player_conf"],
            iou_thresh=player_cfg["iou_threshold"],
            device=device,
            tracker_config=tracker_config,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        bgr: np.ndarray,
        frame_idx: int = 0,
        timestamp_sec: float = 0.0,
    ) -> List[Track]:
        """
        Detect + track players in a single BGR frame.

        Parameters
        ──────────
        bgr           : Raw frame, shape (H, W, 3) uint8.
        frame_idx     : Absolute frame index (used to stamp Track objects).
        timestamp_sec : Frame timestamp (used to stamp Track objects).

        Returns
        ───────
        List of active Track objects, sorted by track_id ascending.
        Tracks that ByteTrack is keeping alive but didn't match a detection
        this frame are NOT returned — they are in the ByteTrack internal
        buffer but not emitted until they match again.
        """
        try:
            results = self._model.track(
                bgr,
                classes=[0],                 # person only
                persist=True,                # maintain ByteTrack state across calls
                tracker=self._tracker_config,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self.device,
                verbose=False,
            )
            tracks = self._parse_results(results, frame_idx, timestamp_sec)
        except np.linalg.LinAlgError:
            # BoT-SORT Kalman filter can become non-positive-definite on MPS
            # due to float precision. Reset and return empty tracks for this frame.
            logger.warning(
                "Kalman filter numerical error at frame %d — resetting tracker",
                frame_idx,
            )
            self.reset()
            tracks = []
        logger.debug("MultiTracker: frame %d — %d active tracks", frame_idx, len(tracks))
        return tracks

    def reset(self) -> None:
        """
        Clear all ByteTrack state.

        Call this when a camera cut is detected (identified by >80% of active
        tracks vanishing in a single frame).  The next update() call will
        re-initialize the tracker and assign fresh IDs from 1.
        """
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor = None
        logger.info("MultiTracker: state reset — all track IDs cleared")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_results(
        results,
        frame_idx: int,
        timestamp_sec: float,
    ) -> List[Track]:
        tracks: List[Track] = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            boxes   = r.boxes.xyxy.cpu().numpy()               # (N, 4)
            ids     = r.boxes.id.cpu().numpy().astype(int)     # (N,)
            confs   = r.boxes.conf.cpu().numpy()               # (N,)
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)    # (N,)

            for bbox, tid, conf, cls_id in zip(boxes, ids, confs, cls_ids):
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                tracks.append(Track(
                    track_id=int(tid),
                    bbox=bbox.astype(np.float32),
                    center=np.array([cx, cy], dtype=np.float32),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                ))

        tracks.sort(key=lambda t: t.track_id)
        return tracks
