"""
src/court/keypoint_detector.py

Detects NBA court landmark keypoints in broadcast video frames.

Two implementations
───────────────────
ClassicalKeyDetector
    Works today, no model training required.  Segments the court floor by
    color, finds white court lines via Hough transform, and extracts paint
    corners as the primary homography anchors.  Reliable on clean broadcast
    frames; degrades on heavy player occlusion or unusual lighting.

NeuralKeyDetector
    Production-grade.  Wraps a YOLOv8-pose model fine-tuned on annotated
    NBA frames to detect all 15+ canonical court keypoints.  The interface
    is fully implemented; the weights path is configured in models.yaml.
    When the weights are absent, falls back to ClassicalKeyDetector.

Both expose the same interface:
    detector.detect(bgr) → Dict[str, Optional[np.ndarray]]
    # keys = canonical keypoint names (subset of COURT_KEYPOINTS)
    # values = [u, v] pixel coordinates, or None if not found

Why paint corners are the primary target
────────────────────────────────────────
The 4 paint corners form a 16×19 ft rectangle that is:
  (a) always on screen in a half-court broadcast angle,
  (b) visually distinctive (sharp white corners on maple wood),
  (c) spans a large fraction of the frame → good H conditioning.
4 non-collinear corners give a valid H; additional keypoints (free throw
line, 3PT arc endpoints, half-court line) over-constrain the system and
improve robustness via RANSAC.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
def _get_court_detector():
    from src.court.court_detector import CourtDetector
    return CourtDetector()

# Keypoint names returned by this module
# (subset of COURT_KEYPOINTS in court_model.py)
KEYPOINT_NAMES = [
    "home_paint_bl", "home_paint_tl",   # baseline corners of home paint
    "home_paint_br", "home_paint_tr",   # free throw line corners of home paint
    "away_paint_bl", "away_paint_tl",   # baseline corners of away paint
    "away_paint_br", "away_paint_tr",   # free throw line corners of away paint
    "half_bot",      "half_top",        # half-court sideline intersections
    "home_three_bl", "home_three_tl",   # home 3PT arc/straight junctions
    "away_three_bl", "away_three_tl",   # away 3PT arc/straight junctions
]

KeypointDict = Dict[str, Optional[np.ndarray]]


# ── Base class ────────────────────────────────────────────────────────────────

class BaseKeyDetector(ABC):
    """Abstract interface all keypoint detectors must implement."""

    @abstractmethod
    def detect(self, bgr: np.ndarray) -> KeypointDict:
        """
        Detect court keypoints in a single BGR frame.

        Returns a dict mapping keypoint name → [u, v] pixel coordinate,
        or None for keypoints that could not be located.
        """

    def detect_batch(self, frames: List[np.ndarray]) -> List[KeypointDict]:
        """Detect keypoints in a list of frames (default: loop over detect)."""
        return [self.detect(f) for f in frames]

    @staticmethod
    def count_detected(kp_dict: KeypointDict) -> int:
        return sum(1 for v in kp_dict.values() if v is not None)


# ── Classical detector ────────────────────────────────────────────────────────

class ClassicalKeyDetector(BaseKeyDetector):
    """
    Keypoint detector using color segmentation + Hough line transform.

    Algorithm
    ─────────
    1. Segment court floor (maple wood color in HSV space).
    2. Mask out player regions (large blobs inside the court).
    3. Threshold for white court markings within the floor mask.
    4. Run probabilistic Hough transform to extract line segments.
    5. Classify lines as near-horizontal or near-vertical.
    6. Find intersections of H×V line pairs to get candidate corners.
    7. Cluster candidates and assign to the nearest canonical keypoint.

    Parameters
    ──────────
    floor_hsv_lo / floor_hsv_hi : HSV range for maple wood floor.
                                   Tune these per arena.
    white_value_thresh          : Minimum V channel value for court lines.
    hough_min_line_len          : Minimum Hough line segment length (px).
    hough_max_gap               : Max pixel gap within a Hough line.
    angle_tol_deg               : Lines within this angle of 0°/90° are
                                   classified as horizontal/vertical.
    """

    def __init__(
        self,
        floor_hsv_lo: Tuple = (10,  30,  80),
        floor_hsv_hi: Tuple = (35, 200, 240),
        white_value_thresh: int = 200,
        hough_min_line_len: int = 60,
        hough_max_gap: int = 20,
        angle_tol_deg: float = 15.0,
    ) -> None:
        self.floor_lo   = np.array(floor_hsv_lo, dtype=np.uint8)
        self.floor_hi   = np.array(floor_hsv_hi, dtype=np.uint8)
        self.white_thresh = white_value_thresh
        self.hough_min  = hough_min_line_len
        self.hough_gap  = hough_max_gap
        self.angle_tol  = np.deg2rad(angle_tol_deg)

    def detect(
        self,
        bgr: np.ndarray,
        full_frame_w: Optional[int] = None,
    ) -> KeypointDict:
        result: KeypointDict = {k: None for k in KEYPOINT_NAMES}

        h, w = bgr.shape[:2]
        # When called on a cropped frame, use the original full-frame width so
        # left/right keypoint assignment isn't relative to the crop midpoint.
        assign_w = full_frame_w if full_frame_w is not None else w

        # 1. Court floor mask
        floor_mask = self._segment_floor(bgr)

        # 2. White line mask (within floor)
        line_mask = self._extract_white_lines(bgr, floor_mask)

        # 3. Hough lines
        h_lines, v_lines = self._hough_lines(line_mask, w, h)

        if len(h_lines) < 2 or len(v_lines) < 2:
            logger.debug(
                "ClassicalKeyDetector: insufficient lines (H=%d, V=%d)",
                len(h_lines), len(v_lines),
            )
            return result

        # 4. Candidate corners from H×V intersections
        candidates = self._intersect(h_lines, v_lines, w, h)
        if len(candidates) < 4:
            return result

        # 5. Assign candidates to known keypoint clusters
        result.update(self._assign_to_keypoints(candidates, assign_w, h))

        n = self.count_detected(result)
        logger.debug("ClassicalKeyDetector: %d keypoints found", n)
        return result

    # ── Private pipeline steps ────────────────────────────────────────────────

    def _segment_floor(self, bgr: np.ndarray) -> np.ndarray:
        """Return a binary mask of court floor pixels."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.floor_lo, self.floor_hi)
        # Close small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Keep only the largest connected region (the court)
        return self._keep_largest_blob(mask)

    def _extract_white_lines(
        self, bgr: np.ndarray, floor_mask: np.ndarray
    ) -> np.ndarray:
        """Threshold for white court markings within the court floor region."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)
        # Only keep white pixels that are on the court floor
        return cv2.bitwise_and(white_mask, floor_mask)

    def _hough_lines(
        self,
        line_mask: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[List, List]:
        """
        Run probabilistic Hough and split into near-horizontal / near-vertical.
        Returns (h_lines, v_lines) as lists of (x1,y1,x2,y2) tuples.
        """
        edges = cv2.Canny(line_mask, 50, 150)
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=self.hough_min,
            maxLineGap=self.hough_gap,
        )
        if raw is None:
            return [], []

        h_lines, v_lines = [], []
        for seg in raw[:, 0]:
            x1, y1, x2, y2 = seg
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            if angle <= self.angle_tol or angle >= np.pi - self.angle_tol:
                h_lines.append(seg)
            elif abs(angle - np.pi / 2) <= self.angle_tol:
                v_lines.append(seg)

        return h_lines, v_lines

    @staticmethod
    def _line_intersection(
        l1: np.ndarray, l2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Compute the intersection of two line segments (extended as infinite lines)."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None   # parallel
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return ix, iy

    def _intersect(
        self,
        h_lines: List,
        v_lines: List,
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        """Return all H×V intersections that lie within the frame."""
        pts = []
        for hl in h_lines:
            for vl in v_lines:
                p = self._line_intersection(hl, vl)
                if p is None:
                    continue
                ix, iy = p
                if 0 <= ix <= frame_w and 0 <= iy <= frame_h:
                    pts.append([ix, iy])
        return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2))

    def _assign_to_keypoints(
        self,
        candidates: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> KeypointDict:
        """
        Assign candidate corner points to paint keypoint names using
        their relative position in the frame.

        This heuristic works for the common broadcast angle where:
          - Home paint occupies the left 40% of the frame
          - Away paint occupies the right 40% of the frame
          - Near corners are at the bottom, far corners higher up

        For radically different camera angles, retrain NeuralKeyDetector.
        """
        result: KeypointDict = {}
        if len(candidates) < 4:
            return result

        mid_x = frame_w / 2

        # Split candidates by screen half (left=home, right=away)
        home_cands = candidates[candidates[:, 0] < mid_x]
        away_cands = candidates[candidates[:, 0] >= mid_x]

        def assign_rect(cands, prefix_near, prefix_far):
            """Given 4 candidates forming a rectangle, assign bl/tl/br/tr."""
            if len(cands) < 4:
                return {}
            # Sort by y (row): bottom = higher y value = near baseline
            cands_sorted = cands[np.argsort(cands[:, 1])][::-1]
            near_pts = cands_sorted[:2]  # higher y = nearer to camera
            far_pts  = cands_sorted[2:]
            # Within each pair, sort by x
            near_pts = near_pts[np.argsort(near_pts[:, 0])]
            far_pts  = far_pts[np.argsort(far_pts[:, 0])]
            return {
                f"{prefix_near}_bl": near_pts[0],
                f"{prefix_near}_tl": near_pts[1],
                f"{prefix_far}_br":  far_pts[0],
                f"{prefix_far}_tr":  far_pts[1],
            }

        r = {}
        r.update(assign_rect(home_cands, "home_paint", "home_paint"))
        r.update(assign_rect(away_cands, "away_paint", "away_paint"))
        return r

    @staticmethod
    def _keep_largest_blob(mask: np.ndarray) -> np.ndarray:
        """Return a mask with only the largest connected component."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return mask
        # Ignore background (label 0)
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest).astype(np.uint8) * 255


# ── Enhanced classical detector (court crop + classical) ─────────────────────

class EnhancedClassicalDetector(BaseKeyDetector):
    """
    Wraps ClassicalKeyDetector with a court bounding-box pre-crop.

    When court_detector_yolov8n.pt is available, crops the frame to the
    detected court region before running Hough-line detection.  This removes
    crowd, scoreboards, and sideline noise that confuse the line finder.

    Keypoint coordinates are corrected back to full-frame pixel space after
    detection so the rest of the pipeline is unaffected.

    Falls back to plain ClassicalKeyDetector if the court detector weights
    are not yet available.
    """

    def __init__(self) -> None:
        self._classical = ClassicalKeyDetector()
        self._court_det = _get_court_detector()
        if self._court_det.is_ready:
            logger.info("EnhancedClassicalDetector: court detector loaded — using crop mode")
        else:
            logger.info("EnhancedClassicalDetector: no court detector weights — full-frame mode")

    def detect(self, bgr: np.ndarray) -> KeypointDict:
        if not self._court_det.is_ready:
            return self._classical.detect(bgr)

        full_h, full_w = bgr.shape[:2]
        crop, x_off, y_off = self._court_det.crop(bgr)
        result = self._classical.detect(crop, full_frame_w=full_w)

        # Shift all detected keypoints back to full-frame coordinates
        if x_off != 0 or y_off != 0:
            offset = np.array([x_off, y_off], dtype=np.float32)
            result = {
                k: (v + offset if v is not None else None)
                for k, v in result.items()
            }

        return result


# ── Neural detector ───────────────────────────────────────────────────────────

class NeuralKeyDetector(BaseKeyDetector):
    """
    Court keypoint detector using a YOLOv8-pose model fine-tuned on NBA frames.

    The model predicts all 15 canonical keypoints (KEYPOINT_NAMES) in a
    single forward pass.  Confidence scores let you filter unreliable
    detections before passing to homography.

    Weights
    ───────
    Fine-tune YOLOv8n-pose on annotated NBA court images.
    A minimal annotation dataset (~500 frames, 15 keypoints each) is
    sufficient for reliable detection.

    Annotation format
    ─────────────────
    Use YOLO-pose format: one bounding box per frame covering the full court,
    with 15 keypoints in the order defined by KEYPOINT_NAMES.

    Fallback
    ────────
    If weights_path does not exist, NeuralKeyDetector silently falls back to
    ClassicalKeyDetector so the pipeline keeps running.
    """

    def __init__(
        self,
        weights_path: str | Path = "models/checkpoints/court_kp_yolov8n.pt",
        conf_thresh: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.conf_thresh  = conf_thresh
        self.device       = device
        self._model       = None
        self._fallback    = EnhancedClassicalDetector()
        self._using_neural = False

        weights_path = Path(weights_path)
        if weights_path.exists():
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(weights_path))
                self._using_neural = True
                logger.info("NeuralKeyDetector: loaded %s", weights_path)
            except Exception as e:
                logger.warning("NeuralKeyDetector: failed to load model (%s) — using fallback", e)
        else:
            logger.info(
                "NeuralKeyDetector: weights not found at %s — using ClassicalKeyDetector fallback",
                weights_path,
            )

    def detect(self, bgr: np.ndarray) -> KeypointDict:
        if not self._using_neural or self._model is None:
            return self._fallback.detect(bgr)
        return self._detect_neural(bgr)

    def _detect_neural(self, bgr: np.ndarray) -> KeypointDict:
        result: KeypointDict = {k: None for k in KEYPOINT_NAMES}
        try:
            preds = self._model.predict(bgr, conf=self.conf_thresh, verbose=False)
            for r in preds:
                if r.keypoints is None or len(r.keypoints) == 0:
                    continue
                kps   = r.keypoints.xy.cpu().numpy()[0]    # (15, 2)
                confs = r.keypoints.conf.cpu().numpy()[0]  # (15,)
                for i, name in enumerate(KEYPOINT_NAMES):
                    if i < len(kps) and confs[i] >= self.conf_thresh:
                        result[name] = kps[i].astype(np.float32)
        except Exception as e:
            logger.error("NeuralKeyDetector inference failed: %s — using fallback", e)
            return self._fallback.detect(bgr)

        n = self.count_detected(result)
        logger.debug("NeuralKeyDetector: %d keypoints found", n)
        return result

    @property
    def using_neural(self) -> bool:
        return self._using_neural
