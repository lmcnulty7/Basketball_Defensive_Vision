"""
src/detection/ball_detector.py

Detects the basketball in each frame, with Kalman-filter extrapolation
for occluded frames.

Two model modes:
  fine_tuned  — loads ball-specific YOLOv8 weights (best accuracy).
  coco_fallback — uses COCO class 32 (sports_ball) when fine-tuned weights
                  are absent.  Reliable enough to start development.

When neither mode detects the ball, the Kalman filter predicts its position
from recent velocity history.  The caller can see whether the returned
position is observed or predicted via the is_predicted flag.

The Kalman filter resets automatically after max_predict_frames consecutive
frames with no observation, because stale trajectory extrapolation misleads
downstream event detectors more than returning None.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection.postprocess import BALL_CLASS_ID, Detection

logger = logging.getLogger(__name__)

_COCO_SPORTS_BALL = 32


# ── Kalman filter ─────────────────────────────────────────────────────────────

class KalmanBallFilter:
    """
    6-state constant-acceleration Kalman filter for the basketball.

    State  : [cx, cy, vx, vy, ax, ay]
    Observe: [cx, cy]

    Upgraded from 4-state (constant-velocity) to 6-state so the filter can
    track curved trajectories — shot arcs, bouncing passes, lobs — without
    pulling the prediction toward a straight line during occlusion.

    During a shot the ball follows a parabola; a constant-velocity model
    diverges within 5–8 frames.  The acceleration state lets the filter
    learn the arc from recent observations and extrapolate it accurately
    through the net-entry occlusion window.
    """

    def __init__(self) -> None:
        # State transition: pos += vel*dt + 0.5*acc*dt², vel += acc*dt  (dt=1 frame)
        self.F = np.array([
            [1, 0, 1, 0, 0.5, 0  ],
            [0, 1, 0, 1, 0,   0.5],
            [0, 0, 1, 0, 1,   0  ],
            [0, 0, 0, 1, 0,   1  ],
            [0, 0, 0, 0, 1,   0  ],
            [0, 0, 0, 0, 0,   1  ],
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=float)

        # Process noise: position tight, velocity moderate, acceleration loose
        # (acceleration changes abruptly at contact; between contacts it's smooth)
        self.Q = np.diag([1.0, 1.0, 8.0, 8.0, 4.0, 4.0])
        self.R = np.diag([5.0, 5.0])   # ~2-pixel measurement uncertainty

        self._x: Optional[np.ndarray] = None    # state estimate (6,)
        self._P: Optional[np.ndarray] = None    # error covariance (6,6)
        self.initialized = False
        self.frames_since_update = 0

    def initialize(self, cx: float, cy: float) -> None:
        self._x = np.array([cx, cy, 0.0, 0.0, 0.0, 0.0])
        self._P = np.eye(6) * 100.0
        self.initialized = True
        self.frames_since_update = 0

    def update(self, cx: float, cy: float) -> np.ndarray:
        """Predict + correct with a new observed center.  Returns updated state."""
        if not self.initialized:
            self.initialize(cx, cy)
            return self._x.copy()

        x_pred = self.F @ self._x
        P_pred = self.F @ self._P @ self.F.T + self.Q

        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        z = np.array([cx, cy])
        self._x = x_pred + K @ (z - self.H @ x_pred)
        self._P = (np.eye(6) - K @ self.H) @ P_pred
        self.frames_since_update = 0
        return self._x.copy()

    def predict(self) -> Optional[np.ndarray]:
        """
        Advance state one frame with no observation.
        Returns current state estimate, or None if never initialized.
        """
        if not self.initialized:
            return None
        self._x = self.F @ self._x
        self._P = self.F @ self._P @ self.F.T + self.Q
        self.frames_since_update += 1
        return self._x.copy()

    @property
    def position(self) -> Optional[np.ndarray]:
        """Current [cx, cy] estimate, or None if uninitialized."""
        return self._x[:2].copy() if self._x is not None else None

    @property
    def velocity(self) -> Optional[np.ndarray]:
        """Current [vx, vy] estimate, or None if uninitialized."""
        return self._x[2:4].copy() if self._x is not None else None

    def reset(self) -> None:
        self._x = None
        self._P = None
        self.initialized = False
        self.frames_since_update = 0


# ── Ball detector ─────────────────────────────────────────────────────────────

class BallDetector:
    """
    Basketball detector with Kalman-filter fallback for occlusions.

    Parameters
    ──────────
    weights_path       : Fine-tuned ball model weights.  If the file does not
                         exist, falls back to COCO sports_ball via fallback_weights.
    fallback_weights   : Base YOLOv8 weights for the COCO-fallback mode.
    conf_thresh        : Minimum model confidence to accept a detection.
    iou_thresh         : NMS IoU threshold.
    device             : "cpu" | "cuda" | "mps"
    max_predict_frames : Reset the Kalman filter after this many consecutive
                         frames with no observed ball.
    """

    # Ball diameter ≈ 0.5–3% of typical broadcast frame width.
    # These bounds gate out noise (< 8px) and whole-player detections (> 15% frame).
    _MIN_DIM_PX    = 8
    _MAX_FRAME_PCT = 0.15
    _MIN_ASPECT    = 0.4      # reject boxes much taller or wider than they are square

    def __init__(
        self,
        weights_path: str | Path = "models/checkpoints/ball_yolo.pt",
        fallback_weights: str | Path = "yolov8m.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: str = "cpu",
        max_predict_frames: int = 20,
    ) -> None:
        from ultralytics import YOLO

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.max_predict_frames = max_predict_frames
        self._kalman = KalmanBallFilter()

        weights_path = Path(weights_path)

        if weights_path.exists():
            logger.info("BallDetector: loading fine-tuned weights: %s", weights_path)
            self._model = YOLO(str(weights_path))
            self._target_classes = None
            self._mode = "fine_tuned"
        else:
            # No ball-specific weights — run Kalman-only mode (no model loaded).
            # Loading the player model as a fallback creates a second YOLO instance
            # in memory which causes a segfault on some platforms.
            # Add models/checkpoints/ball_yolo.pt to enable ball detection.
            logger.warning(
                "BallDetector: %s not found — ball detection disabled (Kalman-only).",
                weights_path,
            )
            self._model = None
            self._target_classes = None
            self._mode = "disabled"

        logger.info("BallDetector ready (mode=%s, device=%s)", self._mode, device)

    @classmethod
    def from_config(
        cls,
        cfg: Dict,
        player_weights: str,
        device: str = "cpu",
    ) -> "BallDetector":
        """Construct from the `detection` section of models.yaml."""
        return cls(
            weights_path=cfg["ball_weights"],
            fallback_weights=player_weights,
            conf_thresh=cfg["ball_conf"],
            iou_thresh=cfg["iou_threshold"],
            device=device,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        bgr: np.ndarray,
    ) -> Tuple[Optional[Detection], bool]:
        """
        Detect the basketball in a single BGR frame.

        Returns
        ───────
        (detection, is_predicted)
          detection    : Detection with class_name "ball" or "ball_predicted",
                         or None if the ball is completely lost.
          is_predicted : True when the position is a Kalman extrapolation.
        """
        if self._mode == "disabled":
            return None, False

        h, w = bgr.shape[:2]
        best = self._color_detect(bgr, w, h) if self._mode == "color_based" else self._run_model(bgr, w, h)

        if best is not None:
            cx, cy = float(best.center[0]), float(best.center[1])
            self._kalman.update(cx, cy)
            return best, False

        # No model detection — try Kalman prediction
        if self._kalman.frames_since_update > self.max_predict_frames:
            self._kalman.reset()
            return None, False

        state = self._kalman.predict()
        if state is None:
            return None, False

        cx, cy = float(state[0]), float(state[1])
        half = 15.0   # synthetic box half-size in pixels
        pred = Detection(
            bbox=np.array(
                [cx - half, cy - half, cx + half, cy + half],
                dtype=np.float32,
            ),
            confidence=0.0,   # no model confidence for Kalman predictions
            class_id=BALL_CLASS_ID,
            class_name="ball_predicted",
        )
        logger.debug(
            "Ball occluded — Kalman prediction (%d frames since last obs)",
            self._kalman.frames_since_update,
        )
        return pred, True

    def attach_model(self, model, target_class: int = _COCO_SPORTS_BALL) -> None:
        """
        Reuse an already-loaded YOLO model for COCO sports-ball detection.

        Call this instead of loading a second YOLO instance (which causes
        a segfault on some platforms due to duplicate OpenMP runtimes).
        """
        self._model = model
        self._target_classes = [target_class]
        self._mode = "coco_fallback"
        logger.info(
            "BallDetector: sharing player model for COCO class %d (sports ball)",
            target_class,
        )

    def enable_color_detection(self) -> None:
        """
        Enable HSV color-based ball detection as the primary fallback.

        NBA basketballs are distinctively orange (HSV H≈10-20°) and circular.
        This mode needs no model and works reliably on broadcast footage where
        COCO class-32 detections are too low-confidence.
        """
        self._mode = "color_based"
        logger.info("BallDetector: HSV color-based detection enabled")

    def reset(self) -> None:
        """Reset the Kalman filter (call at start of each new possession)."""
        self._kalman.reset()

    @property
    def mode(self) -> str:
        return self._mode

    # ── Private ───────────────────────────────────────────────────────────────

    def _color_detect(
        self,
        bgr: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Optional[Detection]:
        """
        Detect the basketball via HSV orange thresholding + circularity filter.

        Works on broadcast footage where COCO class-32 confidence is too low.
        NBA balls: H 5–20° (OpenCV 0–180), S 150–255, V 120–255.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo  = np.array([5,  150, 120], dtype=np.uint8)
        hi  = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)

        # Black out the scoreboard band (bottom ~15%) — orange team logos
        # and ESPN graphics trigger false positives in that strip.
        mask[int(frame_h * 0.85):, :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score, best_det = -1.0, None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circularity = 4.0 * np.pi * area / (perim ** 2)
            if circularity < 0.55:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            det = Detection(
                bbox=np.array([x, y, x + bw, y + bh], dtype=np.float32),
                confidence=float(circularity),
                class_id=BALL_CLASS_ID,
                class_name="ball",
            )
            if not self._is_plausible(det, frame_w, frame_h):
                continue
            if circularity > best_score:
                best_score = circularity
                best_det   = det

        return best_det

    def _run_model(
        self,
        bgr: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Optional[Detection]:
        if self._model is None:
            return None
        results = self._model.predict(
            bgr,
            classes=self._target_classes,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )

        candidates: list[Detection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes   = r.boxes.xyxy.cpu().numpy()
            confs   = r.boxes.conf.cpu().numpy()

            for bbox, conf in zip(boxes, confs):
                det = Detection(
                    bbox=bbox.astype(np.float32),
                    confidence=float(conf),
                    class_id=BALL_CLASS_ID,
                    class_name="ball",
                )
                if self._is_plausible(det, frame_w, frame_h):
                    candidates.append(det)

        return max(candidates, key=lambda d: d.confidence) if candidates else None

    def _is_plausible(
        self,
        det: Detection,
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """
        Reject boxes that cannot be a basketball by size and shape.

        A broadcast-distance NBA ball spans roughly 0.5–3% of frame width.
        """
        max_dim = max(det.width, det.height)
        min_dim = min(det.width, det.height)

        if max_dim < self._MIN_DIM_PX:
            return False
        if max_dim > frame_w * self._MAX_FRAME_PCT:
            return False
        if min_dim < max_dim * self._MIN_ASPECT:
            return False
        return True


# ── TrackNetV4 detector ────────────────────────────────────────────────────────

class TrackNetDetector:
    """
    Basketball detector using TrackNetV3 (pretrained shuttlecock weights).

    Buffers SEQ_LEN=8 consecutive BGR frames, prepends an approximate background
    frame (oldest buffered frame), and runs the TrackNetV3 forward pass.
    Uses the last output heatmap channel (frame 8 = most recent).

    Falls back to Kalman extrapolation when peak confidence is below threshold —
    identical fallback contract as BallDetector.

    Parameters
    ──────────
    weights_path : Path to tracknet_best.pt.
    conf_thresh  : Heatmap peak threshold to accept a detection.
    device       : "cpu" | "cuda" | "mps"
    max_predict  : Kalman frames before filter resets.
    """

    SEQ_LEN = 8   # must match pretrained weights

    def __init__(
        self,
        weights_path: str | Path = "models/checkpoints/tracknet_best.pt",
        conf_thresh: float = 0.50,
        seq_len: int = 8,   # kept for API compat; value ignored, always 8
        device: str = "cpu",
        max_predict: int = 20,
    ) -> None:
        from src.detection.tracknet import TrackNetV3, load_pretrained

        self.conf_thresh  = conf_thresh
        self.seq_len      = self.SEQ_LEN
        self.device       = device
        self._max_predict = max_predict
        self._kalman      = KalmanBallFilter()

        self._model = TrackNetV3()
        load_pretrained(self._model, weights_path, device=device)
        self._model.to(device)
        self._model.eval()

        self._input_h = self._model.INPUT_H   # 288
        self._input_w = self._model.INPUT_W   # 512

        logger.info(
            "TrackNetDetector ready (device=%s  seq_len=%d  conf=%.2f)",
            device, self.seq_len, conf_thresh,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        frames: List[np.ndarray],
    ) -> Tuple[Optional[Detection], bool]:
        """
        Detect the basketball from a buffer of consecutive BGR frames.

        Parameters
        ──────────
        frames : List of BGR frames, most-recent last.  Padded at startup when
                 fewer than SEQ_LEN are available.

        Returns
        ───────
        (detection, is_predicted) — same contract as BallDetector.detect().
        """
        if not frames:
            return self._kalman_fallback()

        # Pad at pipeline start when buffer isn't full yet
        while len(frames) < self.SEQ_LEN:
            frames = [frames[0]] + frames
        frames = list(frames[-self.SEQ_LEN:])

        orig_h, orig_w = frames[-1].shape[:2]
        tensor = self._preprocess(frames)

        import torch
        with torch.no_grad():
            # Output: (1, SEQ_LEN, H, W) — use last channel (current frame)
            out = self._model(tensor.to(self.device))

        heatmap = out[0, -1].cpu().numpy()   # (H, W)

        # Suppress scoreboard band (bottom 15%) and side margins (outer 3%)
        # to avoid false positives on circular UI graphics in the broadcast.
        h_mask = int(self._input_h * 0.85)
        w_lo   = int(self._input_w * 0.03)
        w_hi   = int(self._input_w * 0.97)
        mask   = np.zeros_like(heatmap)
        mask[:h_mask, w_lo:w_hi] = heatmap[:h_mask, w_lo:w_hi]

        peak_conf = float(mask.max())

        if peak_conf < self.conf_thresh:
            return self._kalman_fallback()

        flat_idx = int(mask.argmax())
        peak_row = flat_idx // self._input_w
        peak_col = flat_idx %  self._input_w

        cx = peak_col * orig_w / self._input_w
        cy = peak_row * orig_h / self._input_h

        self._kalman.update(cx, cy)

        radius = max(8.0, orig_w * 0.015)
        return Detection(
            bbox=np.array(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                dtype=np.float32,
            ),
            confidence=peak_conf,
            class_id=BALL_CLASS_ID,
            class_name="ball",
        ), False

    def reset(self) -> None:
        self._kalman.reset()

    @property
    def mode(self) -> str:
        return "tracknet"

    # ── Private ───────────────────────────────────────────────────────────────

    def _preprocess(self, frames: List[np.ndarray]) -> "torch.Tensor":
        """
        Build (1, 27, H, W) input tensor: [bg, f1..f8] each (3, H, W).
        Background = oldest frame in the buffer (a simple approximation of
        the static background used during training).
        """
        import torch

        def _to_tensor(bgr: np.ndarray) -> "torch.Tensor":
            resized = cv2.resize(bgr, (self._input_w, self._input_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W)

        bg = _to_tensor(frames[0])   # oldest frame as background proxy
        seq = [_to_tensor(f) for f in frames]
        return torch.cat([bg] + seq, dim=0).unsqueeze(0)  # (1, 27, H, W)

    def _kalman_fallback(self) -> Tuple[Optional[Detection], bool]:
        if self._kalman.frames_since_update > self._max_predict:
            self._kalman.reset()
            return None, False
        state = self._kalman.predict()
        if state is None:
            return None, False
        cx, cy = float(state[0]), float(state[1])
        half   = 15.0
        return Detection(
            bbox=np.array(
                [cx - half, cy - half, cx + half, cy + half],
                dtype=np.float32,
            ),
            confidence=0.0,
            class_id=BALL_CLASS_ID,
            class_name="ball_predicted",
        ), True
