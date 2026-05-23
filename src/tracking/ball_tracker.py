"""
src/tracking/ball_tracker.py

Tracks the basketball across frames and infers which player has possession.

Wraps BallDetector (which already contains the Kalman filter) and adds:
  1. Rolling position history for velocity computation by stats/speed_calculator.py
  2. Possession inference — which player's bbox is closest to the ball center.

Possession vs. dribble vs. loose ball
──────────────────────────────────────
"Possession" here means the ball center is within POSSESSION_DIST_PX pixels
of a player's bounding box center.  This is a coarse approximation — a player
dribbling in front of their body has the ball ~30–50px from their center at
broadcast resolution.  A steal occurs when the possessor switches from one
team to the other without a shot or pass event in between.

For more precise possession (ball-in-hand vs. near-player), the events module
uses pose keypoints (wrist proximity to ball).  This tracker's job is just to
provide a continuous, smooth possession estimate for coarse event triggering.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from src.detection.ball_detector import BallDetector, TrackNetDetector
from src.tracking.multi_tracker import BALL_TRACK_ID, Track

logger = logging.getLogger(__name__)

POSSESSION_DIST_PX = 90   # pixels — ball center within this of player center = possession


class BallTracker:
    """
    Ball tracking with possession inference.

    Parameters
    ──────────
    ball_detector  : BallDetector instance (already initialized with weights).
    history_len    : Frames of ball position to retain.  30 frames at stride=3
                     covers 3 seconds of real time — enough for shot-arc analysis.

    Example
    ───────
        ball_det = BallDetector(...)
        ball_tracker = BallTracker(ball_det)

        for frame in video_reader:
            ball_track, is_predicted, possessor_id = ball_tracker.update(
                frame.data, player_tracks, frame.frame_idx, frame.timestamp_sec
            )
    """

    def __init__(
        self,
        ball_detector: "BallDetector | TrackNetDetector",
        history_len: int = 30,
        seq_len: int = 8,
    ) -> None:
        self._detector = ball_detector
        self._history: deque[Track] = deque(maxlen=history_len)
        self.history_len = history_len
        buf_len = getattr(ball_detector, "SEQ_LEN", seq_len)
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=buf_len)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        bgr: np.ndarray,
        player_tracks: List[Track],
        frame_idx: int = 0,
        timestamp_sec: float = 0.0,
    ) -> Tuple[Optional[Track], bool, Optional[int]]:
        """
        Detect/predict ball position and infer possession.

        Parameters
        ──────────
        bgr           : Raw BGR frame.
        player_tracks : Active player Track objects from MultiTracker this frame.
        frame_idx     : Absolute frame index.
        timestamp_sec : Frame timestamp.

        Returns
        ───────
        (ball_track, is_predicted, possessor_id)
          ball_track    : Track with track_id=BALL_TRACK_ID, or None.
          is_predicted  : True when position is Kalman-extrapolated.
          possessor_id  : track_id of the player with possession, or None.
        """
        self._frame_buffer.append(bgr)
        if isinstance(self._detector, TrackNetDetector):
            ball_det, is_predicted = self._detector.detect(list(self._frame_buffer))
        else:
            ball_det, is_predicted = self._detector.detect(bgr)

        if ball_det is None:
            return None, False, None

        ball_track = Track(
            track_id=BALL_TRACK_ID,
            bbox=ball_det.bbox.copy(),
            center=ball_det.center.copy(),
            confidence=ball_det.confidence,
            class_id=ball_det.class_id,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
        )

        self._history.append(ball_track)

        possessor_id = self._infer_possessor(ball_track, player_tracks)
        if possessor_id is not None:
            logger.debug(
                "Frame %d: ball possession → track_id=%d", frame_idx, possessor_id
            )

        return ball_track, is_predicted, possessor_id

    def get_history(self, n: Optional[int] = None) -> List[Track]:
        """Return the last n frames of ball Track history (all frames if n=None)."""
        hist = list(self._history)
        return hist[-n:] if n is not None else hist

    def get_velocity(self, n_frames: int = 5) -> Optional[np.ndarray]:
        """
        Estimate ball velocity in px/frame from recent history.

        Returns [vx, vy] or None if fewer than 2 frames are available.
        The events module uses this to detect shot arcs (vy < 0 = rising ball).
        """
        hist = self.get_history(n_frames)
        if len(hist) < 2:
            return None
        delta_pos = hist[-1].center - hist[0].center
        delta_t = max(1, hist[-1].frame_idx - hist[0].frame_idx)
        return delta_pos / float(delta_t)

    def get_smoothed_velocity(self) -> Optional[np.ndarray]:
        """
        Savitzky-Golay smoothed velocity estimate from recent position history.

        More reliable than raw frame-to-frame differences because SG preserves
        the shape of the ball's trajectory (parabolic for shots, sinusoidal for
        dribbles) while suppressing per-frame detection noise.

        Returns [vx, vy] px/frame or None if not enough history.
        """
        hist = list(self._history)
        n = len(hist)
        if n < 4:
            return self.get_velocity()

        xs = np.array([t.center[0] for t in hist], dtype=float)
        ys = np.array([t.center[1] for t in hist], dtype=float)

        # Window must be odd and <= n; polynomial order must be < window
        wl = min(n, 11)
        if wl % 2 == 0:
            wl -= 1
        if wl < 3:
            return self.get_velocity()

        try:
            from scipy.signal import savgol_filter
            xs = savgol_filter(xs, wl, polyorder=2)
            ys = savgol_filter(ys, wl, polyorder=2)
        except Exception:
            pass  # scipy unavailable — fall back to raw positions

        vx = float(xs[-1] - xs[-2])
        vy = float(ys[-1] - ys[-2])
        return np.array([vx, vy], dtype=np.float32)

    def reset(self) -> None:
        """Reset Kalman filter and history (call at start of each new possession)."""
        self._detector.reset()
        self._history.clear()
        self._frame_buffer.clear()

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_possessor(
        ball_track: Track,
        player_tracks: List[Track],
    ) -> Optional[int]:
        """
        Return the track_id of the player most likely holding the ball.

        Two-pass logic:
          Pass 1 — bbox containment: ball center is inside (or just outside)
                   a player's bounding box.  This correctly assigns possession
                   to the dribbler even when a defender's center is closer.
          Pass 2 — center proximity fallback: closest player within
                   POSSESSION_DIST_PX if no bbox overlap found.
        """
        if not player_tracks:
            return None

        bx = float(ball_track.center[0])
        by = float(ball_track.center[1])

        # Pass 1: ball center inside a player's bbox (with 10% padding)
        best_containment_id: Optional[int] = None
        best_containment_score = -float("inf")
        for pt in player_tracks:
            x1, y1, x2, y2 = pt.bbox
            w = x2 - x1
            h = y2 - y1
            px, py = w * 0.10, h * 0.10   # 10% padding
            if (x1 - px) <= bx <= (x2 + px) and (y1 - py) <= by <= (y2 + py):
                # Score = how far inside the bbox the ball is (larger = more inside)
                score = min(bx - x1, x2 - bx) + min(by - y1, y2 - by)
                if score > best_containment_score:
                    best_containment_score = score
                    best_containment_id = pt.track_id

        if best_containment_id is not None:
            return best_containment_id

        # Pass 2: nearest center fallback
        best_dist = float("inf")
        best_id: Optional[int] = None
        for pt in player_tracks:
            dist = float(np.linalg.norm(pt.center - ball_track.center))
            if dist < best_dist:
                best_dist = dist
                best_id = pt.track_id

        return best_id if best_dist <= POSSESSION_DIST_PX else None
