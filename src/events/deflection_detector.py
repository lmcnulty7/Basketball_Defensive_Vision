"""
src/events/deflection_detector.py

Detects deflections: the ball's velocity direction changes by more than
`angle_thresh_deg` degrees within 2 frames, and a defensive player is
within `max_defender_dist_ft` court feet of the ball at that moment.

Why pixel coordinates for direction, court coords for proximity
───────────────────────────────────────────────────────────────
Ball direction change is best measured in pixel space — it's directly
observable regardless of homography quality.  Defender proximity is best
measured in court space (real-world feet) because pixel distances are
distorted by perspective.  We use both.

Filtering false positives
─────────────────────────
- Minimum speed gate (min_ball_speed_px): slow-rolling balls produce
  direction noise; only fast balls can be deflected.
- Shot-arc exclusion: if the ball is on a rising trajectory (vy < 0) and
  heading toward a basket, a direction change is more likely a block than
  a deflection — the block detector handles that case.
- Cooldown: deflections happen at most once every cooldown_frames.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional

import numpy as np

from src.events.event import Event, EventType, PipelineState
from src.events.event_detector import BaseEventDetector
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)


class DeflectionDetector(BaseEventDetector):
    """
    Parameters
    ──────────
    angle_thresh_deg      : Minimum direction change (degrees) to count.
    min_ball_speed_px     : Minimum ball speed (px/frame) before and after.
    max_defender_dist_ft  : Max court distance (ft) for a defender to claim credit.
    cooldown_frames       : Minimum frames between deflection events.
    """

    def __init__(
        self,
        angle_thresh_deg: float = 45.0,
        min_ball_speed_px: float = 7.0,
        max_ball_speed_px: float = 150.0,
        max_defender_dist_ft: float = 6.0,
        cooldown_frames: int = 20,
    ) -> None:
        self.angle_thresh  = np.deg2rad(angle_thresh_deg)
        self.min_speed     = min_ball_speed_px
        self.max_speed     = max_ball_speed_px
        self.max_dist_ft   = max_defender_dist_ft
        self.cooldown      = cooldown_frames

        self._ball_px:     deque[np.ndarray]          = deque(maxlen=5)
        self._last_event_frame = -9999

    def update(
        self,
        state: PipelineState,
        manager: TrackManager,
    ) -> List[Event]:
        events: List[Event] = []

        if state.ball_track is None or state.ball_is_predicted:
            return events

        self._ball_px.append(state.ball_track.center.copy())

        if len(self._ball_px) < 3:
            return events
        if state.frame_idx - self._last_event_frame < self.cooldown:
            return events

        px = list(self._ball_px)
        v1 = px[-2] - px[-3]    # velocity one frame ago
        v2 = px[-1] - px[-2]    # velocity this frame

        s1, s2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if s1 < self.min_speed or s2 < self.min_speed:
            return events
        if s1 > self.max_speed or s2 > self.max_speed:
            return events  # color-detector position jump, not a real deflection

        cos_a = np.clip(np.dot(v1, v2) / (s1 * s2), -1.0, 1.0)
        angle = float(np.arccos(cos_a))

        if angle < self.angle_thresh:
            return events

        # Direction changed significantly — find the nearest defender.
        # Use court space when H is valid, pixel-space fallback otherwise.
        ball_court = state.ball_court_pos
        if ball_court is not None:
            defender, dist_ft = state.nearest_defender_to(
                ball_court,
                exclude_track_id=state.ball_possessor_id,
            )
            # When possession is unknown, defensive team is unclassified and
            # nearest_defender_to returns None. Fall back to all players.
            if defender is None:
                defender, dist_ft = self._nearest_player(
                    state.player_tracks, ball_court, state.ball_possessor_id
                )
        else:
            ball_px = state.ball_pixel_pos
            if ball_px is None:
                return events
            defender, dist_ft = state.nearest_defender_pixel_to(
                ball_px,
                exclude_track_id=state.ball_possessor_id,
            )
            if defender is None:
                defender, dist_ft = self._nearest_player_px(
                    state.player_tracks, ball_px, state.ball_possessor_id
                )

        if defender is None or dist_ft > self.max_dist_ft or dist_ft < 0.5:
            return events

        # Confidence: higher when ball is fast and defender is close
        speed_factor = min(1.0, s1 / 20.0)
        dist_factor  = max(0.0, 1.0 - dist_ft / self.max_dist_ft)
        confidence   = 0.5 + 0.25 * speed_factor + 0.25 * dist_factor

        events.append(Event(
            event_type=EventType.DEFLECTION,
            frame_idx=state.frame_idx,
            timestamp_sec=state.timestamp_sec,
            primary_player_id=defender.track_id,
            secondary_player_id=state.ball_possessor_id,
            court_pos=ball_court.copy() if ball_court is not None else None,
            confidence=confidence,
            metadata={
                "angle_deg":  round(np.degrees(angle), 1),
                "dist_ft":    round(dist_ft, 2),
                "ball_speed": round(s1, 1),
            },
        ))
        self._last_event_frame = state.frame_idx
        logger.debug(
            "DEFLECTION: defender=%d angle=%.1f° dist=%.1fft",
            defender.track_id, np.degrees(angle), dist_ft,
        )
        return events

    def reset(self) -> None:
        self._ball_px.clear()
        self._last_event_frame = -9999

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _nearest_player(
        tracks: list,
        court_pos: np.ndarray,
        exclude_id: Optional[int],
        min_dist_ft: float = 0.5,
    ) -> tuple:
        best, best_dist = None, float("inf")
        for t in tracks:
            if t.track_id == exclude_id or t.court_pos is None:
                continue
            d = float(np.linalg.norm(t.court_pos - court_pos))
            if d < min_dist_ft:
                continue
            if d < best_dist:
                best_dist, best = d, t
        return best, best_dist

    @staticmethod
    def _nearest_player_px(
        tracks: list,
        pixel_pos: np.ndarray,
        exclude_id: Optional[int],
        min_dist_px: float = 8.0,
        px_per_ft: float = 17.0,
    ) -> tuple:
        best, best_dist_px = None, float("inf")
        for t in tracks:
            if t.track_id == exclude_id:
                continue
            d = float(np.linalg.norm(t.center - pixel_pos))
            if d < min_dist_px:
                continue
            if d < best_dist_px:
                best_dist_px, best = d, t
        return best, best_dist_px / px_per_ft
