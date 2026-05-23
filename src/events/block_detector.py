"""
src/events/block_detector.py

Detects blocks: a defender's body (specifically their upper bbox region,
proxying for raised hands) intersects the ball while it is on an upward
trajectory, and the ball's direction changes sharply within 5 frames.

Height proxy
────────────
Court coordinates are 2D (floor plane).  Ball height is approximated by
ball pixel-y: in broadcast footage, a ball moving upward moves toward the
top of the frame (pixel_y DECREASING).  This breaks down near the edges of
the frame where perspective distortion is high, but is reliable in the
central 80% of a broadcast frame.

Two-phase detection
───────────────────
Phase 1 (candidate):  ball pixel_y is decreasing (rising) AND a defender's
                      bbox upper region overlaps the ball pixel position.
Phase 2 (confirm):    within 5 frames of the candidate, ball direction
                      changes by > dir_change_thresh degrees.

This two-phase approach suppresses false positives from arcing passes that
look like rising balls near a defender.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from src.events.event import Event, EventType, PipelineState
from src.events.event_detector import BaseEventDetector
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)


class BlockDetector(BaseEventDetector):
    """
    Parameters
    ──────────
    min_rise_speed_px    : Minimum ball upward speed (px/frame) to consider.
    dir_change_thresh_deg: Ball direction change required for confirmation.
    confirm_window       : Frames to wait for direction change confirmation.
    upper_bbox_frac      : Fraction of bbox from the top considered "hands up".
    cooldown_frames      : Minimum frames between successive block events.
    """

    def __init__(
        self,
        min_rise_speed_px: float = 8.0,
        dir_change_thresh_deg: float = 90.0,
        confirm_window: int = 5,
        upper_bbox_frac: float = 0.30,
        cooldown_frames: int = 45,
        max_ball_defender_dist_px: float = 80.0,
    ) -> None:
        self.min_rise_speed    = min_rise_speed_px
        self.dir_thresh        = np.deg2rad(dir_change_thresh_deg)
        self.confirm_window    = confirm_window
        self.upper_frac        = upper_bbox_frac
        self.cooldown          = cooldown_frames
        self.max_dist_px       = max_ball_defender_dist_px

        # Rolling ball pixel positions for velocity/direction analysis
        self._ball_px: deque[np.ndarray] = deque(maxlen=10)
        # Pending block: (candidate_frame_idx, defender_track_id)
        self._pending: Optional[Tuple[int, int]] = None
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

        # ── Phase 2: check pending block for direction change confirmation ────
        if self._pending is not None:
            cand_frame, defender_id = self._pending
            frames_since = state.frame_idx - cand_frame

            if frames_since <= self.confirm_window:
                if self._direction_changed():
                    if state.frame_idx - self._last_event_frame >= self.cooldown:
                        events.append(Event(
                            event_type=EventType.BLOCK,
                            frame_idx=cand_frame,
                            timestamp_sec=state.timestamp_sec,
                            primary_player_id=defender_id,
                            secondary_player_id=state.ball_possessor_id,
                            court_pos=(state.ball_court_pos.copy()
                                       if state.ball_court_pos is not None else None),
                            confidence=0.70,
                            metadata={"confirmed_at_frame": state.frame_idx},
                        ))
                        self._last_event_frame = cand_frame
                        logger.debug("BLOCK confirmed: defender=%d", defender_id)
                    self._pending = None
            else:
                self._pending = None   # confirmation window expired

        # ── Phase 1: look for a new block candidate ───────────────────────────
        if self._pending is None and state.frame_idx - self._last_event_frame >= self.cooldown:
            ball_vel = self._ball_velocity()
            if ball_vel is not None:
                vx     = ball_vel[0]
                vy     = ball_vel[1]
                speed  = float(np.linalg.norm(ball_vel))
                # vy < 0 means ball moving toward top of frame = rising
                if vy < -self.min_rise_speed and speed > self.min_rise_speed:
                    # Require meaningful horizontal motion — a dribble bounce is
                    # mostly vertical (|vx| ≈ 0); a shot directed at the basket
                    # always has a significant horizontal component.
                    if abs(vx) >= 0.20 * speed:
                        defender = self._find_defender_overlap(state)
                        if defender is not None:
                            self._pending = (state.frame_idx, defender.track_id)
                            logger.debug(
                                "BLOCK candidate: defender=%d (vy=%.1f vx=%.1f)",
                                defender.track_id, vy, vx,
                            )

        return events

    def reset(self) -> None:
        self._ball_px.clear()
        self._pending = None
        self._last_event_frame = -9999

    # ── Private ───────────────────────────────────────────────────────────────

    def _ball_velocity(self) -> Optional[np.ndarray]:
        if len(self._ball_px) < 2:
            return None
        px = list(self._ball_px)
        return px[-1] - px[-2]

    def _direction_changed(self) -> bool:
        """Check if the most recent ball trajectory changed direction sharply."""
        if len(self._ball_px) < 4:
            return False
        px = list(self._ball_px)
        v1 = px[-3] - px[-4]
        v2 = px[-1] - px[-2]
        s1, s2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if s1 < 1.0 or s2 < 1.0:
            return False
        cos_a = np.clip(np.dot(v1, v2) / (s1 * s2), -1.0, 1.0)
        return float(np.arccos(cos_a)) >= self.dir_thresh

    def _find_defender_overlap(self, state: PipelineState):
        """
        Find a defender whose upper bbox region contains the ball pixel center
        AND is within max_dist_px of the ball center.
        Returns the closest such defender Track or None.
        """
        if not state.player_tracks or state.ball_track is None:
            return None

        ball_cx, ball_cy = state.ball_track.center
        off_team = state.offensive_team_id
        best, best_dist = None, float("inf")

        for track in state.player_tracks:
            team = state.team_assignments.get(track.track_id)
            if team == off_team or team is None:
                continue  # same side as possessor or unclassified → skip

            x1, y1, x2, y2 = track.bbox
            # Proximity gate: ball must be near this defender
            cx = (x1 + x2) / 2.0
            cy = y1  # top of bbox ≈ raised hands
            dist = float(np.hypot(ball_cx - cx, ball_cy - cy))
            if dist > self.max_dist_px:
                continue

            upper_y = y1 + (y2 - y1) * self.upper_frac
            if x1 <= ball_cx <= x2 and ball_cy <= upper_y:
                if dist < best_dist:
                    best, best_dist = track, dist

        return best
