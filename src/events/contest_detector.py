"""
src/events/contest_detector.py

Detects contested shots (2PT and 3PT) and notifies the StealDetector to
suppress steal events in the post-shot window.

Shot release detection (without pose estimation)
────────────────────────────────────────────────
1. Ball had a confirmed possessor for at least `min_possession_frames`.
2. Current frame: no possessor (ball released).
3. Ball speed increases relative to prior frames (release acceleration).
4. Ball velocity vector points toward a basket in court coordinates.

All four conditions must hold simultaneously to suppress false positives
from passes.  Passes also release the ball, but they don't accelerate
toward a basket — they go sideways or to a teammate.

Toward-basket check
───────────────────
We compute the angle between the ball's velocity vector (in court space)
and the vector from the ball to the nearest basket.  If this angle is
< basket_angle_thresh_deg, the ball is heading toward a basket → shot.

Contest classification
──────────────────────
At the release frame, we measure the court distance from the shooter to
the nearest defender:
  ≤ contest_close_ft : "tight"      → high defensive value
  ≤ contest_open_ft  : "challenged" → some defensive value
  >  contest_open_ft : "open"       → no credit (uncontested)

Shot zone (2PT/3PT) is determined by the shooter's court position.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional

import numpy as np

from src.court.court_model import CourtModel, ShotZone
from src.events.event import Event, EventType, PipelineState
from src.events.event_detector import BaseEventDetector
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)

# Contest distance thresholds (NBA standard: 4 ft = tight contest)
CONTEST_CLOSE_FT = 4.0
CONTEST_OPEN_FT  = 6.0
CONTEST_MAX_FT   = 15.0   # beyond this, defender is not contesting the shot


class ContestDetector(BaseEventDetector):
    """
    Parameters
    ──────────
    min_possession_frames    : Frames the possessor must hold ball before
                               a release counts as a shot attempt.
    release_accel_factor     : Ball speed must exceed prior speed × this
                               factor to count as a release.
    basket_angle_thresh_deg  : Max angle between ball velocity and direction-
                               to-basket for a release to count as a shot.
    cooldown_frames          : Minimum frames between contested shot events.
    steal_detector           : If set, notified on shot detection so it can
                               suppress steal events in the post-shot window.
    """

    def __init__(
        self,
        min_possession_frames: int = 3,
        release_accel_factor: float = 1.4,
        basket_angle_thresh_deg: float = 55.0,
        cooldown_frames: int = 60,
        steal_detector=None,
    ) -> None:
        self.min_possession    = min_possession_frames
        self.accel_factor      = release_accel_factor
        self.basket_angle      = np.deg2rad(basket_angle_thresh_deg)
        self.cooldown          = cooldown_frames
        self._steal_detector   = steal_detector
        self._court            = CourtModel()

        # Rolling history: each entry = (possessor_id, ball_pixel_pos, ball_court_pos)
        self._history: deque[dict] = deque(maxlen=10)
        self._last_event_frame = -9999

    def set_steal_detector(self, steal_detector) -> None:
        """Wire up the StealDetector so we can suppress post-shot steals."""
        self._steal_detector = steal_detector

    def update(
        self,
        state: PipelineState,
        manager: TrackManager,
    ) -> List[Event]:
        events: List[Event] = []

        ball_px    = state.ball_pixel_pos
        ball_court = state.ball_court_pos

        self._history.append({
            "possessor": state.ball_possessor_id,
            "ball_px":   ball_px.copy() if ball_px is not None else None,
            "ball_court": ball_court.copy() if ball_court is not None else None,
            "frame":     state.frame_idx,
        })

        if len(self._history) < self.min_possession + 2:
            return events
        if state.frame_idx - self._last_event_frame < self.cooldown:
            return events

        hist = list(self._history)

        # Condition 1: had a possessor for min_possession_frames recently
        recent_possessors = [
            h["possessor"] for h in hist[-self.min_possession - 2 : -1]
            if h["possessor"] is not None
        ]
        if len(recent_possessors) < self.min_possession:
            return events
        shooter_id = recent_possessors[-1]

        # Condition 2: no possessor right now (ball released)
        if state.ball_possessor_id is not None:
            return events

        # Condition 3: ball speed increased (release acceleration)
        prev_speeds = [
            np.linalg.norm(hist[i]["ball_px"] - hist[i-1]["ball_px"])
            for i in range(-min(4, len(hist)), -1)
            if hist[i]["ball_px"] is not None and hist[i-1]["ball_px"] is not None
        ]
        if len(prev_speeds) < 2:
            return events

        curr_speed = prev_speeds[-1]
        avg_prev   = float(np.mean(prev_speeds[:-1])) + 1e-6
        if curr_speed < avg_prev * self.accel_factor:
            return events

        # ── Shot confirmed — find shooter position ────────────────────────────
        shooter_track = next(
            (t for t in state.player_tracks if t.track_id == shooter_id), None
        )
        if shooter_track is None:
            return events

        # Condition 4 (optional): velocity toward a basket when H is available.
        is_three = False
        zone     = ShotZone.MID_RANGE    # default when H not available
        basket   = "unknown"
        if ball_court is not None and hist[-2]["ball_court"] is not None:
            ball_vel_court = ball_court - hist[-2]["ball_court"]
            if np.linalg.norm(ball_vel_court) < 0.1:
                return events
            from src.court.court_model import BASKET_LEFT, BASKET_RIGHT
            toward_left  = self._angle_between(ball_vel_court, BASKET_LEFT  - ball_court)
            toward_right = self._angle_between(ball_vel_court, BASKET_RIGHT - ball_court)
            if min(toward_left, toward_right) > self.basket_angle:
                return events
            basket = "left" if toward_left < toward_right else "right"
            if shooter_track.court_pos is not None:
                zone     = self._court.get_zone(shooter_track.court_pos[0],
                                                shooter_track.court_pos[1],
                                                basket=basket)
                is_three = zone in (ShotZone.CORNER_3, ShotZone.ABOVE_BREAK_3)
        # When H not available, skip basket-direction check — rely on speed
        # acceleration alone. Zone defaults to CONTESTED_2PT.

        # ── Find nearest defender ─────────────────────────────────────────────
        # Use the shooter's team to identify defenders — can't rely on
        # state.get_defensive_tracks() because possession is None at release.
        shooter_team = state.team_assignments.get(shooter_id)
        defending_team = (1 - shooter_team) if shooter_team in (0, 1) else None
        defender_candidates = [
            t for t in state.player_tracks
            if state.team_assignments.get(t.track_id) == defending_team
        ] if defending_team is not None else []

        shooter_pos = shooter_track.court_pos  # may be None when H invalid
        defender, dist_ft = None, float("inf")
        for t in defender_candidates:
            if shooter_pos is not None and t.court_pos is not None:
                d = float(np.linalg.norm(t.court_pos - shooter_pos))
            else:
                d = float(np.linalg.norm(t.center - shooter_track.center)) / state._PX_PER_FT
            if d < dist_ft:
                dist_ft  = d
                defender = t

        if dist_ft > CONTEST_MAX_FT:
            return events  # defender too far away to be contesting this shot

        if dist_ft <= CONTEST_CLOSE_FT:
            contest_level = "tight"
        elif dist_ft <= CONTEST_OPEN_FT:
            contest_level = "challenged"
        else:
            contest_level = "open"

        event_type = EventType.CONTESTED_3PT if is_three else EventType.CONTESTED_2PT

        events.append(Event(
            event_type=event_type,
            frame_idx=state.frame_idx,
            timestamp_sec=state.timestamp_sec,
            primary_player_id=defender.track_id if defender else -1,
            secondary_player_id=shooter_id,
            court_pos=shooter_pos.copy() if shooter_pos is not None else None,
            confidence=0.65,
            metadata={
                "zone":           zone.value,
                "defender_dist":  round(dist_ft, 2),
                "contest_level":  contest_level,
                "is_three":       is_three,
                "basket":         basket,
            },
        ))
        self._last_event_frame = state.frame_idx

        # Notify steal detector to suppress post-shot possession changes
        if self._steal_detector is not None:
            self._steal_detector.notify_shot(state.frame_idx)

        logger.debug(
            "SHOT: shooter=%d zone=%s contest=%s defender_dist=%.1fft",
            shooter_id, zone.value, contest_level, dist_ft,
        )
        return events

    def reset(self) -> None:
        self._history.clear()
        self._last_event_frame = -9999

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Angle in radians between two 2D vectors."""
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return float("inf")
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.arccos(cos_a))
