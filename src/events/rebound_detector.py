"""
src/events/rebound_detector.py

Detects offensive and defensive rebounds.

Rebound lifecycle
─────────────────
1. A shot was recently detected (ContestDetector fired a CONTESTED_* event,
   or we detect the ball moving toward basket independently).
2. Ball enters the "basket area" (within `basket_radius_ft` of either basket
   in court coordinates).
3. Ball's trajectory reverses (it was falling → now possessed or still
   moving away from basket) — confirming it hit the rim/backboard.
4. A player gains possession of the ball (ball_possessor_id becomes non-None).
5. If that player is on the defensive team → DEFENSIVE REBOUND.
   If on the offensive team → OFFENSIVE REBOUND.

Basket area detection
─────────────────────
We check if the ball's court position is within `basket_radius_ft` of
BASKET_LEFT or BASKET_RIGHT.  Both baskets are monitored; which one is
relevant depends on which team was shooting (the `_last_shooting_team`).

Fallback: shot-arc detection
─────────────────────────────
Even without a ContestDetector notification, we detect shots via ball
pixel-y trajectory: ball descending (pixel_y INCREASING toward lower frame)
after being high (pixel_y was below a threshold → ball is high in frame).
This makes ReboundDetector self-contained.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional

import numpy as np

from src.classification.team_classifier import TEAM_HOME, TEAM_AWAY
from src.court.court_model import BASKET_LEFT, BASKET_RIGHT
from src.events.event import Event, EventType, PipelineState
from src.events.event_detector import BaseEventDetector
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)

BASKET_RADIUS_FT   = 8.0   # ft — ball within this of basket = near-basket zone
MIN_SHOT_PEAK_PX   = 0.35  # ball pixel_y must be above this fraction of frame
                           # (top 35%) to count as a "high ball" → shot arc


class ReboundDetector(BaseEventDetector):
    """
    Parameters
    ──────────
    basket_radius_ft  : Court-distance threshold defining the basket area.
    cooldown_frames   : Minimum frames between rebound events.
    shot_window       : Frames after a shot to watch for a rebound.
    """

    def __init__(
        self,
        basket_radius_ft: float = BASKET_RADIUS_FT,
        cooldown_frames: int    = 45,
        shot_window: int        = 60,
    ) -> None:
        self.basket_radius    = basket_radius_ft
        self.cooldown         = cooldown_frames
        self.shot_window      = shot_window

        self._ball_px: deque[np.ndarray] = deque(maxlen=10)
        self._shot_frame: Optional[int]  = None   # frame when last shot detected
        self._shooting_team: Optional[int] = None
        self._near_basket    = False               # ball was in basket area
        self._last_event_frame = -9999
        self._prev_possessor: Optional[int] = None

    # ── External notification ─────────────────────────────────────────────────

    def notify_shot(self, frame_idx: int, shooting_team: Optional[int] = None) -> None:
        """Called by ContestDetector when a shot is detected."""
        self._shot_frame    = frame_idx
        self._shooting_team = shooting_team
        self._near_basket   = False
        logger.debug("ReboundDetector: shot notification at frame %d", frame_idx)

    # ── BaseEventDetector ─────────────────────────────────────────────────────

    def update(
        self,
        state: PipelineState,
        manager: TrackManager,
    ) -> List[Event]:
        events: List[Event] = []

        if state.ball_track is None:
            self._prev_possessor = state.ball_possessor_id
            return events

        ball_px    = state.ball_track.center
        ball_court = state.ball_court_pos

        self._ball_px.append(ball_px.copy())

        # ── Fallback shot detection: ball was high, now descending ────────────
        # Works with or without homography — uses pixel height when H absent.
        if self._shot_frame is None:
            if self._ball_is_high(ball_px, state) and self._ball_descending():
                self._shot_frame    = state.frame_idx
                self._shooting_team = state.offensive_team_id
                self._near_basket   = False
                logger.debug(
                    "ReboundDetector: inferred shot at frame %d", state.frame_idx
                )

        # ── Check for rebound ─────────────────────────────────────────────────
        if (self._shot_frame is not None and
                state.frame_idx - self._shot_frame <= self.shot_window):

            # Track whether ball enters basket area
            if ball_court is not None:
                dist_l = float(np.linalg.norm(ball_court - BASKET_LEFT))
                dist_r = float(np.linalg.norm(ball_court - BASKET_RIGHT))
                if min(dist_l, dist_r) <= self.basket_radius:
                    self._near_basket = True

            # Rebound: ball was near basket, now a player has it
            if (self._near_basket and
                    state.ball_possessor_id is not None and
                    self._prev_possessor is None and
                    state.frame_idx - self._last_event_frame >= self.cooldown):

                possessor_team = state.team_assignments.get(state.ball_possessor_id)
                shooting_team  = (self._shooting_team
                                   if self._shooting_team is not None
                                   else state.defensive_team_id)

                if possessor_team == shooting_team:
                    event_type = EventType.REBOUND_OFF
                else:
                    event_type = EventType.REBOUND_DEF

                events.append(Event(
                    event_type=event_type,
                    frame_idx=state.frame_idx,
                    timestamp_sec=state.timestamp_sec,
                    primary_player_id=state.ball_possessor_id,
                    court_pos=(ball_court.copy()
                               if ball_court is not None else None),
                    confidence=0.80,
                    metadata={
                        "rebounding_team": possessor_team,
                        "shooting_team":   shooting_team,
                    },
                ))
                self._last_event_frame = state.frame_idx
                self._shot_frame       = None
                self._near_basket      = False
                logger.debug(
                    "%s: player=%d",
                    event_type.value, state.ball_possessor_id,
                )

        self._prev_possessor = state.ball_possessor_id
        return events

    def reset(self) -> None:
        self._ball_px.clear()
        self._shot_frame       = None
        self._shooting_team    = None
        self._near_basket      = False
        self._last_event_frame = -9999
        self._prev_possessor   = None

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ball_is_high(ball_px: np.ndarray, state: PipelineState) -> bool:
        """True if ball pixel_y is in the upper portion of the frame."""
        if state.raw_bgr is None:
            return False
        frame_h = state.raw_bgr.shape[0]
        return float(ball_px[1]) < frame_h * MIN_SHOT_PEAK_PX

    def _ball_descending(self) -> bool:
        """True if ball pixel_y has been increasing (ball falling) for 2+ frames."""
        if len(self._ball_px) < 3:
            return False
        px = list(self._ball_px)
        return px[-1][1] > px[-2][1] > px[-3][1]
