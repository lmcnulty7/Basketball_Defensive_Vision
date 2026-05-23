"""
src/tracking/possession_tracker.py

Tracks game possession state and determines which basket each team attacks.

Basket assignment
─────────────────
In the first ~15 possessions the tracker observes which x-direction the ball
moves when each team has it.  Home team consistently moving the ball toward
positive-x → home team attacks the right basket.  This is cached and held
until halftime is detected (a sustained reversal in basket assignment).

Halftime detection
──────────────────
If the inferred attacking basket for the home team reverses for 20+
consecutive possession-bearing frames, the tracker declares halftime and
flips all assignments.

Possession state
────────────────
The state machine emits one of five PossessionState values:
  UNKNOWN       — cold start, not enough data
  HOME_OFFENSE  — home team has ball, in half-court set
  AWAY_OFFENSE  — away team has ball, in half-court set
  TRANSITION    — possession just changed, basket direction unclear
  DEAD_BALL     — ball lost for too long (timeout/out-of-bounds/replay)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from src.classification.team_classifier import TEAM_HOME, TEAM_AWAY
from src.court.court_model import BASKET_LEFT, BASKET_RIGHT
from src.events.event import PossessionState
from src.tracking.multi_tracker import Track

logger = logging.getLogger(__name__)

# Minimum ball x-coordinate samples per team before assigning baskets
_MIN_SAMPLES = 15
# Consecutive frames of reversed basket direction before halftime is declared
_HALFTIME_FLIP_FRAMES = 20
# Frames without ball before declaring DEAD_BALL
_DEAD_BALL_TIMEOUT = 45


class PossessionTracker:
    """
    Infers possession state and attacking basket from per-frame data.

    Usage
    ─────
        tracker = PossessionTracker()
        for frame in video:
            state, basket = tracker.update(
                possessor_id, team_assignments, ball_court_pos, frame_idx
            )
    """

    def __init__(self) -> None:
        # x-coordinate samples per team (in court feet) during their possessions
        self._home_x_samples: List[float] = []
        self._away_x_samples: List[float] = []

        # Assigned attacking baskets once enough samples collected
        # "left" → team attacks BASKET_LEFT (-41.75, 0)
        # "right" → team attacks BASKET_RIGHT (+41.75, 0)
        self._home_basket: Optional[str] = None
        self._away_basket: Optional[str] = None

        self._state = PossessionState.UNKNOWN
        self._last_possessor_team: Optional[int] = None
        self._frames_since_ball: int = 0
        self._transition_frames: int = 0

        # Halftime detection
        self._reversed_frames: int = 0
        self._halftime_occurred: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        possessor_id: Optional[int],
        team_assignments: Dict[int, int],
        ball_court_pos: Optional[np.ndarray],
        frame_idx: int,
    ) -> tuple[PossessionState, Optional[str]]:
        """
        Update possession state for this frame.

        Parameters
        ──────────
        possessor_id    : track_id of player with ball, or None.
        team_assignments: {track_id → TEAM_HOME/AWAY/REFEREE}
        ball_court_pos  : Ball position in court feet [x, y], or None if H invalid.
        frame_idx       : Current frame index.

        Returns
        ───────
        (PossessionState, attacking_basket)
        attacking_basket is "left", "right", or None if not yet determined.
        """
        possessor_team = (
            team_assignments.get(possessor_id)
            if possessor_id is not None else None
        )
        possessor_team = (
            possessor_team
            if possessor_team in (TEAM_HOME, TEAM_AWAY) else None
        )

        # Track samples for basket assignment
        if possessor_team is not None and ball_court_pos is not None:
            x = float(ball_court_pos[0])
            if not np.isnan(x):
                if possessor_team == TEAM_HOME:
                    self._home_x_samples.append(x)
                else:
                    self._away_x_samples.append(x)

        # Try to assign baskets once enough samples accumulated
        if self._home_basket is None:
            self._try_assign_baskets()

        # Halftime detection (only once baskets are assigned)
        if self._home_basket is not None and possessor_team is not None and ball_court_pos is not None:
            self._check_halftime(possessor_team, float(ball_court_pos[0]))

        # Update dead-ball counter
        if possessor_id is None:
            self._frames_since_ball += 1
        else:
            self._frames_since_ball = 0

        # State machine
        self._state = self._compute_state(possessor_team)
        self._last_possessor_team = possessor_team

        attacking = self._attacking_basket(possessor_team)
        return self._state, attacking

    @property
    def baskets_assigned(self) -> bool:
        return self._home_basket is not None

    def reset_for_camera_cut(self) -> None:
        """Soft reset on camera cut — keep basket assignments, clear transient state."""
        self._frames_since_ball = 0
        self._transition_frames = 0
        self._state = PossessionState.UNKNOWN

    # ── Private ───────────────────────────────────────────────────────────────

    def _try_assign_baskets(self) -> None:
        if (len(self._home_x_samples) < _MIN_SAMPLES or
                len(self._away_x_samples) < _MIN_SAMPLES):
            return

        home_mean_x = float(np.mean(self._home_x_samples[-_MIN_SAMPLES:]))
        away_mean_x = float(np.mean(self._away_x_samples[-_MIN_SAMPLES:]))

        if home_mean_x > 0:
            self._home_basket = "right"
            self._away_basket = "left"
        else:
            self._home_basket = "left"
            self._away_basket = "right"

        logger.info(
            "Basket assignment: HOME→%s  AWAY→%s  (home_mean_x=%.1f)",
            self._home_basket, self._away_basket, home_mean_x,
        )

    def _check_halftime(self, possessor_team: int, ball_x: float) -> None:
        """Detect sustained basket-direction reversal → halftime."""
        if self._halftime_occurred:
            return

        expected_home_side = "right" if self._home_basket == "right" else "left"
        actual_side = "right" if ball_x > 0 else "left"

        if possessor_team == TEAM_HOME:
            reversed_ = (actual_side != expected_home_side)
        else:
            reversed_ = (actual_side == expected_home_side)

        if reversed_:
            self._reversed_frames += 1
        else:
            self._reversed_frames = max(0, self._reversed_frames - 2)

        if self._reversed_frames >= _HALFTIME_FLIP_FRAMES:
            self._home_basket, self._away_basket = self._away_basket, self._home_basket
            self._halftime_occurred = True
            self._reversed_frames = 0
            self._home_x_samples.clear()
            self._away_x_samples.clear()
            logger.info("Halftime detected — basket assignments flipped")

    def _compute_state(self, possessor_team: Optional[int]) -> PossessionState:
        if self._frames_since_ball > _DEAD_BALL_TIMEOUT:
            return PossessionState.DEAD_BALL

        if possessor_team is None:
            return (PossessionState.TRANSITION
                    if self._last_possessor_team is not None
                    else self._state)

        if possessor_team != self._last_possessor_team and self._last_possessor_team is not None:
            # Possession just changed → brief transition
            self._transition_frames = 10
            return PossessionState.TRANSITION

        if self._transition_frames > 0:
            self._transition_frames -= 1
            return PossessionState.TRANSITION

        if possessor_team == TEAM_HOME:
            return PossessionState.HOME_OFFENSE
        return PossessionState.AWAY_OFFENSE

    def _attacking_basket(self, possessor_team: Optional[int]) -> Optional[str]:
        if self._home_basket is None or possessor_team is None:
            return None
        return (self._home_basket if possessor_team == TEAM_HOME
                else self._away_basket)
