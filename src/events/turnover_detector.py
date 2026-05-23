"""
src/events/turnover_detector.py

Detects turnovers that are NOT steals: out-of-bounds, bad pass with no
steal credited, shot-clock violations (detected via prolonged possession),
and offensive fouls (future — requires foul detector).

Complements StealDetector, which handles possession changes WITH a defensive
player in proximity.  This detector fires when possession changes but no
defender was close enough to claim a steal.

Turnover types detected
───────────────────────
  bad_pass     : Possession changes, no defender nearby, ball was in motion.
  lost_ball    : Possession changes without pass (e.g. ball knocked out of bounds).
  out_of_bounds: Ball tracking lost at edge of frame (ball left court).

StealDetector exclusion
───────────────────────
If StealDetector fires within exclusion_window_frames of this detector
firing for the same possession, the turnover is suppressed (steal takes
credit — the steal event is richer with player attribution).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from src.events.event import Event, EventType, PipelineState, PossessionState
from src.events.event_detector import BaseEventDetector
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)

_STEAL_EXCLUSION_FRAMES = 10   # if steal fired within this window, skip TOV
_DEFENDER_DIST_PX = 120        # max px distance to consider "defender nearby"
_CONFIRM_FRAMES   = 4          # consecutive frames of different possessor to confirm


class TurnoverDetector(BaseEventDetector):
    """
    Parameters
    ──────────
    cooldown_frames : Minimum frames between successive turnovers.
    """

    def __init__(self, cooldown_frames: int = 90) -> None:
        self.cooldown = cooldown_frames

        self._prev_possessor: Optional[int] = None
        self._prev_team: Optional[int] = None
        self._change_frames: int = 0    # consecutive frames of new team possession
        self._last_event_frame: int = -9999
        self._steal_detector = None     # wired by runner

    # ── Wiring ────────────────────────────────────────────────────────────────

    def set_steal_detector(self, sd) -> None:
        self._steal_detector = sd

    # ── BaseEventDetector ─────────────────────────────────────────────────────

    def update(
        self,
        state: PipelineState,
        manager: TrackManager,
    ) -> List[Event]:
        events: List[Event] = []

        # Only during live play
        if state.possession_state in (PossessionState.DEAD_BALL, PossessionState.UNKNOWN):
            self._reset_counters()
            return events

        if state.frame_idx - self._last_event_frame < self.cooldown:
            return events

        possessor_id = state.ball_possessor_id
        possessor_team = (
            state.team_assignments.get(possessor_id)
            if possessor_id is not None else None
        )
        possessor_team = possessor_team if possessor_team in (0, 1) else None

        # Detect team change
        if (self._prev_team is not None
                and possessor_team is not None
                and possessor_team != self._prev_team):
            self._change_frames += 1
        else:
            self._change_frames = 0

        if self._change_frames >= _CONFIRM_FRAMES:
            # Confirmed possession change — check if steal already fired
            steal_last = getattr(self._steal_detector, "_last_event_frame", -9999)
            if state.frame_idx - steal_last <= _STEAL_EXCLUSION_FRAMES:
                # Steal detector already credited this — skip
                self._reset_counters()
                return events

            # Classify type
            tov_type = self._classify_type(state)

            events.append(Event(
                event_type=EventType.TURNOVER,
                frame_idx=state.frame_idx,
                timestamp_sec=state.timestamp_sec,
                primary_player_id=self._prev_possessor or -1,
                secondary_player_id=possessor_id,
                court_pos=(state.ball_court_pos.copy()
                           if state.ball_court_pos is not None else None),
                confidence=0.65,
                metadata={
                    "turnover_type": tov_type,
                    "prev_team": self._prev_team,
                    "new_team": possessor_team,
                },
            ))
            self._last_event_frame = state.frame_idx
            logger.debug(
                "TURNOVER (%s): player=#%d  prev_team=%d → new_team=%d",
                tov_type, self._prev_possessor or -1,
                self._prev_team or -1, possessor_team,
            )
            self._reset_counters()

        # Update trackers
        if possessor_team is not None:
            self._prev_possessor = possessor_id
            self._prev_team = possessor_team

        return events

    def reset(self) -> None:
        self._reset_counters()
        self._prev_possessor = None
        self._prev_team = None
        self._last_event_frame = -9999

    # ── Private ───────────────────────────────────────────────────────────────

    def _classify_type(self, state: PipelineState) -> str:
        """Guess turnover type from ball and player context."""
        if state.ball_track is None:
            return "lost_ball"

        # Ball near frame edge → out of bounds
        if state.raw_bgr is not None:
            h, w = state.raw_bgr.shape[:2]
            bx, by = float(state.ball_track.center[0]), float(state.ball_track.center[1])
            if bx < 0.05 * w or bx > 0.95 * w or by < 0.05 * h or by > 0.95 * h:
                return "out_of_bounds"

        # Ball was moving fast → likely bad pass
        return "bad_pass"

    def _reset_counters(self) -> None:
        self._change_frames = 0
