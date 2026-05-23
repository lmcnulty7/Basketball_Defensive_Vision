"""
src/events/steal_detector.py

Detects steals: ball possession changes from one team to the other
without a shot or out-of-bounds event in between.

Logic
─────
We maintain a rolling buffer of (possessor_id, team_id) tuples.  A steal
fires when the last `confirm_frames` entries are consistently one team and
the `confirm_frames` entries before that were consistently the other team —
and neither team is REFEREE.

Requiring `confirm_frames` consecutive entries on each side:
  - Suppresses false triggers from brief occlusions where the possessor
    flickers to None and back.
  - Typical NBA steal happens over 2-5 frames; confirm_frames=3 is tight
    enough to catch it while filtering possession-tracker noise.

The `shot_clearance_frames` window clears the steal candidate if a
ContestDetector shot event was recently emitted.  This prevents the ball
going from a shooter to a rebounder from being misclassified as a steal.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Tuple

from src.classification.team_classifier import TEAM_REFEREE
from src.events.event import Event, EventType, PipelineState
from src.events.event_detector import BaseEventDetector
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)


class StealDetector(BaseEventDetector):
    """
    Parameters
    ──────────
    confirm_frames         : Consecutive frames each team must hold possession
                             for the transition to count as a steal.
    cooldown_frames        : Minimum frames between successive steal events.
    shot_clearance_frames  : Possession changes within this many frames of a
                             shot event are not credited as steals.
    """

    def __init__(
        self,
        confirm_frames: int = 2,
        cooldown_frames: int = 150,
        shot_clearance_frames: int = 30,
    ) -> None:
        self.confirm_frames   = confirm_frames
        self.cooldown         = cooldown_frames
        self.shot_clearance   = shot_clearance_frames
        self._frames_with_possessor = 0
        self._frames_total = 0

        # Each entry: (possessor_track_id, team_id) or None
        self._buffer: deque[Optional[Tuple[int, int]]] = deque(
            maxlen=confirm_frames * 2 + 2
        )
        self._last_event_frame  = -9999
        self._last_shot_frame   = -9999  # updated by notify_shot()

    # ── External notification ─────────────────────────────────────────────────

    def notify_shot(self, frame_idx: int) -> None:
        """
        Called by ContestDetector when a shot is detected.
        Clears pending steal candidates in the shot_clearance_frames window.
        """
        self._last_shot_frame = frame_idx

    # ── BaseEventDetector ─────────────────────────────────────────────────────

    def update(
        self,
        state: PipelineState,
        manager: TrackManager,
    ) -> List[Event]:
        events: List[Event] = []

        # Append current possessor info to buffer
        self._frames_total += 1
        if state.ball_possessor_id is not None:
            team = state.team_assignments.get(state.ball_possessor_id)
            if team in (0, 1):  # valid team (not referee, not unknown)
                self._buffer.append((state.ball_possessor_id, team))
                self._frames_with_possessor += 1
            else:
                self._buffer.append(None)
        else:
            self._buffer.append(None)

        # Periodically log possession tracking coverage
        if self._frames_total > 0 and self._frames_total % 300 == 0:
            pct = 100.0 * self._frames_with_possessor / self._frames_total
            logger.debug(
                "StealDetector: possessor tracked %.1f%% of frames (%d/%d)",
                pct, self._frames_with_possessor, self._frames_total,
            )

        # Need at least 2 × confirm_frames entries to detect a transition
        if len(self._buffer) < self.confirm_frames * 2:
            return events

        # Cooldown and shot-clearance guards
        if state.frame_idx - self._last_event_frame < self.cooldown:
            return events
        if state.frame_idx - self._last_shot_frame < self.shot_clearance:
            return events

        buf = list(self._buffer)

        # Split into "before" and "after" windows
        before = [x for x in buf[:self.confirm_frames] if x is not None]
        after  = [x for x in buf[-self.confirm_frames:] if x is not None]

        if len(before) < self.confirm_frames or len(after) < self.confirm_frames:
            return events

        # Check both windows are internally consistent (same team each time)
        before_teams = {x[1] for x in before}
        after_teams  = {x[1] for x in after}

        if len(before_teams) != 1 or len(after_teams) != 1:
            return events

        team_before = before_teams.pop()
        team_after  = after_teams.pop()

        # A steal requires a team change
        if team_before == team_after:
            return events

        defender_id  = after[-1][0]
        offensive_id = before[-1][0]

        events.append(Event(
            event_type=EventType.STEAL,
            frame_idx=state.frame_idx,
            timestamp_sec=state.timestamp_sec,
            primary_player_id=defender_id,
            secondary_player_id=offensive_id,
            court_pos=(state.ball_court_pos.copy()
                       if state.ball_court_pos is not None else None),
            confidence=0.75,
            metadata={
                "prev_team": team_before,
                "new_team":  team_after,
            },
        ))
        self._last_event_frame = state.frame_idx
        logger.debug("STEAL: defender=%d took from player=%d", defender_id, offensive_id)
        return events

    def reset(self) -> None:
        self._buffer.clear()
        self._last_event_frame = -9999
        self._last_shot_frame  = -9999
        self._frames_with_possessor = 0
        self._frames_total = 0
