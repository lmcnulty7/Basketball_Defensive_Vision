"""
src/events/assist_detector.py

Credits the last passer before a made shot with an assist.

Logic
─────
We maintain a rolling buffer of (frame_idx, player_id, team_id) entries
representing which player had possession each frame.  When the shot detector
signals a made basket (SHOT_MADE_2PT or SHOT_MADE_3PT), we scan backwards
through the buffer to find the last player from the same team who had the ball
BEFORE the shooter — that player gets the assist.

No-assist conditions (matching NBA rules):
  - Shooter held the ball for more than assist_max_hold_frames before shooting
    (equivalent to ~2 seconds — dribble-heavy play, no assist)
  - No other same-team possessor found within assist_window_frames
  - Last possessor was the same player as the shooter
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

from src.events.event import Event, EventType, PipelineState, PossessionState
from src.events.event_detector import BaseEventDetector
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)


class AssistDetector(BaseEventDetector):
    """
    Parameters
    ──────────
    assist_window_frames   : How far back to look for the last passer.
    assist_max_hold_frames : If shooter held ball longer than this before shot,
                             no assist (they created their own shot).
    cooldown_frames        : Minimum frames between assist events for same player.
    """

    def __init__(
        self,
        assist_window_frames: int   = 150,   # ~5 s at 30 fps
        assist_max_hold_frames: int = 60,    # ~2 s — beyond this = own creation
        cooldown_frames: int        = 45,
    ) -> None:
        self.window    = assist_window_frames
        self.max_hold  = assist_max_hold_frames
        self.cooldown  = cooldown_frames

        # buffer entries: (frame_idx, player_id, team_id)
        self._buffer: deque[Tuple[int, int, int]] = deque(maxlen=assist_window_frames)
        self._last_event_frame: Dict[int, int] = {}
        self._shot_detector = None   # wired by runner

    # ── Wiring ────────────────────────────────────────────────────────────────

    def set_shot_detector(self, sd) -> None:
        self._shot_detector = sd

    # ── BaseEventDetector ─────────────────────────────────────────────────────

    def update(
        self,
        state: PipelineState,
        manager: TrackManager,
    ) -> List[Event]:
        events: List[Event] = []

        # Record possession this frame
        if state.ball_possessor_id is not None:
            team = state.team_assignments.get(state.ball_possessor_id)
            if team in (0, 1):
                self._buffer.append(
                    (state.frame_idx, state.ball_possessor_id, team)
                )

        # Check if shot detector just emitted a made shot
        made_events = [
            e for e in getattr(state, "_frame_events", [])
            if e.event_type in (EventType.SHOT_MADE_2PT, EventType.SHOT_MADE_3PT)
        ]

        for shot_ev in made_events:
            assist = self._find_assister(shot_ev, state)
            if assist is not None:
                assister_id, attempt_frame = assist
                last = self._last_event_frame.get(assister_id, -9999)
                if state.frame_idx - last >= self.cooldown:
                    events.append(Event(
                        event_type=EventType.ASSIST,
                        frame_idx=state.frame_idx,
                        timestamp_sec=state.timestamp_sec,
                        primary_player_id=assister_id,
                        secondary_player_id=shot_ev.primary_player_id,
                        court_pos=shot_ev.court_pos,
                        confidence=0.70,
                        metadata={"shot_frame": attempt_frame},
                    ))
                    self._last_event_frame[assister_id] = state.frame_idx
                    logger.debug(
                        "ASSIST: player #%d → shooter #%d",
                        assister_id, shot_ev.primary_player_id,
                    )

        return events

    def reset(self) -> None:
        self._buffer.clear()
        self._last_event_frame.clear()

    # ── Private ───────────────────────────────────────────────────────────────

    def _find_assister(
        self,
        shot_ev: Event,
        state: PipelineState,
    ) -> Optional[Tuple[int, int]]:
        """
        Search buffer backwards from shot attempt for the last passer.

        Returns (assister_id, attempt_frame) or None.
        """
        shooter_id  = shot_ev.primary_player_id
        shooter_team = state.team_assignments.get(shooter_id)
        if shooter_team is None:
            return None

        # Get shot attempt frame from shot detector if available
        attempt_frame = (
            self._shot_detector.pending_attempt_frame
            if self._shot_detector else state.frame_idx
        ) or state.frame_idx

        buf = list(self._buffer)

        # Find continuous shooter hold leading up to shot
        hold_start = attempt_frame
        for frame_idx, pid, tid in reversed(buf):
            if frame_idx > attempt_frame:
                continue
            if pid == shooter_id:
                hold_start = frame_idx
            else:
                break

        hold_duration = attempt_frame - hold_start
        if hold_duration > self.max_hold:
            # Shooter held ball too long — own creation, no assist
            return None

        # Find last same-team passer before shooter's hold
        for frame_idx, pid, tid in reversed(buf):
            if frame_idx >= hold_start:
                continue
            if tid == shooter_team and pid != shooter_id:
                return pid, attempt_frame

        return None
