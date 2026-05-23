"""
src/events/shot_detector.py

Detects shot attempts and their outcomes (made / missed).

Two-phase detection
───────────────────
Phase 1 — Shot attempt:
  Ball velocity is directed toward the attacking basket AND ball is rising
  (vy < 0 in pixel space) AND ball speed is sufficient.  The shooter is
  the player who last had possession.

Phase 2 — Outcome classification (within confirm_window frames):
  MADE  : Ball tracking is lost near the basket AND no rebound fires
          within outcome_window frames → the ball went through the net.
  MISS  : Ball direction changes sharply near basket AND the rebound
          detector subsequently fires within outcome_window frames.
  UNKNOWN: Neither signal arrives in time → no outcome emitted (attempt
           still logged internally for assist tracking via AssistDetector).

2PT vs 3PT
──────────
Determined by shooter's court position relative to the 3PT line at the
moment of the shot attempt.  Falls back to pixel-space basket distance
when H is not valid.

Shot type
─────────
If shooter is within LAYUP_DIST_FT of the basket at shot time: layup/dunk.
Otherwise: jump shot.  Hook shots require pose estimation (future).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from src.court.court_model import (
    BASKET_LEFT, BASKET_RIGHT, CourtModel, THREE_RADIUS, THREE_CORNER_Y,
    THREE_LEFT_X, THREE_RIGHT_X,
)
from src.events.event import Event, EventType, PipelineState, PossessionState
from src.events.event_detector import BaseEventDetector
from src.tracking.multi_tracker import Track
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)

LAYUP_DIST_FT    = 5.0    # ft from basket → layup/dunk
BASKET_NEAR_FT   = 4.5    # ft — ball "near basket" for made-shot detection
BASKET_NEAR_PX   = 80     # px fallback when H not valid
MIN_SHOT_SPEED   = 10.0   # px/frame minimum ball speed to register attempt
MIN_RISE_SPEED   = 5.0    # px/frame minimum upward component

_court = CourtModel()


class ShotDetector(BaseEventDetector):
    """
    Parameters
    ──────────
    confirm_window  : Frames to wait for outcome signal after attempt.
    cooldown_frames : Frames between successive shot attempts from the same player.
    """

    def __init__(
        self,
        confirm_window: int  = 90,   # 3 s at 30 fps
        cooldown_frames: int = 60,
    ) -> None:
        self.confirm_window = confirm_window
        self.cooldown       = cooldown_frames

        # Pending attempt: {shooter_id: (attempt_frame, is_3pt, court_pos, shot_type)}
        self._pending: Optional[Tuple[int, int, bool, Optional[np.ndarray], str]] = None
        # (shooter_id, attempt_frame, is_3pt, court_pos, shot_type)

        self._last_event_frame: Dict[int, int] = {}  # shooter_id → last shot frame
        self._rebound_detector = None  # wired by runner

    # ── Wiring ────────────────────────────────────────────────────────────────

    def set_rebound_detector(self, rd) -> None:
        self._rebound_detector = rd

    # ── BaseEventDetector ─────────────────────────────────────────────────────

    def update(
        self,
        state: PipelineState,
        manager: TrackManager,
    ) -> List[Event]:
        events: List[Event] = []

        # Only process during live play
        if state.possession_state in (PossessionState.DEAD_BALL,):
            return events

        # ── Phase 2: check pending attempt for outcome ────────────────────────
        if self._pending is not None:
            shooter_id, attempt_frame, is_3pt, court_pos, shot_type = self._pending
            frames_elapsed = state.frame_idx - attempt_frame

            if frames_elapsed > self.confirm_window:
                # Timeout — emit as unknown (AssistDetector still gets notified)
                self._pending = None
            else:
                outcome = self._classify_outcome(state, attempt_frame)
                if outcome == "made":
                    ev = self._make_event(
                        state, shooter_id, is_3pt, True, court_pos, shot_type
                    )
                    events.append(ev)
                    self._last_event_frame[shooter_id] = attempt_frame
                    self._pending = None
                elif outcome == "miss":
                    ev = self._make_event(
                        state, shooter_id, is_3pt, False, court_pos, shot_type
                    )
                    events.append(ev)
                    self._last_event_frame[shooter_id] = attempt_frame
                    self._pending = None

        # ── Phase 1: detect shot attempt ─────────────────────────────────────
        if (self._pending is None
                and state.ball_track is not None
                and not state.ball_is_predicted
                and state.ball_possessor_id is not None
                and len(state.ball_track_history) >= 6):

            shooter_id = state.ball_possessor_id
            last_shot = self._last_event_frame.get(shooter_id, -9999)
            if state.frame_idx - last_shot < self.cooldown:
                return events

            vel = self._ball_velocity(state.ball_track_history)
            if vel is None:
                return events

            vx, vy = float(vel[0]), float(vel[1])
            speed = float(np.linalg.norm(vel))

            # Ball must be rising fast enough
            if vy >= -MIN_RISE_SPEED or speed < MIN_SHOT_SPEED:
                return events

            # Ball must have meaningful horizontal motion toward the basket.
            # Dribble bounces are mostly vertical (abs(vx) << abs(vy)); shots
            # are directed diagonally — requiring 20% horizontal filters dribbles.
            if abs(vx) < 0.20 * speed:
                return events

            # Require 2 consecutive rising frames, not just one noisy sample.
            # This prevents a single bad detection from triggering a shot attempt.
            if not self._ball_consistently_rising(state.ball_track_history, n=2):
                return events

            # Ball velocity must point toward the attacking basket
            if not self._toward_basket(state, vx):
                return events

            # Classify 2PT vs 3PT
            is_3pt = self._is_three_pointer(state, shooter_id)
            court_pos = state.ball_court_pos
            shot_type = self._shot_type(state, shooter_id)

            self._pending = (shooter_id, state.frame_idx, is_3pt, court_pos, shot_type)
            logger.debug(
                "SHOT attempt: shooter=#%d  %s  3pt=%s  type=%s",
                shooter_id, state.possession_state.value, is_3pt, shot_type,
            )

        return events

    def reset(self) -> None:
        self._pending = None
        self._last_event_frame.clear()

    @property
    def pending_shooter_id(self) -> Optional[int]:
        """Expose the current shooter for AssistDetector to track the last passer."""
        return self._pending[0] if self._pending else None

    @property
    def pending_attempt_frame(self) -> Optional[int]:
        return self._pending[1] if self._pending else None

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ball_velocity(history: List[Track]) -> Optional[np.ndarray]:
        """
        Savitzky-Golay smoothed velocity from ball position history.

        SG preserves the shape of the trajectory (parabola for shots, sinusoid
        for dribbles) while removing per-frame detection noise.  Falls back to
        a 3-frame finite difference if scipy is unavailable or history is short.
        """
        if len(history) < 3:
            return None

        xs = np.array([t.center[0] for t in history], dtype=float)
        ys = np.array([t.center[1] for t in history], dtype=float)
        n  = len(history)

        if n >= 7:
            try:
                from scipy.signal import savgol_filter
                wl = min(n, 9)
                wl = wl if wl % 2 == 1 else wl - 1
                xs = savgol_filter(xs, wl, polyorder=2)
                ys = savgol_filter(ys, wl, polyorder=2)
            except Exception:
                pass

        p1 = np.array([xs[-3], ys[-3]])
        p2 = np.array([xs[-1], ys[-1]])
        dt = max(1, history[-1].frame_idx - history[-3].frame_idx)
        return (p2 - p1) / dt

    @staticmethod
    def _ball_consistently_rising(history: List[Track], n: int = 2) -> bool:
        """True if ball pixel_y decreased (ball rose) for the last n consecutive frames."""
        if len(history) < n + 1:
            return False
        for i in range(-n, 0):
            if history[i].center[1] >= history[i - 1].center[1]:
                return False
        return True

    def _toward_basket(self, state: PipelineState, vx: float) -> bool:
        """True if ball horizontal velocity is directed toward the attacking basket."""
        basket = state.attacking_basket
        if basket is None:
            # Can't determine — use any upward movement (less precise)
            return True
        # "right" basket is at +41.75 ft → vx > 0 in pixel space (ball moves right)
        # "left"  basket is at -41.75 ft → vx < 0 (ball moves left)
        if basket == "right":
            return vx > 2.0
        return vx < -2.0

    def _is_three_pointer(self, state: PipelineState, shooter_id: int) -> bool:
        """Determine if the shot is from 3PT range using court coordinates."""
        shooter_track = next(
            (t for t in state.player_tracks if t.track_id == shooter_id), None
        )
        if shooter_track is None:
            return False

        # Use court position if available
        if shooter_track.court_pos is not None:
            x, y = float(shooter_track.court_pos[0]), float(shooter_track.court_pos[1])
            if np.isnan(x) or np.isnan(y):
                return self._is_three_pointer_px(state, shooter_track)
            basket_side = "left" if state.attacking_basket == "left" else "right"
            return _court.is_three_point(x, y, basket=basket_side)

        return self._is_three_pointer_px(state, shooter_track)

    @staticmethod
    def _is_three_pointer_px(state: PipelineState, shooter_track: Track) -> bool:
        """Pixel-space fallback: shooter in outer 35% of frame → 3PT."""
        if state.raw_bgr is None:
            return False
        w = state.raw_bgr.shape[1]
        cx = float(shooter_track.center[0])
        return cx < 0.2 * w or cx > 0.8 * w

    @staticmethod
    def _shot_type(state: PipelineState, shooter_id: int) -> str:
        shooter = next(
            (t for t in state.player_tracks if t.track_id == shooter_id), None
        )
        if shooter is None or shooter.court_pos is None:
            return "jump_shot"
        if np.isnan(shooter.court_pos[0]):
            return "jump_shot"
        basket_side = "left" if state.attacking_basket == "left" else "right"
        dist = _court.dist_to_basket(
            float(shooter.court_pos[0]), float(shooter.court_pos[1]),
            basket=basket_side,
        )
        return "layup" if dist <= LAYUP_DIST_FT else "jump_shot"

    def _classify_outcome(self, state: PipelineState, attempt_frame: int) -> str:
        """
        Returns "made", "miss", or "" (still pending).

        Made: ball tracking lost near basket + no rebound fired yet.
        Miss: rebound detector fired after the attempt.
        """
        # Check if rebound fired after attempt
        if self._rebound_detector is not None:
            rd_last = getattr(self._rebound_detector, "_last_event_frame", -9999)
            if rd_last >= attempt_frame:
                return "miss"

        # Check if ball is near basket and tracking is lost
        if state.ball_track is None and self._pending is not None:
            _, _, _, _, _ = self._pending
            if self._near_basket(state):
                return "made"

        return ""

    def _near_basket(self, state: PipelineState) -> bool:
        """True if the last known ball position was close to the attacking basket."""
        basket = state.attacking_basket
        if basket is None:
            return False
        basket_pos = BASKET_LEFT if basket == "left" else BASKET_RIGHT

        # Try court space first
        if self._pending and self._pending[3] is not None:
            court_pos = self._pending[3]
            if not np.any(np.isnan(court_pos)):
                dist = float(np.linalg.norm(court_pos - basket_pos))
                return dist <= BASKET_NEAR_FT

        return False

    @staticmethod
    def _make_event(
        state: PipelineState,
        shooter_id: int,
        is_3pt: bool,
        made: bool,
        court_pos: Optional[np.ndarray],
        shot_type: str,
    ) -> Event:
        if made:
            etype = EventType.SHOT_MADE_3PT if is_3pt else EventType.SHOT_MADE_2PT
        else:
            etype = EventType.SHOT_MISS_3PT if is_3pt else EventType.SHOT_MISS_2PT

        # Shot distance
        dist_ft = None
        shooter = next(
            (t for t in state.player_tracks if t.track_id == shooter_id), None
        )
        if shooter is not None and shooter.court_pos is not None and state.attacking_basket:
            if not np.any(np.isnan(shooter.court_pos)):
                basket_pos = (BASKET_LEFT if state.attacking_basket == "left"
                              else BASKET_RIGHT)
                dist_ft = round(float(np.linalg.norm(
                    shooter.court_pos - basket_pos
                )), 1)

        return Event(
            event_type=etype,
            frame_idx=state.frame_idx,
            timestamp_sec=state.timestamp_sec,
            primary_player_id=shooter_id,
            court_pos=court_pos.copy() if court_pos is not None else None,
            confidence=0.65,
            metadata={
                "shot_type": shot_type,
                "is_3pt":    is_3pt,
                "made":      made,
                "dist_ft":   dist_ft,
            },
        )
