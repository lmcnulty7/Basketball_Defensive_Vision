"""
src/events/event.py

Core data structures for the event detection layer.

Event       — a discrete defensive action (steal, block, etc.)
EventType   — the enumerated set of supported events
PipelineState — the per-frame snapshot that every detector reads from
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from src.classification.team_classifier import TEAM_HOME, TEAM_AWAY, TEAM_REFEREE
from src.tracking.multi_tracker import Track


class PossessionState(str, Enum):
    UNKNOWN       = "unknown"        # not enough data yet
    HOME_OFFENSE  = "home_offense"   # home team attacking
    AWAY_OFFENSE  = "away_offense"   # away team attacking
    TRANSITION    = "transition"     # fast break — basket not yet clear
    DEAD_BALL     = "dead_ball"      # whistle / out of bounds / timeout


# ── Event types ───────────────────────────────────────────────────────────────

class EventType(str, Enum):
    # ── Offensive ─────────────────────────────────────────────────────────────
    SHOT_MADE_2PT  = "shot_made_2pt"
    SHOT_MADE_3PT  = "shot_made_3pt"
    SHOT_MISS_2PT  = "shot_miss_2pt"
    SHOT_MISS_3PT  = "shot_miss_3pt"
    FREE_THROW_MADE = "free_throw_made"    # TODO: requires scene classifier
    FREE_THROW_MISS = "free_throw_miss"    # TODO: requires scene classifier
    ASSIST         = "assist"
    TURNOVER       = "turnover"
    # ── Defensive ─────────────────────────────────────────────────────────────
    STEAL          = "steal"
    BLOCK          = "block"
    DEFLECTION     = "deflection"
    CONTESTED_2PT  = "contested_2pt"
    CONTESTED_3PT  = "contested_3pt"
    REBOUND_DEF    = "defensive_rebound"
    REBOUND_OFF    = "offensive_rebound"
    CHARGE         = "charge"
    # ── Foul (requires audio/contact detection — future) ──────────────────────
    FOUL_PERSONAL  = "foul_personal"       # TODO
    FOUL_SHOOTING  = "foul_shooting"       # TODO


# ── Event ─────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """
    A discrete defensive action detected in the video.

    Attributes
    ──────────
    event_type          : Category of defensive event.
    frame_idx           : Absolute frame index when the event occurred.
    timestamp_sec       : Frame time in seconds.
    primary_player_id   : track_id of the defender who earns the stat.
    secondary_player_id : track_id of the offensive player involved (if any).
    court_pos           : [x_ft, y_ft] where the event occurred.
    confidence          : Detector confidence in [0, 1].  Events below 0.4
                          are flagged for manual review.
    metadata            : Event-specific details (zone, dist_ft, etc.).
    """
    event_type: EventType
    frame_idx: int
    timestamp_sec: float
    primary_player_id: int
    secondary_player_id: Optional[int] = None
    court_pos: Optional[np.ndarray]   = None
    confidence: float                 = 1.0
    metadata: dict                    = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type":          self.event_type.value,
            "frame_idx":           self.frame_idx,
            "timestamp_sec":       round(self.timestamp_sec, 3),
            "primary_player_id":   self.primary_player_id,
            "secondary_player_id": self.secondary_player_id,
            "court_pos": (self.court_pos.tolist()
                          if self.court_pos is not None else None),
            "confidence":          round(self.confidence, 3),
            "metadata":            self.metadata,
        }


# ── PipelineState ─────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """
    The per-frame snapshot flowing through the event detection layer.

    Every detector reads exclusively from this object plus the TrackManager.
    Fields are populated by earlier pipeline stages:
      player_tracks  ← MultiTracker + homography + team_classifier
      ball_track     ← BallTracker + homography
      team_assignments ← TeamClassifier
      ball_possessor_id ← BallTracker

    Convenience properties (offensive_team_id, get_defensive_tracks, etc.)
    spare detectors from recomputing the same lookups every frame.
    """

    frame_idx: int
    timestamp_sec: float
    raw_bgr: np.ndarray

    player_tracks: List[Track]           # court_pos + team_id already filled
    ball_track: Optional[Track]          # court_pos filled; None = ball lost
    ball_is_predicted: bool              # True = Kalman extrapolation
    ball_possessor_id: Optional[int]     # track_id of current ball holder
    ball_track_history: List[Track]      # recent ball positions for trajectory analysis

    team_assignments: Dict[int, int]     # track_id → TEAM_HOME/AWAY/REFEREE

    # Possession context (filled by PossessionTracker in runner)
    possession_state: PossessionState = PossessionState.UNKNOWN
    attacking_basket: Optional[str]   = None   # "left" or "right"

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def ball_court_pos(self) -> Optional[np.ndarray]:
        """Court coordinates of the ball, or None if unknown."""
        if self.ball_track is not None and self.ball_track.court_pos is not None:
            return self.ball_track.court_pos
        return None

    @property
    def ball_pixel_pos(self) -> Optional[np.ndarray]:
        """Pixel coordinates (bbox center) of the ball, or None."""
        if self.ball_track is not None:
            return self.ball_track.center
        return None

    @property
    def offensive_team_id(self) -> Optional[int]:
        """Team ID of the player currently holding the ball."""
        if self.ball_possessor_id is None:
            return None
        team = self.team_assignments.get(self.ball_possessor_id)
        return team if team in (TEAM_HOME, TEAM_AWAY) else None

    @property
    def defensive_team_id(self) -> Optional[int]:
        """Team ID of the defending team (opposite of possessor's team)."""
        off = self.offensive_team_id
        if off is None:
            return None
        return TEAM_AWAY if off == TEAM_HOME else TEAM_HOME

    def get_team_tracks(self, team_id: int) -> List[Track]:
        """All tracks assigned to a specific team this frame."""
        return [t for t in self.player_tracks
                if self.team_assignments.get(t.track_id) == team_id]

    def get_offensive_tracks(self) -> List[Track]:
        off = self.offensive_team_id
        return self.get_team_tracks(off) if off is not None else []

    def get_defensive_tracks(self) -> List[Track]:
        def_ = self.defensive_team_id
        return self.get_team_tracks(def_) if def_ is not None else []

    def nearest_defender_to(
        self,
        court_pos: np.ndarray,
        exclude_track_id: Optional[int] = None,
    ) -> tuple[Optional[Track], float]:
        """
        Find the closest defensive player to a given court position.

        Returns (Track, distance_ft) or (None, inf) if no defenders have
        valid court positions.
        """
        best_track = None
        best_dist  = float("inf")
        for track in self.get_defensive_tracks():
            if track.track_id == exclude_track_id:
                continue
            if track.court_pos is None:
                continue
            dist = float(np.linalg.norm(track.court_pos - court_pos))
            if dist < best_dist:
                best_dist  = dist
                best_track = track
        return best_track, best_dist

    # ~17 px per foot at standard broadcast distance (50ft court width ≈ 854px)
    _PX_PER_FT: float = 17.0

    def nearest_defender_pixel_to(
        self,
        pixel_pos: np.ndarray,
        exclude_track_id: Optional[int] = None,
    ) -> tuple[Optional[Track], float]:
        """
        Closest defensive player in pixel space, result converted to feet.

        Used as a fallback when homography is not available.
        Returns (Track, approx_distance_ft) or (None, inf).
        """
        best_track = None
        best_dist  = float("inf")
        for track in self.get_defensive_tracks():
            if track.track_id == exclude_track_id:
                continue
            dist_px = float(np.linalg.norm(track.center - pixel_pos))
            if dist_px < best_dist:
                best_dist  = dist_px
                best_track = track
        return best_track, best_dist / self._PX_PER_FT
