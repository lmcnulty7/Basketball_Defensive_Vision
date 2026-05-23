"""
src/stats/speed_calculator.py

Computes defensive speed in mph from smoothed court-coordinate trajectories.

Two speed metrics
─────────────────
1. avg_speed_mph      : Average speed over the entire tracked period.
2. defensive_speed_mph: Average speed during "while guarding" windows only
                        (frames where the player has an active matchup
                        within guard_dist_ft).  This is the most meaningful
                        defensive metric — it measures how fast a player
                        moves when actively defending, filtering out
                        standing-around time.

Unit conversion
───────────────
  court coordinates → feet
  frame stride / fps → seconds per frame
  feet / second × 0.6818 → mph
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)

FPS_DEFAULT = 30.0
FT_PER_SEC_TO_MPH = 0.681818


class SpeedCalculator:
    """
    Parameters
    ──────────
    stride           : Frame stride used in pipeline (e.g. 3 = every 3rd frame).
    fps              : Source video fps.
    smoothing_frames : Apply a rolling average over this many frames before
                       computing velocity to reduce detection jitter.
    guard_dist_ft    : Distance threshold defining "actively guarding."
                       Speed is counted as defensive_speed only when a
                       matchup exists and the defender is within this distance.
    """

    def __init__(
        self,
        stride: int    = 3,
        fps: float     = FPS_DEFAULT,
        smoothing_frames: int = 5,
        guard_dist_ft: float  = 8.0,
    ) -> None:
        self.stride    = stride
        self.fps       = fps
        self.dt        = stride / fps           # seconds per processed frame
        self.smoothing = smoothing_frames
        self.guard_dist = guard_dist_ft

        # track_id → list of per-frame speeds (ft/sec)
        self._speeds_all:  Dict[int, List[float]] = defaultdict(list)
        self._speeds_guard: Dict[int, List[float]] = defaultdict(list)

    def update(
        self,
        manager: TrackManager,
        current_matchups: Dict[int, Optional[int]],
        defensive_positions: Dict[int, np.ndarray],
        offensive_positions: Dict[int, np.ndarray],
    ) -> None:
        """
        Compute per-defender speed for the current frame.

        Parameters
        ──────────
        manager             : TrackManager (for position history).
        current_matchups    : {defender_id: offensive_id} from MatchupTracker.
        defensive_positions : {track_id: [x_ft, y_ft]} for defenders.
        offensive_positions : {track_id: [x_ft, y_ft]} for offensive players.
        """
        for did, dpos in defensive_positions.items():
            vel = manager.get_velocity(did, n_frames=self.smoothing)
            if vel is None:
                continue

            # Velocity is in px/frame from TrackManager; but we need ft/frame.
            # Court positions from homography are already in feet, so we can
            # compute velocity directly from court-coordinate history.
            court_hist = manager.get_history(did, n=self.smoothing)
            if len(court_hist) < 2:
                continue

            # Use court_pos if available; otherwise skip this track
            valid = [s for s in court_hist if s.court_pos is not None]
            if len(valid) < 2:
                continue

            delta_pos = valid[-1].court_pos - valid[0].court_pos   # feet
            delta_t   = max(1, valid[-1].frame_idx - valid[0].frame_idx) * self.dt
            speed_ft_s = float(np.linalg.norm(delta_pos) / delta_t)

            self._speeds_all[did].append(speed_ft_s)

            # Defensive speed: only count if actively guarding
            oid = current_matchups.get(did)
            if oid is not None and oid in offensive_positions:
                dist = float(np.linalg.norm(dpos - offensive_positions[oid]))
                if dist <= self.guard_dist:
                    self._speeds_guard[did].append(speed_ft_s)

    # ── Queries ───────────────────────────────────────────────────────────────

    def avg_speed_mph(self, track_id: int) -> float:
        speeds = self._speeds_all[track_id]
        if not speeds:
            return 0.0
        return float(np.mean(speeds)) * FT_PER_SEC_TO_MPH

    def avg_defensive_speed_mph(self, track_id: int) -> float:
        """Average speed while actively guarding an opponent."""
        speeds = self._speeds_guard[track_id]
        if not speeds:
            return 0.0
        return float(np.mean(speeds)) * FT_PER_SEC_TO_MPH

    def max_speed_mph(self, track_id: int) -> float:
        speeds = self._speeds_all[track_id]
        if not speeds:
            return 0.0
        return float(np.max(speeds)) * FT_PER_SEC_TO_MPH

    def all_speeds(self) -> Dict[int, Dict[str, float]]:
        """Return {track_id: {avg_mph, defensive_mph, max_mph}} for all defenders."""
        all_ids = set(self._speeds_all.keys()) | set(self._speeds_guard.keys())
        return {
            tid: {
                "avg_mph":       round(self.avg_speed_mph(tid), 2),
                "defensive_mph": round(self.avg_defensive_speed_mph(tid), 2),
                "max_mph":       round(self.max_speed_mph(tid), 2),
            }
            for tid in all_ids
        }

    def reset(self) -> None:
        self._speeds_all.clear()
        self._speeds_guard.clear()
