"""
src/stats/matchup_tracker.py

Tracks matchup time: for each defender, how many seconds did they spend
guarding each offensive player?

Assignment algorithm
────────────────────
Each frame, we solve the linear assignment problem (Hungarian algorithm via
scipy.optimize.linear_sum_assignment) to make optimal one-to-one pairings
between defenders and offensive players.  Cost = court distance in feet.

Players farther than `max_matchup_dist_ft` apart are not considered matched
(they're guarding different areas of the court, not each other).

Output
──────
matchup_seconds[defender_id][offensive_id] = total seconds on that matchup.
primary_matchup[defender_id] = the offensive player they guarded most.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

MAX_MATCHUP_DIST_FT = 10.0   # ft — beyond this = not a direct matchup
LARGE_COST          = 1e6    # sentinel for infeasible assignments


class MatchupTracker:
    """
    Parameters
    ──────────
    max_matchup_dist_ft : Court distance threshold for a valid matchup.
    stride              : Frame stride used in the pipeline.
    fps                 : Source video fps.  Together, stride / fps = seconds
                          per processed frame → used to convert frame counts
                          to seconds.
    """

    def __init__(
        self,
        max_matchup_dist_ft: float = MAX_MATCHUP_DIST_FT,
        stride: int = 3,
        fps: float  = 30.0,
    ) -> None:
        self.max_dist    = max_matchup_dist_ft
        self.seconds_per_frame = stride / fps

        # defender_id → {offensive_id: total_seconds}
        self._matchup_secs: Dict[int, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # defender_id → current offensive_id being guarded (for display)
        self._current: Dict[int, Optional[int]] = {}

    def update(
        self,
        offensive_positions: Dict[int, np.ndarray],
        defensive_positions: Dict[int, np.ndarray],
    ) -> Dict[int, Optional[int]]:
        """
        Run one frame of matchup assignment.

        Parameters
        ──────────
        offensive_positions : {track_id: [x_ft, y_ft]} for offensive players.
        defensive_positions : {track_id: [x_ft, y_ft]} for defensive players.

        Returns
        ───────
        {defender_id: offensive_id or None}  — this frame's assignments.
        """
        if not offensive_positions or not defensive_positions:
            self._current = {did: None for did in defensive_positions}
            return self._current

        off_ids = list(offensive_positions.keys())
        def_ids = list(defensive_positions.keys())
        n_off, n_def = len(off_ids), len(def_ids)

        # Build cost matrix (n_def × n_off)
        cost = np.full((n_def, n_off), LARGE_COST)
        for di, did in enumerate(def_ids):
            dp = defensive_positions[did]
            for oi, oid in enumerate(off_ids):
                op = offensive_positions[oid]
                dist = float(np.linalg.norm(dp - op))
                if dist <= self.max_dist:
                    cost[di, oi] = dist

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)

        assignments: Dict[int, Optional[int]] = {did: None for did in def_ids}
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] < LARGE_COST:
                did = def_ids[ri]
                oid = off_ids[ci]
                assignments[did] = oid
                self._matchup_secs[did][oid] += self.seconds_per_frame

        self._current = assignments
        return assignments

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_matchup_seconds(
        self,
        defender_id: int,
    ) -> Dict[int, float]:
        """Return {offensive_id: seconds} for a defender."""
        return dict(self._matchup_secs[defender_id])

    def get_primary_matchup(self, defender_id: int) -> Optional[int]:
        """Return the offensive player a defender guarded the most."""
        secs = self._matchup_secs[defender_id]
        return max(secs, key=secs.get) if secs else None

    def get_total_matchup_seconds(self, defender_id: int) -> float:
        """Total seconds a defender spent in any matchup."""
        return sum(self._matchup_secs[defender_id].values())

    def get_current_matchup(self, defender_id: int) -> Optional[int]:
        return self._current.get(defender_id)

    def all_matchup_times(self) -> Dict[int, Dict[int, float]]:
        """Return the full matchup seconds table."""
        return {did: dict(v) for did, v in self._matchup_secs.items()}

    def reset(self) -> None:
        self._matchup_secs.clear()
        self._current.clear()
