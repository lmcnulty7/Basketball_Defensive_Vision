"""
src/stats/aggregator.py

Full per-player box score aggregator — offensive and defensive stats.

Output schema (per player)
──────────────────────────
Offensive
  points             : int   — 2×FGM2 + 3×FGM3 + FTM
  fgm_2pt / fga_2pt  : int
  fgm_3pt / fga_3pt  : int
  fg_pct             : float
  three_pct          : float
  ftm / fta          : int   (when free throw detector is active)
  assists            : int
  off_rebounds       : int
  turnovers          : int

Defensive
  steals             : int
  blocks             : int
  deflections        : int
  contested_2pt      : int
  contested_3pt      : int
  def_rebounds       : int
  charges_drawn      : int

Movement / matchup
  matchup_time_sec   : float
  avg_speed_mph      : float
  def_speed_mph      : float
  max_speed_mph      : float
  contest_pct_tight  : float
"""

from __future__ import annotations

from typing import Dict, List

from src.events.event import Event, EventType
from src.stats.matchup_tracker import MatchupTracker
from src.stats.speed_calculator import SpeedCalculator


class StatsAggregator:
    """
    Assembles full per-player stats from the event log,
    MatchupTracker, and SpeedCalculator.
    """

    def compute(
        self,
        event_log: List[Event],
        matchup_tracker: MatchupTracker,
        speed_calculator: SpeedCalculator,
        track_ids: List[int],
    ) -> Dict[int, dict]:
        stats: Dict[int, dict] = {}
        for tid in track_ids:
            stats[tid] = self._empty_row(tid)

        for event in event_log:
            tid = event.primary_player_id
            if tid not in stats:
                stats[tid] = self._empty_row(tid)

            etype = event.event_type

            # ── Offensive ────────────────────────────────────────────────────
            if etype == EventType.SHOT_MADE_2PT:
                stats[tid]["fgm_2pt"] += 1
                stats[tid]["fga_2pt"] += 1
                stats[tid]["points"]  += 2
            elif etype == EventType.SHOT_MISS_2PT:
                stats[tid]["fga_2pt"] += 1
            elif etype == EventType.SHOT_MADE_3PT:
                stats[tid]["fgm_3pt"] += 1
                stats[tid]["fga_3pt"] += 1
                stats[tid]["points"]  += 3
            elif etype == EventType.SHOT_MISS_3PT:
                stats[tid]["fga_3pt"] += 1
            elif etype == EventType.FREE_THROW_MADE:
                stats[tid]["ftm"]    += 1
                stats[tid]["fta"]    += 1
                stats[tid]["points"] += 1
            elif etype == EventType.FREE_THROW_MISS:
                stats[tid]["fta"] += 1
            elif etype == EventType.ASSIST:
                stats[tid]["assists"] += 1
            elif etype == EventType.TURNOVER:
                stats[tid]["turnovers"] += 1
            elif etype == EventType.REBOUND_OFF:
                stats[tid]["off_rebounds"] += 1

            # ── Defensive ────────────────────────────────────────────────────
            elif etype == EventType.STEAL:
                stats[tid]["steals"] += 1
            elif etype == EventType.BLOCK:
                stats[tid]["blocks"] += 1
            elif etype == EventType.DEFLECTION:
                stats[tid]["deflections"] += 1
            elif etype == EventType.CONTESTED_2PT:
                stats[tid]["contested_2pt"] += 1
                if event.metadata.get("contest_level") == "tight":
                    stats[tid]["_tight_contests"] += 1
            elif etype == EventType.CONTESTED_3PT:
                stats[tid]["contested_3pt"] += 1
                if event.metadata.get("contest_level") == "tight":
                    stats[tid]["_tight_contests"] += 1
            elif etype == EventType.REBOUND_DEF:
                stats[tid]["def_rebounds"] += 1
            elif etype == EventType.CHARGE:
                stats[tid]["charges_drawn"] += 1

        # Matchup time
        for tid in stats:
            stats[tid]["matchup_time_sec"] = round(
                matchup_tracker.get_total_matchup_seconds(tid), 1
            )

        # Speed stats
        speed_data = speed_calculator.all_speeds()
        for tid in stats:
            sd = speed_data.get(tid, {})
            stats[tid]["avg_speed_mph"] = sd.get("avg_mph", 0.0)
            stats[tid]["def_speed_mph"] = sd.get("defensive_mph", 0.0)
            stats[tid]["max_speed_mph"] = sd.get("max_mph", 0.0)

        # Derived shooting percentages
        for tid in stats:
            fga = stats[tid]["fga_2pt"] + stats[tid]["fga_3pt"]
            fgm = stats[tid]["fgm_2pt"] + stats[tid]["fgm_3pt"]
            stats[tid]["fg_pct"] = round(fgm / fga, 3) if fga > 0 else 0.0
            fga3 = stats[tid]["fga_3pt"]
            stats[tid]["three_pct"] = (
                round(stats[tid]["fgm_3pt"] / fga3, 3) if fga3 > 0 else 0.0
            )
            total_contests = stats[tid]["contested_2pt"] + stats[tid]["contested_3pt"]
            tight = stats[tid].pop("_tight_contests", 0)
            stats[tid]["contest_pct_tight"] = (
                round(tight / total_contests, 3) if total_contests > 0 else 0.0
            )

        return stats

    def to_table(self, stats: Dict[int, dict]) -> List[dict]:
        """Return stats sorted by total impact (points + defensive actions)."""
        rows = list(stats.values())
        rows.sort(
            key=lambda r: (
                r["points"] +
                r["steals"] * 3 + r["blocks"] * 2 +
                r["assists"] * 1.5 + r["deflections"]
            ),
            reverse=True,
        )
        return rows

    @staticmethod
    def _empty_row(track_id: int) -> dict:
        return {
            "track_id":          track_id,
            # Offensive
            "points":            0,
            "fgm_2pt":           0,
            "fga_2pt":           0,
            "fgm_3pt":           0,
            "fga_3pt":           0,
            "fg_pct":            0.0,
            "three_pct":         0.0,
            "ftm":               0,
            "fta":               0,
            "assists":           0,
            "off_rebounds":      0,
            "turnovers":         0,
            # Defensive
            "steals":            0,
            "blocks":            0,
            "deflections":       0,
            "contested_2pt":     0,
            "contested_3pt":     0,
            "def_rebounds":      0,
            "charges_drawn":     0,
            # Movement
            "matchup_time_sec":  0.0,
            "avg_speed_mph":     0.0,
            "def_speed_mph":     0.0,
            "max_speed_mph":     0.0,
            "contest_pct_tight": 0.0,
            "_tight_contests":   0,
        }
