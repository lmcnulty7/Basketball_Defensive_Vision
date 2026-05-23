"""
src/tracking/player_identity.py

Resolves track_id + jersey_number → player_name using the roster database.

The resolver is initialized with a game_id.  It looks up jersey numbers
from the `rosters` table in data/pbp.db and maps them to player names.
When a JerseyReader locks a jersey number for a track, call resolve() to
get the player's name — and it persists for the lifetime of the clip.

Usage
─────
  identity = PlayerIdentityResolver("201606190GSW")
  # After JerseyReader locks jersey #23 for track 7:
  name = identity.resolve(track_id=7, jersey_number="23", team_side="away")
  # → "LeBron James"
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "pbp.db"


class PlayerIdentityResolver:
    """
    Maps (jersey_number, team_side) → player_name for a specific game.

    Parameters
    ──────────
    game_id : The game_id string (e.g. "201606190GSW").  Must exist in pbp.db.
    """

    def __init__(self, game_id: str, db_path: Path = DB_PATH) -> None:
        self._game_id = game_id
        # {(team_side, jersey_number): player_name}
        self._roster: Dict[tuple, str] = {}
        # track_id → player_name (once resolved)
        self._resolved: Dict[int, str] = {}
        self._load_roster(db_path)

    def _load_roster(self, db_path: Path) -> None:
        if not db_path.exists():
            logger.warning("PlayerIdentityResolver: pbp.db not found at %s", db_path)
            return
        try:
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("""
                SELECT team_side, jersey_number, player_name
                FROM rosters WHERE game_id = ?
            """, (self._game_id,)).fetchall()
            conn.close()
            for side, jersey, name in rows:
                self._roster[(side, jersey)] = name
                # Also index without team side (fallback)
                if ("any", jersey) not in self._roster:
                    self._roster[("any", jersey)] = name
            logger.info(
                "PlayerIdentityResolver: loaded %d players for %s",
                len(rows), self._game_id,
            )
        except Exception as e:
            logger.warning("PlayerIdentityResolver: DB error (%s)", e)

    def resolve(
        self,
        track_id: int,
        jersey_number: str,
        team_side: Optional[str] = None,
    ) -> Optional[str]:
        """
        Look up a player name by jersey number.

        Parameters
        ──────────
        track_id      : The track ID to cache the result against.
        jersey_number : Jersey number string (e.g. "23").
        team_side     : "home" or "away" (None = try both).

        Returns
        ───────
        Player name string, or None if not found.
        """
        if track_id in self._resolved:
            return self._resolved[track_id]

        name = None
        if team_side:
            name = self._roster.get((team_side, jersey_number))
        if name is None:
            name = self._roster.get(("any", jersey_number))

        if name:
            self._resolved[track_id] = name
            logger.debug(
                "Identity resolved: track #%d → #%s %s (%s)",
                track_id, jersey_number, name, team_side or "?",
            )
        return name

    def get_name(self, track_id: int) -> Optional[str]:
        """Return cached player name for a track, or None."""
        return self._resolved.get(track_id)

    def display_name(self, track_id: int) -> str:
        """Return player name if known, else '#track_id'."""
        name = self._resolved.get(track_id)
        return name if name else f"#{track_id}"

    @property
    def has_roster(self) -> bool:
        return bool(self._roster)
