"""
scripts/fetch_rosters.py

Scrape team rosters (jersey numbers + player names) from basketball-reference
for every game stored in data/pbp.db.  Stores in a new `rosters` table.

Usage
─────
  python scripts/fetch_rosters.py           # fetch all stored games
  python scripts/fetch_rosters.py --list    # show what's stored
  python scripts/fetch_rosters.py --force   # re-scrape all
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "pbp.db"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.basketball-reference.com/",
}

# Basketball-reference uses 3-letter franchise codes
TEAM_NAME_TO_CODE: dict[str, str] = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK", "New Jersey Nets": "NJN",
    "Charlotte Hornets": "CHO", "Charlotte Bobcats": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New Orleans Hornets": "NOH",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS", "Seattle SuperSonics": "SEA",
    "Vancouver Grizzlies": "VAN",
}


def _game_year(game_id: str) -> int:
    """Extract NBA season end-year from game_id (e.g. '201606190GSW' → 2016)."""
    month = int(game_id[4:6])
    year  = int(game_id[:4])
    # NBA regular season starts Oct–Nov; if month < 7 it's still the same season
    return year if month >= 7 else year


def _team_code_from_game_id(game_id: str) -> str:
    """Home team code is the last 3 chars of game_id."""
    return game_id[-3:]


def _team_code_from_name(name: str) -> str | None:
    """Map full team name → 3-letter code."""
    # Try exact match first
    if name in TEAM_NAME_TO_CODE:
        return TEAM_NAME_TO_CODE[name]
    # Partial match fallback
    name_lower = name.lower()
    for full, code in TEAM_NAME_TO_CODE.items():
        if full.lower() in name_lower or name_lower in full.lower():
            return code
    return None


def scrape_roster(team_code: str, year: int) -> list[dict]:
    """
    Scrape the roster table from basketball-reference.
    Returns list of {jersey_number, player_name}.
    """
    url = f"https://www.basketball-reference.com/teams/{team_code}/{year}.html"
    time.sleep(1.0)
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        print(f"    HTTP {r.status_code} for {url}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": "roster"})
    if not table:
        print(f"    No roster table at {url}")
        return []

    players = []
    for row in table.find("tbody").find_all("tr"):
        if row.get("class") and "thead" in row.get("class", []):
            continue
        cols = row.find_all("td")
        if not cols:
            continue
        # Col 0 = jersey number, Col 1 = player name (link)
        try:
            jersey = cols[0].get_text(strip=True)
            name_cell = cols[1]
            name = name_cell.find("a")
            name = name.get_text(strip=True) if name else name_cell.get_text(strip=True)
            if name and jersey:
                players.append({"jersey_number": jersey, "player_name": name})
        except (IndexError, AttributeError):
            continue

    return players


def init_rosters_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rosters (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id       TEXT,
            team_code     TEXT,
            team_side     TEXT,
            jersey_number TEXT,
            player_name   TEXT,
            UNIQUE (game_id, team_code, jersey_number)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_roster_game ON rosters(game_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_roster_jersey ON rosters(jersey_number)")
    conn.commit()


def _get_both_team_codes(game_id: str) -> tuple[str, str]:
    """
    Scrape the boxscore page to find both team codes.
    Box score tables have ids like 'box-GSW-game-basic'.
    Returns (home_code, away_code).
    """
    url = f"https://www.basketball-reference.com/boxscores/{game_id}.html"
    time.sleep(0.8)
    r = requests.get(url, headers=HEADERS, timeout=20)
    home_code = _team_code_from_game_id(game_id)
    if r.status_code != 200:
        return home_code, ""
    soup = BeautifulSoup(r.text, "html.parser")
    codes = []
    for table in soup.find_all("table", id=True):
        m = re.match(r"box-([A-Z]{2,3})-game-basic", table.get("id", ""))
        if m:
            codes.append(m.group(1))
    codes = list(dict.fromkeys(codes))  # deduplicate preserving order
    away_code = next((c for c in codes if c != home_code), "")
    return home_code, away_code


def fetch_for_game(conn: sqlite3.Connection, game: dict, force: bool = False) -> None:
    game_id = game["game_id"]

    existing = conn.execute(
        "SELECT COUNT(*) FROM rosters WHERE game_id = ?", (game_id,)
    ).fetchone()[0]

    if existing > 0 and not force:
        print(f"  {game_id}: roster already stored ({existing} players) — skip")
        return

    if force:
        conn.execute("DELETE FROM rosters WHERE game_id = ?", (game_id,))

    year = _game_year(game_id)
    home_code, away_code = _get_both_team_codes(game_id)
    print(f"  Teams: {away_code} @ {home_code}")

    teams = [(home_code, "home")]
    if away_code:
        teams.append((away_code, "away"))
    else:
        print(f"  {game_id}: could not resolve away team")

    for team_code, side in teams:
        print(f"  Fetching {team_code} {year} roster ({side})...")
        players = scrape_roster(team_code, year)
        if not players:
            continue
        conn.executemany("""
            INSERT OR IGNORE INTO rosters
              (game_id, team_code, team_side, jersey_number, player_name)
            VALUES (?, ?, ?, ?, ?)
        """, [
            (game_id, team_code, side, p["jersey_number"], p["player_name"])
            for p in players
        ])
        conn.commit()
        print(f"    Stored {len(players)} players")


def list_rosters(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT game_id, team_code, team_side, COUNT(*) as n
        FROM rosters GROUP BY game_id, team_code, team_side
        ORDER BY game_id
    """).fetchall()
    if not rows:
        print("No rosters stored yet.")
        return
    print(f"\n{'game_id':<22} {'team':<6} {'side':<6} {'players'}")
    print("-" * 45)
    for gid, code, side, n in rows:
        print(f"{gid:<22} {code:<6} {side:<6} {n}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list",  action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--db",    default=str(DB_PATH))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    init_rosters_table(conn)

    if args.list:
        list_rosters(conn)
        return

    games = [dict(r) for r in conn.execute("SELECT * FROM games ORDER BY date").fetchall()]
    print(f"Fetching rosters for {len(games)} games...")
    for game in games:
        print(f"\n[{game['game_id']}]")
        fetch_for_game(conn, game, force=args.force)

    print("\nDone.")
    list_rosters(conn)


if __name__ == "__main__":
    main()
