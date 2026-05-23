"""
scripts/fetch_pbp.py

Scrape play-by-play from basketball-reference.com and store in a local
SQLite database (data/pbp.db).  Run once per game — subsequent calls are
no-ops if the game is already stored.

Usage
─────
  # Single game by URL
  python scripts/fetch_pbp.py --url "https://www.basketball-reference.com/boxscores/pbp/201606190GSW.html"

  # All 10 validation games at once
  python scripts/fetch_pbp.py --batch

  # List all stored games
  python scripts/fetch_pbp.py --list

  # Force re-scrape a game
  python scripts/fetch_pbp.py --url "..." --force

Database schema
───────────────
  games (
    game_id      TEXT PRIMARY KEY,   -- e.g. "201606190GSW"
    date         TEXT,               -- "2016-06-19"
    home_team    TEXT,               -- "GSW"
    away_team    TEXT,               -- "CLE"
    home_score   INTEGER,
    away_score   INTEGER,
    source_url   TEXT,
    fetched_at   TEXT
  )

  pbp_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id      TEXT,
    quarter      INTEGER,
    game_clock   TEXT,               -- "11:42.0"
    game_secs    REAL,               -- seconds elapsed from tip-off
    home_score   INTEGER,
    away_score   INTEGER,
    event_text   TEXT,               -- raw description
    event_type   TEXT,               -- normalized: shot_made_2pt, steal, etc.
    player       TEXT,               -- primary player name
    player2      TEXT,               -- secondary (blocker, stealer, assister)
    team         TEXT,               -- "home" or "away"
    shot_dist_ft REAL,
    is_3pt       INTEGER,            -- 0/1
    FOREIGN KEY (game_id) REFERENCES games(game_id)
  )
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "pbp.db"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.basketball-reference.com/",
}

# The 10 validation/test games — add more here as needed
BATCH_GAMES = [
    # Validation set
    ("https://www.basketball-reference.com/boxscores/pbp/201606190GSW.html", "2016 NBA Finals G7 CLE@GSW"),
    ("https://www.basketball-reference.com/boxscores/pbp/201306200MIA.html", "2013 NBA Finals G6 SAS@MIA"),
    ("https://www.basketball-reference.com/boxscores/pbp/201706120GSW.html", "2017 NBA Finals G5 GSW@CLE"),
    ("https://www.basketball-reference.com/boxscores/pbp/200806050BOS.html", "2008 NBA Finals G1 LAL@BOS"),
    ("https://www.basketball-reference.com/boxscores/pbp/200806080BOS.html", "2008 ECSF G7 CLE@BOS"),
    ("https://www.basketball-reference.com/boxscores/pbp/202501080MIL.html", "Bucks vs Spurs Jan 2025"),
    ("https://www.basketball-reference.com/boxscores/pbp/202501290PHO.html", "Suns vs Wolves Jan 2025"),
    # Test set (hold out — don't tune against these)
    ("https://www.basketball-reference.com/boxscores/pbp/201906130GSW.html", "2019 NBA Finals G6 TOR@GSW"),
    ("https://www.basketball-reference.com/boxscores/pbp/202010110MIA.html", "2020 NBA Finals G6 LAL@MIA"),
    ("https://www.basketball-reference.com/boxscores/pbp/202501250GSW.html", "Lakers vs Warriors Jan 2025"),
]

# Quarter duration in seconds (12 min regulation, 5 min OT)
_Q_DUR = {1: 720, 2: 720, 3: 720, 4: 720}


def _clock_to_secs(quarter: int, clock: str) -> float:
    """Convert quarter + game clock string to seconds elapsed from tip-off."""
    try:
        parts = clock.replace(",", ".").split(":")
        mins = float(parts[0])
        secs = float(parts[1]) if len(parts) > 1 else 0.0
        remaining = mins * 60 + secs
        q_start = sum(_Q_DUR.get(q, 300) for q in range(1, quarter))
        q_dur = _Q_DUR.get(quarter, 300)
        elapsed = q_start + (q_dur - remaining)
        return round(elapsed, 1)
    except Exception:
        return 0.0


def _normalize_event(text: str) -> tuple[str, str, Optional[str], Optional[float], int]:
    """
    Parse raw PBP text into (event_type, player, player2, shot_dist_ft, is_3pt).
    """
    t = text.strip()

    # Shot made
    m = re.search(r"makes (\d)-pt (\w+).*?from (\d+) ft", t)
    if m:
        pts, shot_type, dist = int(m.group(1)), m.group(2), float(m.group(3))
        is_3 = 1 if pts == 3 else 0
        etype = "shot_made_3pt" if is_3 else "shot_made_2pt"
        assist = re.search(r"assist by(.+?)$", t)
        p2 = assist.group(1).strip() if assist else None
        return etype, _extract_player(t), p2, dist, is_3

    # Dunk / layup made (no distance listed)
    m = re.search(r"(\w+) at rim", t)
    if m and "makes" in t.lower():
        is_3 = 0
        assist = re.search(r"assist by(.+?)$", t)
        p2 = assist.group(1).strip() if assist else None
        return "shot_made_2pt", _extract_player(t), p2, 0.0, 0

    # Shot missed
    m = re.search(r"misses (\d)-pt.*?from (\d+) ft", t)
    if m:
        pts, dist = int(m.group(1)), float(m.group(2))
        is_3 = 1 if pts == 3 else 0
        block = re.search(r"block by(.+?)$", t)
        p2 = block.group(1).strip() if block else None
        return ("shot_miss_3pt" if is_3 else "shot_miss_2pt"), _extract_player(t), p2, dist, is_3

    # Free throw
    if "free throw" in t.lower():
        if "makes" in t.lower():
            return "free_throw_made", _extract_player(t), None, None, 0
        if "misses" in t.lower():
            return "free_throw_miss", _extract_player(t), None, None, 0

    # Rebound
    if "rebound" in t.lower():
        if "defensive" in t.lower():
            return "defensive_rebound", _extract_player(t), None, None, 0
        if "offensive" in t.lower():
            return "offensive_rebound", _extract_player(t), None, None, 0

    # Turnover + steal
    if "turnover" in t.lower() or "Turnover" in t:
        steal = re.search(r"steal by(.+?)[\);]", t)
        p2 = steal.group(1).strip() if steal else None
        etype = "steal" if p2 else "turnover"
        return etype, _extract_player(t), p2, None, 0

    # Block (appears in missed shot — handled above)
    if "block" in t.lower() and "foul" not in t.lower():
        return "block", _extract_player(t), None, None, 0

    # Fouls
    if "personal foul" in t.lower():
        return "foul_personal", _extract_player(t), None, None, 0
    if "shooting foul" in t.lower():
        return "foul_shooting", _extract_player(t), None, None, 0
    if "offensive foul" in t.lower():
        return "foul_offensive", _extract_player(t), None, None, 0

    # Substitution / timeout — still stored, just labeled
    if "enters the game" in t.lower():
        return "substitution", _extract_player(t), None, None, 0
    if "timeout" in t.lower():
        return "timeout", "", None, None, 0
    if "violation" in t.lower():
        return "violation", _extract_player(t), None, None, 0

    return "other", _extract_player(t), None, None, 0


def _extract_player(text: str) -> str:
    """Extract the primary player name (first 'F. Lastname' pattern)."""
    m = re.search(r"([A-Z]\. [A-Za-z'\-]+)", text)
    return m.group(1) if m else ""


def _determine_team(col_idx: int) -> str:
    """PBP table has home events in col 1/2 and away in col 3/4."""
    return "home" if col_idx <= 2 else "away"


def scrape_game(url: str) -> Optional[dict]:
    """
    Scrape a single PBP page.  Returns a dict with 'game' metadata and
    'events' list, or None if the page is unavailable.
    """
    time.sleep(1.0)   # polite rate-limiting
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} for {url}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": "pbp"})
    if not table:
        print(f"  No PBP table found at {url}")
        return None

    # Extract game_id from URL
    game_id = re.search(r"/pbp/(\w+)\.html", url)
    game_id = game_id.group(1) if game_id else url.split("/")[-1]

    # Extract date and teams from scorebox
    date_str = ""
    home_team = game_id[-3:]   # last 3 chars of game_id = home team
    away_team = ""
    home_score = away_score = 0

    scorebox = soup.find("div", {"class": "scorebox"})
    if scorebox:
        date_div = scorebox.find("div", {"class": "scorebox_meta"})
        if date_div:
            date_text = date_div.find("div")
            if date_text:
                date_str = date_text.get_text(strip=True)
        teams = scorebox.find_all("a", {"itemprop": "name"})
        if len(teams) >= 2:
            away_team = teams[0].get_text(strip=True)
            home_team_name = teams[1].get_text(strip=True)
        scores = scorebox.find_all("div", {"class": "score"})
        if len(scores) >= 2:
            try: away_score = int(scores[0].get_text(strip=True))
            except: pass
            try: home_score = int(scores[1].get_text(strip=True))
            except: pass

    # Parse rows
    events = []
    current_quarter = 1
    home_pts = away_pts = 0

    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if not cols:
            continue

        # Quarter header row
        if len(cols) == 1 and cols[0].get("colspan"):
            text = cols[0].get_text(strip=True).lower()
            if "1st" in text: current_quarter = 1
            elif "2nd" in text: current_quarter = 2
            elif "3rd" in text: current_quarter = 3
            elif "4th" in text: current_quarter = 4
            elif "ot" in text: current_quarter += 1
            continue

        if len(cols) < 3:
            continue

        clock_text = cols[0].get_text(strip=True)
        if not re.match(r"\d+:\d+", clock_text):
            continue

        # Score column (index 3 in 6-col layout)
        score_text = ""
        event_text = ""
        team_side = "home"

        if len(cols) >= 6:
            # Standard layout: time | home_event | score | away_event
            home_ev = cols[1].get_text(strip=True)
            away_ev = cols[5].get_text(strip=True)
            score_text = cols[3].get_text(strip=True)
            if home_ev:
                event_text = home_ev
                team_side = "home"
            elif away_ev:
                event_text = away_ev
                team_side = "away"
        else:
            for i, col in enumerate(cols[1:], 1):
                ct = col.get_text(strip=True)
                if ct and not re.match(r"\d+-\d+", ct):
                    event_text = ct
                    team_side = _determine_team(i)
                    break
                elif re.match(r"\d+-\d+", ct):
                    score_text = ct

        if not event_text:
            continue

        # Parse score
        sm = re.match(r"(\d+)-(\d+)", score_text)
        if sm:
            away_pts = int(sm.group(1))
            home_pts = int(sm.group(2))

        game_secs = _clock_to_secs(current_quarter, clock_text)
        etype, player, player2, dist, is_3pt = _normalize_event(event_text)

        events.append({
            "quarter":      current_quarter,
            "game_clock":   clock_text,
            "game_secs":    game_secs,
            "home_score":   home_pts,
            "away_score":   away_pts,
            "event_text":   event_text,
            "event_type":   etype,
            "player":       player,
            "player2":      player2 or "",
            "team":         team_side,
            "shot_dist_ft": dist,
            "is_3pt":       is_3pt,
        })

    return {
        "game": {
            "game_id":    game_id,
            "date":       date_str,
            "home_team":  home_team,
            "away_team":  away_team,
            "home_score": home_score,
            "away_score": away_score,
            "source_url": url,
            "fetched_at": datetime.utcnow().isoformat(),
        },
        "events": events,
    }


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id    TEXT PRIMARY KEY,
            date       TEXT,
            home_team  TEXT,
            away_team  TEXT,
            home_score INTEGER,
            away_score INTEGER,
            source_url TEXT,
            fetched_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pbp_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id      TEXT,
            quarter      INTEGER,
            game_clock   TEXT,
            game_secs    REAL,
            home_score   INTEGER,
            away_score   INTEGER,
            event_text   TEXT,
            event_type   TEXT,
            player       TEXT,
            player2      TEXT,
            team         TEXT,
            shot_dist_ft REAL,
            is_3pt       INTEGER,
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pbp_game ON pbp_events(game_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pbp_type ON pbp_events(event_type)")
    conn.commit()
    return conn


def store_game(conn: sqlite3.Connection, data: dict, force: bool = False) -> bool:
    game = data["game"]
    game_id = game["game_id"]

    existing = conn.execute(
        "SELECT game_id FROM games WHERE game_id = ?", (game_id,)
    ).fetchone()

    if existing and not force:
        print(f"  Already stored: {game_id} — skipping (use --force to re-scrape)")
        return False

    if existing and force:
        conn.execute("DELETE FROM pbp_events WHERE game_id = ?", (game_id,))
        conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))

    conn.execute("""
        INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game["game_id"], game["date"], game["home_team"], game["away_team"],
        game["home_score"], game["away_score"], game["source_url"], game["fetched_at"],
    ))

    conn.executemany("""
        INSERT INTO pbp_events
          (game_id, quarter, game_clock, game_secs, home_score, away_score,
           event_text, event_type, player, player2, team, shot_dist_ft, is_3pt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (game_id, e["quarter"], e["game_clock"], e["game_secs"],
         e["home_score"], e["away_score"], e["event_text"], e["event_type"],
         e["player"], e["player2"], e["team"], e["shot_dist_ft"], e["is_3pt"])
        for e in data["events"]
    ])

    conn.commit()
    n = len(data["events"])
    print(f"  Stored {game_id}: {n} events  ({game['away_team']} @ {game['home_team']})")
    return True


def list_games(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT g.game_id, g.date, g.away_team, g.home_team,
               g.away_score, g.home_score, COUNT(e.id) as n_events
        FROM games g
        LEFT JOIN pbp_events e ON g.game_id = e.game_id
        GROUP BY g.game_id
        ORDER BY g.date
    """).fetchall()

    if not rows:
        print("No games stored yet.")
        return

    print(f"\n{'game_id':<22} {'date':<12} {'matchup':<30} {'score':<10} {'events'}")
    print("-" * 85)
    for gid, date, away, home, ascore, hscore, n in rows:
        matchup = f"{away} @ {home}"
        score = f"{ascore}-{hscore}"
        print(f"{gid:<22} {date:<12} {matchup:<30} {score:<10} {n}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",   help="Basketball-reference PBP URL to scrape")
    ap.add_argument("--batch", action="store_true", help="Scrape all 10 validation games")
    ap.add_argument("--list",  action="store_true", help="List stored games")
    ap.add_argument("--force", action="store_true", help="Re-scrape even if already stored")
    ap.add_argument("--db",    default=str(DB_PATH), help="SQLite database path")
    args = ap.parse_args()

    conn = init_db(Path(args.db))

    if args.list:
        list_games(conn)
        return

    if args.batch:
        print(f"Scraping {len(BATCH_GAMES)} games...")
        for url, label in BATCH_GAMES:
            print(f"\n[{label}]")
            data = scrape_game(url)
            if data:
                store_game(conn, data, force=args.force)
        print("\nDone.")
        list_games(conn)
        return

    if args.url:
        data = scrape_game(args.url)
        if data:
            store_game(conn, data, force=args.force)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
