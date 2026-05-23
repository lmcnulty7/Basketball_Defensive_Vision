"""
api/main.py

FastAPI backend for the Basketball Defensive Vision dashboard.

Endpoints
─────────
GET /api/stats          — aggregated per-player stats across all clips
GET /api/events         — all events (optionally filtered by clip or type)
GET /api/clips          — list of processed clips with metadata
GET /api/leaderboard    — top defenders sorted by total defensive activity

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Basketball Vision API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

PROCESSED = Path("data/processed")


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


INT_FIELDS = [
    "track_id", "steals", "blocks", "deflections",
    "contested_2pt", "contested_3pt", "charges_drawn",
    "def_rebounds", "off_rebounds",
    "points", "fgm_2pt", "fga_2pt", "fgm_3pt", "fga_3pt",
    "ftm", "fta", "assists", "turnovers",
]
FLOAT_FIELDS = [
    "matchup_time_sec", "avg_speed_mph", "def_speed_mph",
    "max_speed_mph", "contest_pct_tight", "fg_pct", "three_pct",
]


def _coerce_stats(rows: list[dict]) -> list[dict]:
    for row in rows:
        for k in INT_FIELDS:
            if k in row:
                try: row[k] = int(row[k])
                except (ValueError, TypeError): row[k] = 0
        for k in FLOAT_FIELDS:
            if k in row:
                try: row[k] = float(row[k])
                except (ValueError, TypeError): row[k] = 0.0
    return rows


@app.get("/api/stats")
def get_stats():
    """Aggregated per-player defensive stats across all processed clips."""
    rows = _load_csv(PROCESSED / "aggregated_stats.csv")
    if not rows:
        # Fall back to single-clip stats
        paths = sorted(PROCESSED.glob("*_stats.csv"))
        rows = _load_csv(paths[-1]) if paths else []
    return _coerce_stats(rows)


@app.get("/api/clips")
def get_clips():
    """List all processed clips with event counts and frame counts."""
    clips = []
    for events_path in sorted(PROCESSED.glob("*_events.json")):
        stem = events_path.stem.replace("_events", "")
        events = _load_json(events_path)
        stats_path = PROCESSED / f"{stem}_stats.csv"
        stats = _load_csv(stats_path)

        by_type: dict[str, int] = {}
        for ev in events:
            t = ev.get("event_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        duration = max((e.get("timestamp_sec", 0) for e in events), default=0)

        clips.append({
            "clip":          stem,
            "n_events":      len(events),
            "n_players":     len(stats),
            "duration_sec":  round(duration, 1),
            "event_types":   by_type,
        })
    return clips


@app.get("/api/events")
def get_events(
    clip: Optional[str] = Query(None, description="Filter by clip name"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    min_confidence: float = Query(0.0, description="Minimum confidence threshold"),
):
    """All detected defensive events, with optional filters."""
    events: list[dict] = []
    pattern = f"{clip}_events.json" if clip else "*_events.json"
    for path in sorted(PROCESSED.glob(pattern)):
        stem = path.stem.replace("_events", "")
        for ev in _load_json(path):
            ev["clip"] = stem
            events.append(ev)

    if event_type:
        events = [e for e in events if e.get("event_type") == event_type]
    if min_confidence > 0:
        events = [e for e in events if e.get("confidence", 0) >= min_confidence]

    return events


@app.get("/api/boxscore")
def get_boxscore(clip: Optional[str] = Query(None)):
    """Full per-player box score (offensive + defensive stats)."""
    if clip:
        rows = _load_csv(PROCESSED / f"{clip}_stats.csv")
    else:
        rows = _load_csv(PROCESSED / "aggregated_stats.csv")
        if not rows:
            paths = sorted(PROCESSED.glob("*_stats.csv"))
            rows = _load_csv(paths[-1]) if paths else []
    return _coerce_stats(rows)


@app.get("/api/shotchart")
def get_shotchart(
    clip: Optional[str] = Query(None),
    player_id: Optional[int] = Query(None),
):
    """Shot locations for court visualization (made + missed 2PT and 3PT)."""
    shot_types = {
        "shot_made_2pt", "shot_made_3pt",
        "shot_miss_2pt", "shot_miss_3pt",
    }
    events: list[dict] = []
    pattern = f"{clip}_events.json" if clip else "*_events.json"
    for path in sorted(PROCESSED.glob(pattern)):
        stem = path.stem.replace("_events", "")
        for ev in _load_json(path):
            if ev.get("event_type") not in shot_types:
                continue
            if player_id is not None and ev.get("primary_player_id") != player_id:
                continue
            ev["clip"] = stem
            events.append(ev)
    return events


@app.get("/api/leaderboard")
def get_leaderboard(top_n: int = Query(20, description="Number of players to return")):
    """Top defenders ranked by total defensive actions per minute."""
    stats = _coerce_stats(_load_csv(PROCESSED / "aggregated_stats.csv"))
    events = []
    for path in PROCESSED.glob("*_events.json"):
        events.extend(_load_json(path))

    total_min = max(
        (e.get("timestamp_sec", 0) for e in events), default=60
    ) / 60.0

    leaderboard = []
    for row in stats:
        tid = row.get("track_id", -1)
        if tid < 0:
            continue
        steals  = row.get("steals", 0)
        blocks  = row.get("blocks", 0)
        defl    = row.get("deflections", 0)
        cont    = row.get("contested_2pt", 0) + row.get("contested_3pt", 0)
        points  = row.get("points", 0)
        assists = row.get("assists", 0)
        total   = steals + blocks + defl + cont
        clips   = row.get("clips_seen", 1)

        leaderboard.append({
            "track_id":        tid,
            "points":          points,
            "assists":         assists,
            "steals":          steals,
            "blocks":          blocks,
            "deflections":     defl,
            "contested":       cont,
            "total_actions":   total,
            "clips_seen":      clips,
            "fg_pct":          row.get("fg_pct", 0.0),
            "actions_per_min": round(total / max(total_min, 1), 3),
        })

    leaderboard.sort(key=lambda r: r["actions_per_min"], reverse=True)
    return leaderboard[:top_n]
