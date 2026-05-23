"""
scripts/validate_pipeline.py

Validates pipeline event detection against official play-by-play ground truth.

Two things this script does:
  1. Auto time-alignment  — finds the offset between video timestamps and
     game clock using cross-correlation of detected vs PBP events.
  2. Metric computation   — precision, recall, F1 per event type, with a
     configurable tolerance window (default ±10 s).

Usage
─────
  # Validate existing ECF 2012 clips against PBP
  python scripts/validate_pipeline.py --game 201206070BOS

  # Validate a specific set of clip files against a game
  python scripts/validate_pipeline.py --game 201606190GSW \\
      --clips data/raw/clip_A.mp4 data/raw/clip_B.mp4

  # Validate all clips that have a game mapping
  python scripts/validate_pipeline.py --all

  # Show time alignment without computing metrics (diagnostic)
  python scripts/validate_pipeline.py --game 201206070BOS --dry-run

  # Use a different tolerance window
  python scripts/validate_pipeline.py --game 201206070BOS --tolerance 15

Event type mapping
──────────────────
PBP event names ↔ pipeline EventType values are matched via ETYPE_MAP below.
PBP steals appear as event_type='steal' where player2 = defender.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT      = Path(__file__).resolve().parent.parent
DB_PATH   = ROOT / "data" / "pbp.db"
PROC_DIR  = ROOT / "data" / "processed"

# ── Known clip → game_id mapping for existing processed clips ─────────────────
# Add entries here as new games are processed.
CLIP_GAME_MAP: Dict[str, str] = {
    "clip_10m00_18m00": "201206070BOS",
    "clip_26m00_34m00": "201206070BOS",
    "clip_40m00_48m00": "201206070BOS",
    "clip_55m00_63m00": "201206070BOS",
    "clip_70m00_78m00": "201206070BOS",
}

# ── PBP → pipeline event type aliases ────────────────────────────────────────
# Keys are PBP event_type values; values are sets of pipeline event_type strings.
ETYPE_MAP: Dict[str, List[str]] = {
    "steal":             ["steal"],
    "block":             ["block"],
    "shot_made_2pt":     ["shot_made_2pt"],
    "shot_made_3pt":     ["shot_made_3pt"],
    "shot_miss_2pt":     ["shot_miss_2pt"],
    "shot_miss_3pt":     ["shot_miss_3pt"],
    "turnover":          ["turnover"],
    "defensive_rebound": ["defensive_rebound"],
    "offensive_rebound": ["offensive_rebound"],
    "assist":            ["assist"],
    "free_throw_made":   ["free_throw_made"],
    "free_throw_miss":   ["free_throw_miss"],
}

# Event types to use for time alignment (most reliably detected on both sides)
ALIGNMENT_TYPES = ["steal", "block"]

# Minimum PBP events of alignment types needed to attempt auto-alignment
MIN_ALIGNMENT_EVENTS = 2

# Tolerance window: detected event matches PBP event if within this many seconds
DEFAULT_TOLERANCE = 10.0

# Clip duration guard: only match PBP events whose game_secs (adjusted by offset)
# falls within [0, clip_duration + CLIP_BUFFER_SEC]
CLIP_BUFFER_SEC = 30.0


# ── Database helpers ──────────────────────────────────────────────────────────

def load_pbp(game_id: str) -> List[dict]:
    """Load all PBP events for a game from the local database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM pbp_events WHERE game_id = ? ORDER BY game_secs",
        (game_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_stored_games() -> List[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM games ORDER BY date").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Detection helpers ─────────────────────────────────────────────────────────

def load_detections(clip_name: str) -> List[dict]:
    """Load pipeline-detected events for a clip from data/processed/."""
    path = PROC_DIR / f"{clip_name}_events.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def clip_duration(clip_name: str) -> float:
    """Estimate clip duration from the last event timestamp."""
    events = load_detections(clip_name)
    if not events:
        return 600.0  # default 10 min
    return max(e.get("timestamp_sec", 0) for e in events) + 60.0


# ── Time alignment ────────────────────────────────────────────────────────────

def find_time_offset(
    detections: List[dict],
    pbp_events: List[dict],
    alignment_types: List[str] = ALIGNMENT_TYPES,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Tuple[Optional[float], int, str]:
    """
    Find the offset such that:
        video_timestamp ≈ pbp_game_secs + offset

    Strategy: for each (detected, PBP) pair of the same event type,
    compute candidate_offset = det_timestamp - pbp_game_secs.
    Find the offset cluster with the most votes within ±tolerance.

    Returns
    ───────
    (offset_sec, n_votes, confidence_label)
    offset is None if alignment fails.
    """
    candidates: List[float] = []

    for atype in alignment_types:
        det = [e for e in detections if e.get("event_type") == atype]
        pbp = [e for e in pbp_events  if e.get("event_type") == atype]
        for d in det:
            for p in pbp:
                candidates.append(d["timestamp_sec"] - p["game_secs"])

    if not candidates:
        return None, 0, "FAILED"

    # Vote: for each candidate, count how many others agree within ±tolerance
    best_offset = None
    best_votes  = 0
    for cand in candidates:
        votes = sum(1 for c in candidates if abs(c - cand) <= tolerance)
        if votes > best_votes:
            best_votes  = votes
            best_offset = cand

    if best_votes < MIN_ALIGNMENT_EVENTS:
        confidence = "LOW"
    elif best_votes < 5:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    # Refine: average of the winning cluster
    cluster = [c for c in candidates if abs(c - best_offset) <= tolerance]
    best_offset = sum(cluster) / len(cluster)

    return round(best_offset, 1), best_votes, confidence


# ── Metric computation ────────────────────────────────────────────────────────

def match_events(
    detections: List[dict],
    pbp_events: List[dict],
    offset: float,
    tolerance: float,
    pipeline_types: List[str],
    pbp_type: str,
    clip_dur: float,
) -> Tuple[int, int, int]:
    """
    Compute (TP, FP, FN) for one event category.

    A detected event matches a PBP event if they are within `tolerance` seconds
    of each other (after applying the offset).  Each PBP event can be matched
    at most once (greedy, nearest first).
    """
    det = [e for e in detections if e.get("event_type") in pipeline_types]

    # Only PBP events that fall inside the clip window
    pbp_in_window = [
        e for e in pbp_events
        if e.get("event_type") == pbp_type
        and 0 <= (e["game_secs"] + offset) <= clip_dur + CLIP_BUFFER_SEC
    ]

    matched_pbp: set = set()
    tp = fp = 0

    for det_ev in sorted(det, key=lambda e: e["timestamp_sec"]):
        det_t = det_ev["timestamp_sec"]
        best_i, best_dist = None, float("inf")

        for i, pbp_ev in enumerate(pbp_in_window):
            if i in matched_pbp:
                continue
            pbp_video_t = pbp_ev["game_secs"] + offset
            dist = abs(det_t - pbp_video_t)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_i    = i

        if best_i is not None:
            tp += 1
            matched_pbp.add(best_i)
        else:
            fp += 1

    fn = len(pbp_in_window) - len(matched_pbp)
    return tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
    }


# ── Per-game validation ───────────────────────────────────────────────────────

def validate_game(
    game_id: str,
    clip_names: List[str],
    tolerance: float = DEFAULT_TOLERANCE,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Validate all clips for one game against PBP ground truth.

    Returns a dict with per-event-type metrics and alignment info.
    """
    pbp_events = load_pbp(game_id)
    if not pbp_events:
        print(f"  No PBP data for {game_id} — run fetch_pbp.py first.")
        return {}

    # Aggregate detections across all clips (timestamps are per-clip, reset each clip)
    # We process each clip independently and combine metrics.
    all_metrics: Dict[str, Dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    alignment_results = []

    for clip_name in clip_names:
        detections = load_detections(clip_name)
        if not detections:
            if verbose:
                print(f"  [{clip_name}] No detections — skipping")
            continue

        clip_dur = clip_duration(clip_name)
        offset, n_votes, conf = find_time_offset(detections, pbp_events, tolerance=tolerance)

        if verbose:
            if offset is not None:
                print(f"  [{clip_name}] offset={offset:+.1f}s  votes={n_votes}  confidence={conf}")
            else:
                print(f"  [{clip_name}] Time alignment FAILED — not enough matching events")

        alignment_results.append({
            "clip": clip_name, "offset": offset,
            "votes": n_votes, "confidence": conf,
        })

        if dry_run or offset is None:
            continue

        for pbp_type, pipeline_types in ETYPE_MAP.items():
            tp, fp, fn = match_events(
                detections, pbp_events, offset, tolerance,
                pipeline_types, pbp_type, clip_dur,
            )
            all_metrics[pbp_type]["tp"] += tp
            all_metrics[pbp_type]["fp"] += fp
            all_metrics[pbp_type]["fn"] += fn

    # Compute final metrics
    results = {}
    for etype, counts in all_metrics.items():
        if counts["tp"] + counts["fp"] + counts["fn"] > 0:
            results[etype] = compute_metrics(
                counts["tp"], counts["fp"], counts["fn"]
            )

    return {
        "game_id":   game_id,
        "clips":     clip_names,
        "alignment": alignment_results,
        "metrics":   results,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(result: dict, game_id: str) -> None:
    if not result:
        return

    print(f"\n{'═'*65}")
    print(f"  Game: {game_id}")
    clips = result.get("clips", [])
    print(f"  Clips: {', '.join(clips)}")
    print()

    # Alignment summary
    for a in result.get("alignment", []):
        conf  = a["confidence"]
        off   = a["offset"]
        votes = a["votes"]
        flag  = "✓" if conf in ("HIGH", "MEDIUM") else "⚠"
        offset_str = f"{off:+.1f}s" if off is not None else "N/A"
        print(f"  {flag} {a['clip']:<28} offset={offset_str:<8} votes={votes}  {conf}")

    metrics = result.get("metrics", {})
    if not metrics:
        print("\n  (dry-run or alignment failed — no metrics computed)")
        print(f"{'═'*65}")
        return

    print(f"\n  {'Event Type':<22} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'-'*56}")

    # Sort: best F1 first
    for etype, m in sorted(metrics.items(), key=lambda x: -x[1]["f1"]):
        print(
            f"  {etype:<22} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}"
        )

    # Macro averages (over event types with at least 1 ground truth event)
    active = [m for m in metrics.values() if m["tp"] + m["fn"] > 0]
    if active:
        mac_p  = sum(m["precision"] for m in active) / len(active)
        mac_r  = sum(m["recall"]    for m in active) / len(active)
        mac_f1 = sum(m["f1"]        for m in active) / len(active)
        print(f"  {'-'*56}")
        print(f"  {'MACRO AVERAGE':<22} {mac_p:>6.3f} {mac_r:>6.3f} {mac_f1:>6.3f}")

    print(f"{'═'*65}\n")


def print_aggregate(all_results: List[dict], tolerance: float) -> None:
    """Print aggregate metrics across all validated games."""
    combined: Dict[str, Dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in all_results:
        for etype, m in r.get("metrics", {}).items():
            combined[etype]["tp"] += m["tp"]
            combined[etype]["fp"] += m["fp"]
            combined[etype]["fn"] += m["fn"]

    if not combined:
        return

    print(f"\n{'═'*65}")
    print(f"  AGGREGATE ({len(all_results)} games, tolerance={tolerance}s)")
    print(f"  {'Event Type':<22} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*58}")
    for etype, counts in sorted(combined.items(), key=lambda x: -(x[1]["tp"])):
        m = compute_metrics(counts["tp"], counts["fp"], counts["fn"])
        print(
            f"  {etype:<22} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {counts['tp']:>5} {counts['fp']:>5} {counts['fn']:>5}"
        )
    active = [
        compute_metrics(c["tp"], c["fp"], c["fn"])
        for c in combined.values()
        if c["tp"] + c["fn"] > 0
    ]
    if active:
        print(f"  {'-'*58}")
        print(
            f"  {'MACRO AVERAGE':<22} "
            f"{sum(m['precision'] for m in active)/len(active):>6.3f} "
            f"{sum(m['recall']    for m in active)/len(active):>6.3f} "
            f"{sum(m['f1']        for m in active)/len(active):>6.3f}"
        )
    print(f"{'═'*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game",      help="Single game_id to validate")
    ap.add_argument("--clips",     nargs="*",
                    help="Clip stems or .mp4 paths (default: all mapped clips for game)")
    ap.add_argument("--all",       action="store_true",
                    help="Validate all clips that have a CLIP_GAME_MAP entry")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help=f"Match tolerance in seconds (default {DEFAULT_TOLERANCE})")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Only compute time alignment, skip P/R/F1")
    ap.add_argument("--quiet",     action="store_true")
    args = ap.parse_args()

    all_results = []

    if args.all:
        # Group clips by game_id
        game_clips: Dict[str, List[str]] = defaultdict(list)
        for clip, gid in CLIP_GAME_MAP.items():
            game_clips[gid].append(clip)
        for gid, clips in game_clips.items():
            print(f"\nValidating {gid} ({len(clips)} clips)...")
            result = validate_game(gid, clips, args.tolerance, args.dry_run, not args.quiet)
            print_report(result, gid)
            all_results.append(result)
        if len(all_results) > 1:
            print_aggregate(all_results, args.tolerance)
        return

    if args.game:
        # Resolve clip names
        if args.clips:
            clip_names = [
                Path(c).stem.replace("_events", "") for c in args.clips
            ]
        else:
            # Default: all clips mapped to this game
            clip_names = [c for c, g in CLIP_GAME_MAP.items() if g == args.game]
            if not clip_names:
                # Fall back to all processed clips
                clip_names = [
                    p.stem.replace("_events", "")
                    for p in sorted(PROC_DIR.glob("*_events.json"))
                ]

        print(f"\nValidating {args.game} ({len(clip_names)} clips)...")
        result = validate_game(
            args.game, clip_names, args.tolerance, args.dry_run, not args.quiet
        )
        print_report(result, args.game)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
