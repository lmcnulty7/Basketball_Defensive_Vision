"""
scripts/analyze_stats.py

Read aggregated_stats.csv + all events JSON files and print a clean
defensive leaderboard with per-minute rates.

Usage
─────
    python scripts/analyze_stats.py
    python scripts/analyze_stats.py --csv data/processed/aggregated_stats.csv
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

PROCESSED = Path("data/processed")


def load_stats(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def load_all_events(processed_dir: Path) -> list[dict]:
    events = []
    for p in processed_dir.glob("*_events.json"):
        with open(p) as f:
            evs = json.load(f)
        for ev in evs:
            ev["_clip"] = p.stem.replace("_events", "")
        events.extend(evs)
    return events


def total_game_minutes(processed_dir: Path) -> float:
    """Sum duration_sec from all stat CSVs (rough proxy: max timestamp in events)."""
    total_sec = 0.0
    for p in processed_dir.glob("*_events.json"):
        with open(p) as f:
            evs = json.load(f)
        if evs:
            total_sec += max(e.get("timestamp_sec", 0) for e in evs)
    return max(total_sec / 60.0, 1.0)


def event_counts_by_player(events: list[dict]) -> dict[int, dict]:
    counts: dict[int, dict] = defaultdict(lambda: defaultdict(int))
    for ev in events:
        pid = ev.get("primary_player_id", -1)
        if pid < 0:
            continue
        etype = ev.get("event_type", "unknown")
        counts[pid][etype] += 1
        counts[pid]["total"] += 1
    return counts


def print_leaderboard(stats: list[dict], event_counts: dict, total_min: float) -> None:
    print(f"\n{'═'*72}")
    print(f"  DEFENSIVE LEADERBOARD  ({total_min:.1f} min of footage)")
    print(f"{'═'*72}")
    hdr = f"  {'ID':>4}  {'steals':>6}  {'blocks':>6}  {'defl':>5}  {'cont':>5}  "
    hdr += f"{'defl/m':>6}  {'cont/m':>6}  {'total/m':>7}"
    print(hdr)
    print("  " + "─" * 68)

    rows = []
    for row in stats:
        tid = int(row["track_id"])
        if tid < 0:
            continue
        steals  = int(row.get("steals", 0))
        blocks  = int(row.get("blocks", 0))
        defl    = int(row.get("deflections", 0))
        cont    = int(row.get("contested_2pt", 0)) + int(row.get("contested_3pt", 0))
        total   = steals + blocks + defl + cont
        if total == 0:
            continue

        ev = event_counts.get(tid, {})
        defl_pm  = defl  / total_min
        cont_pm  = cont  / total_min
        total_pm = total / total_min

        rows.append((total_pm, tid, steals, blocks, defl, cont, defl_pm, cont_pm, total_pm))

    rows.sort(reverse=True)

    for _, tid, steals, blocks, defl, cont, defl_pm, cont_pm, total_pm in rows:
        print(f"  {tid:>4}  {steals:>6}  {blocks:>6}  {defl:>5}  {cont:>5}  "
              f"{defl_pm:>6.2f}  {cont_pm:>6.2f}  {total_pm:>7.2f}")

    print(f"{'═'*72}\n")


def event_type_summary(events: list[dict]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for ev in events:
        counts[ev.get("event_type", "?")] += 1
    print("  Event breakdown:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<25} {v}")
    print()


def contested_shot_quality(events: list[dict]) -> None:
    """Breakdown of contested shot tightness."""
    levels: dict[str, int] = defaultdict(int)
    for ev in events:
        if "contested" in ev.get("event_type", ""):
            level = ev.get("metadata", {}).get("contest_level", "unknown")
            levels[level] += 1
    if not levels:
        return
    total = sum(levels.values())
    print("  Contested shot quality:")
    for level in ["tight", "challenged", "open"]:
        n = levels.get(level, 0)
        pct = 100 * n / total if total else 0
        print(f"    {level:<12} {n:>3}  ({pct:.0f}%)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(PROCESSED / "aggregated_stats.csv"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Fall back to single-clip stats
        candidates = sorted(PROCESSED.glob("*_stats.csv"))
        if not candidates:
            print("No stats CSV found. Run the pipeline first.")
            return
        csv_path = candidates[-1]
        print(f"Using: {csv_path.name}")

    stats  = load_stats(csv_path)
    events = load_all_events(PROCESSED)
    total_min = total_game_minutes(PROCESSED)
    ev_counts = event_counts_by_player(events)

    print(f"\n  Total clips analysed : {len(list(PROCESSED.glob('*_events.json')))}")
    print(f"  Total events         : {len(events)}")
    event_type_summary(events)
    contested_shot_quality(events)
    print_leaderboard(stats, ev_counts, total_min)


if __name__ == "__main__":
    main()
