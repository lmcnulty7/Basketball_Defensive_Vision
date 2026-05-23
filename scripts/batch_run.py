"""
scripts/batch_run.py

Download multiple time-segmented clips from a YouTube URL and run the
defensive stats pipeline on each one.  Aggregates per-player stats across
all clips into a single CSV.

Usage
─────
    python scripts/batch_run.py --url <youtube_url> --segments "18:00-26:00,26:00-34:00,34:00-42:00"
    python scripts/batch_run.py --local            # process all .mp4s already in data/raw/

Output
──────
    data/processed/<stem>_stats.csv  — per-clip stats
    data/processed/<stem>_events.json
    data/processed/aggregated_stats.csv — combined across all clips
"""

# ── Environment flags (before all imports) ────────────────────────────────────
import os, sys, types
os.environ.setdefault("OMP_NUM_THREADS",             "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",        "1")
os.environ.setdefault("MKL_NUM_THREADS",             "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK",        "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as _np
from scipy.optimize import linear_sum_assignment as _lsa
def _lapjv(cost, **kw):
    r, c = _lsa(cost.astype(float))
    x = _np.full(cost.shape[0], -1, dtype=_np.int32)
    y = _np.full(cost.shape[1], -1, dtype=_np.int32)
    for ri, ci in zip(r, c): x[ri] = ci; y[ci] = ri
    return 0.0, x, y
_lap = types.ModuleType("lap"); _lap.lapjv = _lapjv; sys.modules["lap"] = _lap

# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import json
import logging
import subprocess
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.runner import PipelineRunner
from src.pipeline.state import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_segment(url: str, start: str, end: str, out_path: Path) -> bool:
    """Download a time-segment of a YouTube video using yt-dlp."""
    if out_path.exists():
        logger.info("Already exists, skipping download: %s", out_path.name)
        return True

    cmd = [
        "yt-dlp",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
        url,
    ]
    logger.info("Downloading %s → %s", f"{start}-{end}", out_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("yt-dlp failed: %s", result.stderr[-500:])
        return False
    return out_path.exists()


def run_clip(clip_path: Path, runner: PipelineRunner) -> list[dict]:
    """Run pipeline on one clip, save per-clip files, return stats table."""
    import csv as _csv, json as _json
    logger.info("=" * 60)
    logger.info("Processing: %s", clip_path.name)
    runner.reset()
    try:
        output = runner.run(str(clip_path))
        output.print_summary()

        stem = clip_path.stem
        events_path = OUT_DIR / f"{stem}_events.json"
        stats_path  = OUT_DIR / f"{stem}_stats.csv"

        with open(events_path, "w") as f:
            _json.dump([e.to_dict() for e in output.event_log], f, indent=2)

        if output.stats_table:
            with open(stats_path, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=output.stats_table[0].keys())
                w.writeheader(); w.writerows(output.stats_table)

        logger.info("Saved → %s  %s", events_path.name, stats_path.name)

        for row in output.stats_table:
            row["clip"] = stem
        return output.stats_table
    except Exception as e:
        logger.error("Pipeline failed on %s: %s", clip_path.name, e, exc_info=True)
        return []


def aggregate(all_rows: list[dict]) -> list[dict]:
    """Sum per-player stats across clips, keyed by track_id."""
    totals: dict[int, dict] = defaultdict(lambda: defaultdict(int))
    numeric = ["steals", "blocks", "deflections", "contested_2pt", "contested_3pt",
               "charges_drawn", "def_rebounds", "matchup_time_sec"]
    for row in all_rows:
        tid = row["track_id"]
        for k in numeric:
            totals[tid][k] += row.get(k, 0)
        totals[tid]["track_id"] = tid
        totals[tid]["clips_seen"] = totals[tid].get("clips_seen", 0) + 1

    return sorted(totals.values(),
                  key=lambda r: r["deflections"] + r["steals"] + r["blocks"],
                  reverse=True)


def save_aggregated(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fields = ["track_id", "clips_seen", "steals", "blocks", "deflections",
              "contested_2pt", "contested_3pt", "charges_drawn",
              "def_rebounds", "matchup_time_sec"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Aggregated stats → %s", out_path)


def parse_segments(seg_str: str) -> list[tuple[str, str]]:
    """Parse 'HH:MM:SS-HH:MM:SS,...' into [(start,end), ...]."""
    out = []
    for seg in seg_str.split(","):
        seg = seg.strip()
        if "-" in seg:
            parts = seg.rsplit("-", 1)
            out.append((parts[0].strip(), parts[1].strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch download + pipeline runner")
    ap.add_argument("--url",      default=None, help="YouTube URL to download from")
    ap.add_argument("--segments", default=None,
                    help='Comma-separated time ranges, e.g. "18:00-26:00,26:00-34:00"')
    ap.add_argument("--local",    action="store_true",
                    help="Skip download; process all .mp4 files already in data/raw/")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml()
    runner = PipelineRunner(cfg)

    clips_to_process: list[Path] = []

    # ── Download phase ────────────────────────────────────────────────────────
    if not args.local:
        if not args.url:
            ap.error("Provide --url or --local")
        segments = parse_segments(args.segments) if args.segments else [
            ("18:00", "26:00"),
            ("26:00", "34:00"),
            ("34:00", "42:00"),
            ("42:00", "50:00"),
            ("50:00", "58:00"),
        ]
        for start, end in segments:
            slug = f"{start.replace(':', 'm')}_{end.replace(':', 'm')}".replace(" ", "")
            out_path = RAW_DIR / f"clip_{slug}.mp4"
            if download_segment(args.url, start, end, out_path):
                clips_to_process.append(out_path)
    else:
        clips_to_process = sorted(RAW_DIR.glob("*.mp4"))
        logger.info("Found %d local clips", len(clips_to_process))

    if not clips_to_process:
        logger.error("No clips to process.")
        sys.exit(1)

    # ── Pipeline phase ────────────────────────────────────────────────────────
    all_rows: list[dict] = []
    for clip in clips_to_process:
        rows = run_clip(clip, runner)
        all_rows.extend(rows)

    # ── Aggregation ───────────────────────────────────────────────────────────
    if all_rows:
        agg = aggregate(all_rows)
        save_aggregated(agg, OUT_DIR / "aggregated_stats.csv")
        print("\n" + "=" * 60)
        print(f"  AGGREGATED STATS ({len(clips_to_process)} clips)")
        print("=" * 60)
        fmt = "  {:>8}  {:>6}  {:>6}  {:>5}  {:>6}  {:>6}  {:>9}"
        print(fmt.format("track_id", "steals", "blocks", "defl", "cont_2", "cont_3", "clips_seen"))
        print("  " + "-" * 56)
        for row in agg[:15]:
            print(fmt.format(
                row["track_id"], row["steals"], row["blocks"],
                row["deflections"], row["contested_2pt"],
                row["contested_3pt"], row["clips_seen"],
            ))
        print("=" * 60)


if __name__ == "__main__":
    main()
