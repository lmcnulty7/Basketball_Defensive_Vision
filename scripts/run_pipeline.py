# ── Environment flags (must be before ALL other imports) ──────────────────────
import os, sys, types
os.environ.setdefault("OMP_NUM_THREADS",           "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",           "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK",      "TRUE")
# Allow MPS to fall back to CPU for ops not yet implemented on Metal
# (torchvision::nms is the main one). Everything else runs on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ── Inject pure-scipy lap stub ─────────────────────────────────────────────────
# The official lap wheel on PyPI is x86_64-only. On Apple Silicon (arm64) the
# binary is run under Rosetta, but mixing x86_64 lap with native arm64 numpy /
# PyTorch corrupts memory and causes a segfault on first YOLO inference.
# Injecting a scipy-backed stub into sys.modules before ultralytics loads ensures
# the x86_64 binary is never imported.
import numpy as _np
from scipy.optimize import linear_sum_assignment as _lsa

def _lapjv(cost, extend_cost=False, cost_limit=float("inf"), return_cost=True):
    """scipy-backed linear assignment; same API as lap.lapjv."""
    c = cost.astype(float)
    if extend_cost:
        n = max(c.shape)
        fill = cost_limit if cost_limit < float("inf") else c.max() + 1
        padded = _np.full((n, n), fill)
        padded[:c.shape[0], :c.shape[1]] = c
        c = padded
    row_ind, col_ind = _lsa(c)
    n_rows, n_cols = cost.shape
    x = _np.full(n_rows, -1, dtype=_np.int32)
    y = _np.full(n_cols, -1, dtype=_np.int32)
    for r, c_ in zip(row_ind, col_ind):
        if r < n_rows and c_ < n_cols:
            x[r] = c_
            y[c_] = r
    opt = float(cost[x >= 0, x[x >= 0]].sum()) if return_cost else 0.0
    return opt, x, y

_lap = types.ModuleType("lap")
_lap.lapjv = _lapjv
sys.modules["lap"] = _lap   # shadows the broken x86_64 binary

"""
scripts/run_pipeline.py

CLI entry point for the Basketball Defensive Vision pipeline.

Usage
─────
    python scripts/run_pipeline.py --clip data/raw/game_clip.mp4
    python scripts/run_pipeline.py --clip game.mp4 --save-video
    python scripts/run_pipeline.py --clip game.mp4 --config configs/pipeline.yaml

Outputs
───────
  data/processed/<clip_name>_events.json   — full event log
  data/processed/<clip_name>_stats.csv     — per-player defensive stats table
  data/processed/<clip_name>_annotated.mp4 — annotated video (if --save-video)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import PipelineRunner
from src.pipeline.state import PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="NBA Defensive Vision Pipeline")
    parser.add_argument("--clip",       required=True,              help="Path to input video")
    parser.add_argument("--config",     default="configs/pipeline.yaml")
    parser.add_argument("--models",     default="configs/models.yaml")
    parser.add_argument("--out-dir",    default="data/processed")
    parser.add_argument("--save-video", action="store_true",        help="Write annotated video")
    parser.add_argument("--device",     default=None,               help="Override device (cpu/cuda/mps)")
    args = parser.parse_args()

    # Load config
    config = PipelineConfig.from_yaml(args.config, args.models)
    config.output_dir = args.out_dir
    if args.save_video:
        config.save_annotated_video = True
    if args.device:
        config.device = args.device

    # Run
    print(f"Processing: {args.clip}")
    print(f"Device: {config.device}  |  Stride: {config.stride}  |  Save video: {config.save_annotated_video}")
    runner = PipelineRunner(config)
    output = runner.run(args.clip)

    # Print summary
    output.print_summary()

    # Save outputs
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = Path(args.clip).stem

    # Event log → JSON
    events_path = out_dir / f"{stem}_events.json"
    with open(events_path, "w") as f:
        json.dump([e.to_dict() for e in output.event_log], f, indent=2)
    print(f"Events saved → {events_path}")

    # Stats → CSV
    stats_path = out_dir / f"{stem}_stats.csv"
    if output.stats_table:
        with open(stats_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output.stats_table[0].keys())
            writer.writeheader()
            writer.writerows(output.stats_table)
        print(f"Stats  saved → {stats_path}")


if __name__ == "__main__":
    main()
