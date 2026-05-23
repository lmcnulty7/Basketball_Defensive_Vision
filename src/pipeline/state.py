"""
src/pipeline/state.py

PipelineConfig — all tunable pipeline parameters in one place.
PipelineOutput — the final result returned after processing a clip.

Both are loaded from configs/pipeline.yaml + configs/models.yaml so you
never need to change code to change thresholds — only the YAML files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    All configurable pipeline parameters.

    Sensible defaults allow the pipeline to run out of the box; tweak
    individual fields or call PipelineConfig.from_yaml() to load from disk.
    """

    # ── Video ─────────────────────────────────────────────────────────────────
    stride: int     = 3        # process every Nth frame (3 → ~10fps from 30fps)
    input_size: int = 640      # YOLO input resolution
    source_fps: float = 30.0   # original video fps (for time calculations)

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "cpu"        # "cpu" | "cuda" | "mps"

    # ── Detection ─────────────────────────────────────────────────────────────
    player_weights: str  = "yolov8m.pt"
    ball_weights:   str  = "models/checkpoints/ball_yolo.pt"
    player_conf:    float = 0.40
    ball_conf:      float = 0.35
    iou_threshold:  float = 0.45

    # ── TrackNet ball detector (takes priority over ball_weights when set) ─────
    tracknet_weights: str   = "models/checkpoints/tracknet_best.pt"
    tracknet_seq_len: int   = 8
    tracknet_conf:    float = 0.50

    # ── Tracking ──────────────────────────────────────────────────────────────
    tracker_config: str = "botsort.yaml"

    # ── Court ─────────────────────────────────────────────────────────────────
    court_kp_weights: str = "models/checkpoints/court_kp_yolov8n.pt"
    min_keypoints:    int = 4   # minimum detected KPs to compute H

    # ── Team classification ───────────────────────────────────────────────────
    min_samples_to_fit: int = 30   # frames of jersey samples before KMeans fits
    sat_thresh:         int = 50   # HSV-S threshold to flag referees

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir:           str  = "data/processed"
    save_annotated_video: bool = False
    annotated_video_fps:  int  = 10

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    @classmethod
    def from_yaml(
        cls,
        pipeline_yaml: str = "configs/pipeline.yaml",
        models_yaml:   str = "configs/models.yaml",
    ) -> "PipelineConfig":
        """Load config from the project YAML files."""
        import yaml

        cfg = cls()

        p_path = Path(pipeline_yaml)
        if p_path.exists():
            with open(p_path) as f:
                p = yaml.safe_load(f) or {}
            cfg.stride      = p.get("video", {}).get("stride",     cfg.stride)
            cfg.input_size  = p.get("video", {}).get("input_size", cfg.input_size)
            cfg.device      = p.get("device",                      cfg.device)
            cfg.log_level   = p.get("logging", {}).get("level",    cfg.log_level)
            out = p.get("output", {})
            cfg.save_annotated_video = out.get("save_annotated_video", cfg.save_annotated_video)
            cfg.annotated_video_fps  = out.get("annotated_video_fps",  cfg.annotated_video_fps)
        else:
            logger.warning("pipeline.yaml not found at %s — using defaults", p_path)

        m_path = Path(models_yaml)
        if m_path.exists():
            with open(m_path) as f:
                m = yaml.safe_load(f) or {}
            det = m.get("detection", {})
            cfg.player_weights = det.get("player_weights", cfg.player_weights)
            cfg.ball_weights   = det.get("ball_weights",   cfg.ball_weights)
            cfg.player_conf    = det.get("player_conf",    cfg.player_conf)
            cfg.ball_conf      = det.get("ball_conf",      cfg.ball_conf)
            cfg.iou_threshold  = det.get("iou_threshold",  cfg.iou_threshold)
            tn = m.get("tracknet", {})
            cfg.tracknet_weights = tn.get("weights",  cfg.tracknet_weights)
            cfg.tracknet_seq_len = tn.get("seq_len",  cfg.tracknet_seq_len)
            cfg.tracknet_conf    = tn.get("conf",     cfg.tracknet_conf)
        else:
            logger.warning("models.yaml not found at %s — using defaults", m_path)

        logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO))
        return cfg


@dataclass
class PipelineOutput:
    """
    Everything produced by a single PipelineRunner.run() call.

    Attributes
    ──────────
    event_log           : All defensive events fired during the clip.
    stats_table         : List of per-player stat dicts, sorted by impact.
    event_summary       : {event_type_name: total_count} across all players.
    n_frames_processed  : Number of frames that went through the pipeline.
    duration_sec        : Wall-clock seconds of video processed.
    h_valid             : Whether a valid homography was computed.
    teams_fitted        : Whether team classification converged.
    """

    event_log:          List         = field(default_factory=list)
    stats_table:        List[dict]   = field(default_factory=list)
    event_summary:      Dict[str, int] = field(default_factory=dict)
    n_frames_processed: int          = 0
    duration_sec:       float        = 0.0
    h_valid:            bool         = False
    teams_fitted:       bool         = False

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print(f"\n{'═'*60}")
        print(f"  Pipeline Output Summary")
        print(f"{'═'*60}")
        print(f"  Frames processed : {self.n_frames_processed}")
        print(f"  Video duration   : {self.duration_sec:.1f}s")
        print(f"  Homography valid : {self.h_valid}")
        print(f"  Teams fitted     : {self.teams_fitted}")
        print(f"  Total events     : {sum(self.event_summary.values())}")
        if self.event_summary:
            print(f"\n  Event breakdown:")
            for etype, count in sorted(self.event_summary.items()):
                print(f"    {etype:<22} {count}")
        if self.stats_table:
            print(f"\n  Per-player defensive stats:")
            header = ("track_id", "steals", "blocks", "defl",
                      "cont_2", "cont_3", "def_reb", "matchup_s", "def_mph")
            fmt = "  {:>8}  {:>6}  {:>6}  {:>5}  {:>6}  {:>6}  {:>7}  {:>9}  {:>7}"
            print(fmt.format(*header))
            print("  " + "-" * 72)
            for row in self.stats_table:
                print(fmt.format(
                    row["track_id"],
                    row["steals"],
                    row["blocks"],
                    row["deflections"],
                    row["contested_2pt"],
                    row["contested_3pt"],
                    row["def_rebounds"],
                    f"{row['matchup_time_sec']:.1f}",
                    f"{row['def_speed_mph']:.2f}",
                ))
        print(f"{'═'*60}\n")
