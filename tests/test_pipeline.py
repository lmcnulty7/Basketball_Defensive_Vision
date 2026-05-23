"""
tests/test_pipeline.py

Integration tests for src/pipeline/.

Tests PipelineConfig, PipelineOutput, and the PipelineRunner's
frame-processing logic using mocked trackers (no YOLO weights needed).

Run with:  pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.classification.team_classifier import TEAM_HOME, TEAM_AWAY
from src.events.event import EventType
from src.pipeline.runner import PipelineRunner
from src.pipeline.state import PipelineConfig, PipelineOutput
from src.tracking.multi_tracker import BALL_TRACK_ID, Track


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_track(tid=1, cx=320.0, cy=300.0, team_id=None) -> Track:
    t = Track(
        track_id=tid,
        bbox=np.array([cx-25, cy-60, cx+25, cy+60], dtype=np.float32),
        center=np.array([cx, cy], dtype=np.float32),
        confidence=0.85,
        class_id=0,
        frame_idx=0,
        timestamp_sec=0.0,
        team_id=team_id,
    )
    return t


def make_ball_track(px=400.0, py=300.0) -> Track:
    t = Track(
        track_id=BALL_TRACK_ID,
        bbox=np.array([px-10, py-10, px+10, py+10], dtype=np.float32),
        center=np.array([px, py], dtype=np.float32),
        confidence=0.7,
        class_id=BALL_TRACK_ID,
        frame_idx=0,
        timestamp_sec=0.0,
    )
    return t


def make_frame(h=480, w=640) -> np.ndarray:
    # Maple-wood floor color (BGR ≈ 45, 130, 210) so _is_court_visible passes.
    frame = np.full((h, w, 3), (45, 130, 210), dtype=np.uint8)
    return frame


# ── PipelineConfig ────────────────────────────────────────────────────────────

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.stride == 3
        assert cfg.device == "cpu"
        assert cfg.player_conf == pytest.approx(0.40)
        assert cfg.min_keypoints == 4

    def test_from_yaml_loads_pipeline_file(self):
        cfg = PipelineConfig.from_yaml(
            pipeline_yaml="configs/pipeline.yaml",
            models_yaml="configs/models.yaml",
        )
        assert isinstance(cfg.stride, int)
        assert cfg.stride > 0

    def test_from_yaml_falls_back_on_missing_file(self):
        cfg = PipelineConfig.from_yaml(
            pipeline_yaml="nonexistent.yaml",
            models_yaml="also_missing.yaml",
        )
        # Should return defaults without crashing
        assert cfg.stride == 3

    def test_save_annotated_video_default_false(self):
        assert PipelineConfig().save_annotated_video is False

    def test_from_yaml_overrides_device(self):
        cfg = PipelineConfig.from_yaml("configs/pipeline.yaml")
        assert isinstance(cfg.device, str)


# ── PipelineOutput ────────────────────────────────────────────────────────────

class TestPipelineOutput:
    def test_empty_output_is_valid(self):
        out = PipelineOutput()
        assert out.event_log == []
        assert out.stats_table == []
        assert out.n_frames_processed == 0

    def test_print_summary_runs_without_error(self, capsys):
        out = PipelineOutput(
            n_frames_processed=100,
            duration_sec=10.0,
            h_valid=True,
            teams_fitted=True,
            event_summary={"steal": 2, "block": 1},
            stats_table=[{
                "track_id": 5, "steals": 2, "blocks": 1,
                "deflections": 0, "contested_2pt": 3, "contested_3pt": 1,
                "def_rebounds": 2, "matchup_time_sec": 45.0,
                "avg_speed_mph": 4.2, "def_speed_mph": 5.1,
                "max_speed_mph": 12.3, "charges_drawn": 0,
                "off_rebounds": 0, "contest_pct_tight": 0.5,
            }],
        )
        out.print_summary()
        captured = capsys.readouterr()
        assert "steal" in captured.out
        assert "100" in captured.out


# ── PipelineRunner construction ───────────────────────────────────────────────

class TestPipelineRunnerInit:
    def test_constructs_with_default_config(self):
        with patch("src.pipeline.runner.MultiTracker") as mock_mt, \
             patch("src.pipeline.runner.BallDetector") as mock_bd:
            mock_mt.return_value = MagicMock()
            mock_bd.return_value = MagicMock()
            runner = PipelineRunner(PipelineConfig())
            assert runner.config.stride == 3

    def test_get_output_before_run(self):
        with patch("src.pipeline.runner.MultiTracker"), \
             patch("src.pipeline.runner.BallDetector"):
            runner = PipelineRunner(PipelineConfig())
            out = runner.get_output(duration_sec=0.0)
            assert isinstance(out, PipelineOutput)
            assert out.n_frames_processed == 0


# ── PipelineRunner.process_frame (mocked modules) ────────────────────────────

class TestPipelineRunnerProcessFrame:
    """
    Verify the frame-processing logic chains correctly when
    detection/tracking are mocked to return known tracks.
    """

    @pytest.fixture
    def runner_with_mocks(self):
        """Runner with YOLO replaced by mocks that return known tracks."""
        with patch("src.pipeline.runner.MultiTracker") as MockMT, \
             patch("src.pipeline.runner.BallDetector") as MockBD:

            # MultiTracker returns 2 tracks each frame
            mock_tracker = MagicMock()
            mock_tracker.update.return_value = [
                make_track(1, cx=300, team_id=TEAM_HOME),
                make_track(2, cx=500, team_id=TEAM_AWAY),
            ]
            MockMT.return_value = mock_tracker

            # BallDetector detect returns (detection, False)
            mock_ball_det = MagicMock()
            from src.detection.postprocess import Detection
            mock_det = Detection(
                bbox=np.array([390.0, 290.0, 410.0, 310.0], dtype=np.float32),
                confidence=0.8,
                class_id=100,
                class_name="ball",
            )
            mock_ball_det.detect.return_value = (mock_det, False)
            MockBD.return_value = mock_ball_det

            runner = PipelineRunner(PipelineConfig(min_samples_to_fit=999))
            yield runner

    def test_process_frame_returns_list(self, runner_with_mocks):
        bgr = make_frame()
        result = runner_with_mocks.process_frame(bgr, frame_idx=0, timestamp_sec=0.0)
        assert isinstance(result, list)

    def test_frame_counter_increments(self, runner_with_mocks):
        bgr = make_frame()
        runner_with_mocks.process_frame(bgr, 0, 0.0)
        runner_with_mocks.process_frame(bgr, 3, 0.1)
        assert runner_with_mocks._n_frames == 2

    def test_track_manager_receives_tracks(self, runner_with_mocks):
        bgr = make_frame()
        runner_with_mocks.process_frame(bgr, 0, 0.0)
        runner_with_mocks.process_frame(bgr, 3, 0.1)
        assert runner_with_mocks._manager.n_active() == 2

    def test_output_builds_after_frames(self, runner_with_mocks):
        bgr = make_frame()
        for i in range(5):
            runner_with_mocks.process_frame(bgr, i * 3, i * 0.1)
        out = runner_with_mocks.get_output(duration_sec=5.0)
        assert out.n_frames_processed == 5
        assert isinstance(out.event_summary, dict)

    def test_run_raises_on_missing_video(self, runner_with_mocks):
        with pytest.raises(FileNotFoundError):
            runner_with_mocks.run("nonexistent_video.mp4")

    def test_reset_clears_state(self, runner_with_mocks):
        bgr = make_frame()
        for i in range(3):
            runner_with_mocks.process_frame(bgr, i, i / 30.0)
        runner_with_mocks.reset()
        assert runner_with_mocks._n_frames == 0


# ── End-to-end: injected synthetic sequence ───────────────────────────────────

class TestPipelineEndToEnd:
    """
    Inject a synthetic defensive sequence directly into the orchestrator
    to verify the full stats aggregation chain.
    """

    def test_steal_appears_in_stats(self):
        from src.events.event import Event, EventType
        from src.stats.aggregator import StatsAggregator
        from src.stats.matchup_tracker import MatchupTracker
        from src.stats.speed_calculator import SpeedCalculator

        events = [
            Event(EventType.STEAL, 10, 1.0, primary_player_id=3,
                  secondary_player_id=7),
            Event(EventType.CONTESTED_2PT, 20, 2.0, primary_player_id=3,
                  metadata={"contest_level": "tight"}),
            Event(EventType.REBOUND_DEF, 30, 3.0, primary_player_id=4),
        ]

        agg    = StatsAggregator()
        mt     = MatchupTracker()
        sp     = SpeedCalculator()
        stats  = agg.compute(events, mt, sp, track_ids=[3, 4])
        table  = agg.to_table(stats)

        player3 = next(r for r in table if r["track_id"] == 3)
        player4 = next(r for r in table if r["track_id"] == 4)

        assert player3["steals"] == 1
        assert player3["contested_2pt"] == 1
        assert player3["contest_pct_tight"] == pytest.approx(1.0)
        assert player4["def_rebounds"] == 1
