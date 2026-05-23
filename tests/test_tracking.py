"""
tests/test_tracking.py

Unit tests for src/tracking/.

Covers:
  - Track / TrackingResult dataclasses
  - TrackState dataclass
  - TrackManager: update, get_history, get_velocity, frames_active, camera cut detection
  - BallTracker._infer_possessor (isolated, no model needed)
  - MultiTracker._parse_results (static method, mocked ultralytics output)

Run with:  pytest tests/test_tracking.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from collections import deque
from unittest.mock import MagicMock, patch

from src.tracking.multi_tracker import BALL_TRACK_ID, Track, TrackingResult, MultiTracker
from src.tracking.ball_tracker import BallTracker, POSSESSION_DIST_PX
from src.tracking.track_manager import TrackState, TrackManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_track(
    track_id: int = 1,
    cx: float = 100.0,
    cy: float = 200.0,
    w: float = 60.0,
    h: float = 120.0,
    conf: float = 0.85,
    frame_idx: int = 0,
    timestamp_sec: float = 0.0,
    team_id: int = None,
) -> Track:
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return Track(
        track_id=track_id,
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        center=np.array([cx, cy], dtype=np.float32),
        confidence=conf,
        class_id=0,
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        team_id=team_id,
    )


def make_ball_track(cx=300.0, cy=400.0, frame_idx=0) -> Track:
    half = 15.0
    return Track(
        track_id=BALL_TRACK_ID,
        bbox=np.array([cx-half, cy-half, cx+half, cy+half], dtype=np.float32),
        center=np.array([cx, cy], dtype=np.float32),
        confidence=0.7,
        class_id=BALL_TRACK_ID,
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
    )


def make_frame_bgr(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── Track dataclass ───────────────────────────────────────────────────────────

class TestTrack:
    def test_width(self):
        t = make_track(cx=100, w=60)
        assert t.width == pytest.approx(60.0)

    def test_height(self):
        t = make_track(cy=200, h=120)
        assert t.height == pytest.approx(120.0)

    def test_to_dict_keys(self):
        t = make_track()
        d = t.to_dict()
        expected = {"track_id", "bbox", "center", "confidence", "class_id",
                    "frame_idx", "timestamp_sec", "team_id", "court_pos"}
        assert set(d.keys()) == expected

    def test_to_dict_bbox_is_list(self):
        assert isinstance(make_track().to_dict()["bbox"], list)

    def test_to_dict_center_is_list(self):
        assert isinstance(make_track().to_dict()["center"], list)

    def test_court_pos_none_by_default(self):
        assert make_track().court_pos is None

    def test_team_id_settable(self):
        t = make_track()
        t.team_id = 1
        assert t.team_id == 1

    def test_bbox_dtype_float32(self):
        assert make_track().bbox.dtype == np.float32

    def test_center_dtype_float32(self):
        assert make_track().center.dtype == np.float32


# ── TrackingResult ────────────────────────────────────────────────────────────

class TestTrackingResult:
    def test_to_dict_structure(self):
        result = TrackingResult(
            frame_idx=5,
            timestamp_sec=0.5,
            player_tracks=[make_track(1), make_track(2)],
            ball_track=make_ball_track(),
            ball_is_predicted=False,
            ball_possessor_id=1,
            raw_bgr=make_frame_bgr(),
        )
        d = result.to_dict()
        assert d["frame_idx"] == 5
        assert len(d["player_tracks"]) == 2
        assert d["ball_possessor_id"] == 1
        assert d["ball_is_predicted"] is False

    def test_to_dict_no_ball(self):
        result = TrackingResult(
            frame_idx=0,
            timestamp_sec=0.0,
            player_tracks=[],
            ball_track=None,
            ball_is_predicted=False,
            ball_possessor_id=None,
            raw_bgr=make_frame_bgr(),
        )
        assert result.to_dict()["ball_track"] is None
        assert result.to_dict()["ball_possessor_id"] is None


# ── TrackState ────────────────────────────────────────────────────────────────

class TestTrackState:
    def test_construction(self):
        ts = TrackState(
            track_id=3,
            center=np.array([50.0, 100.0]),
            bbox=np.array([20.0, 40.0, 80.0, 160.0]),
            frame_idx=10,
            timestamp_sec=1.0,
        )
        assert ts.track_id == 3
        assert ts.team_id is None
        assert ts.court_pos is None

    def test_court_pos_assignable(self):
        ts = TrackState(
            track_id=1,
            center=np.array([50.0, 50.0]),
            bbox=np.array([0.0, 0.0, 100.0, 100.0]),
            frame_idx=0,
            timestamp_sec=0.0,
        )
        ts.court_pos = np.array([20.0, 10.0])
        assert ts.court_pos[0] == pytest.approx(20.0)


# ── TrackManager ──────────────────────────────────────────────────────────────

class TestTrackManager:
    def test_n_active_after_update(self):
        mgr = TrackManager()
        tracks = [make_track(1), make_track(2), make_track(3)]
        mgr.update(tracks, None, frame_idx=0, timestamp_sec=0.0)
        assert mgr.n_active() == 3

    def test_is_active_true(self):
        mgr = TrackManager()
        mgr.update([make_track(7)], None, frame_idx=0, timestamp_sec=0.0)
        assert mgr.is_active(7)

    def test_is_active_false_for_unseen(self):
        mgr = TrackManager()
        mgr.update([make_track(1)], None, frame_idx=0, timestamp_sec=0.0)
        assert not mgr.is_active(999)

    def test_track_disappears_from_active(self):
        mgr = TrackManager()
        mgr.update([make_track(1), make_track(2)], None, frame_idx=0, timestamp_sec=0.0)
        mgr.update([make_track(1)], None, frame_idx=1, timestamp_sec=0.033)
        assert mgr.is_active(1)
        assert not mgr.is_active(2)
        assert mgr.n_active() == 1

    def test_history_accumulates(self):
        mgr = TrackManager()
        for i in range(5):
            mgr.update([make_track(1, frame_idx=i)], None, frame_idx=i, timestamp_sec=i/30.0)
        assert len(mgr.get_history(1)) == 5

    def test_history_capped_at_history_len(self):
        mgr = TrackManager(history_len=5)
        for i in range(10):
            mgr.update([make_track(1, frame_idx=i)], None, frame_idx=i, timestamp_sec=i/30.0)
        assert len(mgr.get_history(1)) == 5

    def test_history_empty_for_unknown_track(self):
        mgr = TrackManager()
        assert mgr.get_history(999) == []

    def test_history_n_slices_last_n(self):
        mgr = TrackManager()
        for i in range(10):
            mgr.update([make_track(1, frame_idx=i)], None, frame_idx=i, timestamp_sec=i/30.0)
        hist = mgr.get_history(1, n=3)
        assert len(hist) == 3
        assert hist[-1].frame_idx == 9

    def test_ball_track_stored_in_history(self):
        mgr = TrackManager()
        ball = make_ball_track(frame_idx=0)
        mgr.update([], ball, frame_idx=0, timestamp_sec=0.0)
        assert len(mgr.get_history(BALL_TRACK_ID)) == 1

    def test_frames_active_single_frame(self):
        mgr = TrackManager()
        mgr.update([make_track(5, frame_idx=3)], None, frame_idx=3, timestamp_sec=0.1)
        assert mgr.frames_active(5) == 1

    def test_frames_active_across_multiple(self):
        mgr = TrackManager()
        for i in range(10, 20):
            mgr.update([make_track(5, frame_idx=i)], None, frame_idx=i, timestamp_sec=i/30.0)
        # born at frame 10, last_seen at frame 19 → 10 frames
        assert mgr.frames_active(5) == 10

    def test_frames_active_zero_for_unknown(self):
        assert TrackManager().frames_active(999) == 0

    def test_velocity_none_with_one_history_entry(self):
        mgr = TrackManager()
        mgr.update([make_track(1, cx=100, frame_idx=0)], None, frame_idx=0, timestamp_sec=0.0)
        assert mgr.get_velocity(1) is None

    def test_velocity_linear_motion(self):
        """Player moving right at 10px/frame → vx ≈ 10, vy ≈ 0."""
        mgr = TrackManager()
        for i in range(10):
            mgr.update(
                [make_track(1, cx=float(i * 10), cy=200.0, frame_idx=i)],
                None,
                frame_idx=i,
                timestamp_sec=i / 30.0,
            )
        vel = mgr.get_velocity(1, n_frames=10)
        assert vel is not None
        assert vel[0] == pytest.approx(10.0, abs=1.0)  # vx ≈ 10 px/frame
        assert vel[1] == pytest.approx(0.0, abs=1.0)   # vy ≈ 0

    def test_velocity_returns_array(self):
        mgr = TrackManager()
        for i in range(3):
            mgr.update([make_track(1, cx=float(i * 5), frame_idx=i)], None, frame_idx=i, timestamp_sec=0.0)
        vel = mgr.get_velocity(1)
        assert isinstance(vel, np.ndarray)
        assert vel.shape == (2,)

    def test_update_court_pos(self):
        mgr = TrackManager()
        mgr.update([make_track(1, frame_idx=0)], None, frame_idx=0, timestamp_sec=0.0)
        mgr.update_court_pos(1, np.array([25.0, 10.0]))
        hist = mgr.get_history(1)
        assert hist[-1].court_pos[0] == pytest.approx(25.0)
        assert mgr.get_active_tracks()[1].court_pos[0] == pytest.approx(25.0)

    def test_update_team_id(self):
        mgr = TrackManager()
        mgr.update([make_track(2, frame_idx=0)], None, frame_idx=0, timestamp_sec=0.0)
        mgr.update_team_id(2, team_id=0)
        assert mgr.get_history(2)[-1].team_id == 0
        assert mgr.get_active_tracks()[2].team_id == 0

    def test_camera_cut_detected(self):
        """When ≥80% of tracks vanish simultaneously, should return True."""
        mgr = TrackManager(camera_cut_threshold=0.80)
        # 5 tracks active
        mgr.update(
            [make_track(i, frame_idx=0) for i in range(1, 6)],
            None, frame_idx=0, timestamp_sec=0.0,
        )
        # 4 of 5 vanish → 80% → camera cut
        cut = mgr.update(
            [make_track(1, frame_idx=1)],
            None, frame_idx=1, timestamp_sec=0.033,
        )
        assert cut is True

    def test_no_camera_cut_below_threshold(self):
        mgr = TrackManager(camera_cut_threshold=0.80)
        mgr.update(
            [make_track(i, frame_idx=0) for i in range(1, 5)],
            None, frame_idx=0, timestamp_sec=0.0,
        )
        # 2 of 4 vanish → 50% → not a cut
        cut = mgr.update(
            [make_track(1, frame_idx=1), make_track(2, frame_idx=1)],
            None, frame_idx=1, timestamp_sec=0.033,
        )
        assert cut is False

    def test_get_active_tracks_returns_copy(self):
        mgr = TrackManager()
        mgr.update([make_track(1)], None, frame_idx=0, timestamp_sec=0.0)
        active = mgr.get_active_tracks()
        active[999] = make_track(999)   # modify the copy
        assert 999 not in mgr.get_active_tracks()


# ── BallTracker._infer_possessor ──────────────────────────────────────────────

class TestBallTrackerInferPossessor:
    """
    Tests for the possession inference logic.
    No BallDetector or YOLO model needed — we call the static method directly.
    """

    def test_closest_player_within_threshold_gets_possession(self):
        ball = make_ball_track(cx=300.0, cy=400.0)
        # Player at (310, 410) — distance = sqrt(100+100) ≈ 14px < 45
        near = make_track(1, cx=310.0, cy=410.0)
        # Player at (500, 400) — distance = 200px > 45
        far  = make_track(2, cx=500.0, cy=400.0)
        result = BallTracker._infer_possessor(ball, [near, far])
        assert result == 1

    def test_no_player_within_threshold_returns_none(self):
        ball = make_ball_track(cx=100.0, cy=100.0)
        far1 = make_track(1, cx=300.0, cy=300.0)   # ~283px away
        far2 = make_track(2, cx=400.0, cy=100.0)   # 300px away
        assert BallTracker._infer_possessor(ball, [far1, far2]) is None

    def test_empty_player_list_returns_none(self):
        ball = make_ball_track()
        assert BallTracker._infer_possessor(ball, []) is None

    def test_exactly_at_threshold_gets_possession(self):
        ball = make_ball_track(cx=0.0, cy=0.0)
        # Distance exactly POSSESSION_DIST_PX
        player = make_track(1, cx=float(POSSESSION_DIST_PX), cy=0.0)
        assert BallTracker._infer_possessor(ball, [player]) == 1

    def test_just_outside_threshold_returns_none(self):
        ball = make_ball_track(cx=0.0, cy=0.0)
        player = make_track(1, cx=float(POSSESSION_DIST_PX + 1), cy=0.0)
        assert BallTracker._infer_possessor(ball, [player]) is None

    def test_returns_closest_when_multiple_within_threshold(self):
        ball = make_ball_track(cx=200.0, cy=200.0)
        closer = make_track(1, cx=210.0, cy=200.0)  # 10px
        farther = make_track(2, cx=230.0, cy=200.0) # 30px  (both < 45)
        assert BallTracker._infer_possessor(ball, [closer, farther]) == 1


# ── MultiTracker._parse_results (static method) ──────────────────────────────

class TestMultiTrackerParseResults:
    def _make_mock_result(self, bboxes, ids, confs, classes):
        r = MagicMock()
        if bboxes and ids:
            r.boxes.xyxy.cpu().numpy.return_value = np.array(bboxes, dtype=np.float32)
            r.boxes.id.cpu().numpy.return_value   = np.array(ids, dtype=np.float32)
            r.boxes.conf.cpu().numpy.return_value = np.array(confs, dtype=np.float32)
            r.boxes.cls.cpu().numpy.return_value  = np.array(classes, dtype=np.float32)
        else:
            r.boxes = None
        return r

    def test_parses_track_ids(self):
        bboxes = [[10, 20, 50, 120], [200, 100, 280, 250]]
        mock_r = self._make_mock_result(bboxes, [3, 7], [0.9, 0.8], [0, 0])
        tracks = MultiTracker._parse_results([mock_r], frame_idx=5, timestamp_sec=0.5)
        ids = [t.track_id for t in tracks]
        assert 3 in ids and 7 in ids

    def test_sorted_by_track_id(self):
        bboxes = [[200, 100, 280, 250], [10, 20, 50, 120]]
        mock_r = self._make_mock_result(bboxes, [7, 3], [0.8, 0.9], [0, 0])
        tracks = MultiTracker._parse_results([mock_r], frame_idx=0, timestamp_sec=0.0)
        assert tracks[0].track_id < tracks[1].track_id

    def test_center_computed_correctly(self):
        bboxes = [[0.0, 0.0, 100.0, 200.0]]
        mock_r = self._make_mock_result(bboxes, [1], [0.9], [0])
        tracks = MultiTracker._parse_results([mock_r], frame_idx=0, timestamp_sec=0.0)
        assert tracks[0].center[0] == pytest.approx(50.0)
        assert tracks[0].center[1] == pytest.approx(100.0)

    def test_frame_idx_stamped(self):
        bboxes = [[0.0, 0.0, 50.0, 100.0]]
        mock_r = self._make_mock_result(bboxes, [2], [0.7], [0])
        tracks = MultiTracker._parse_results([mock_r], frame_idx=42, timestamp_sec=1.4)
        assert tracks[0].frame_idx == 42
        assert tracks[0].timestamp_sec == pytest.approx(1.4)

    def test_none_boxes_returns_empty(self):
        mock_r = self._make_mock_result([], [], [], [])
        tracks = MultiTracker._parse_results([mock_r], frame_idx=0, timestamp_sec=0.0)
        assert tracks == []

    def test_none_id_skips_result(self):
        r = MagicMock()
        r.boxes.id = None
        tracks = MultiTracker._parse_results([r], frame_idx=0, timestamp_sec=0.0)
        assert tracks == []
