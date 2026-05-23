"""
tests/test_detection.py

Unit tests for src/detection/.

Covers:
  - Detection / DetectionResult dataclasses
  - filter_by_confidence, filter_by_area, compute_iou
  - KalmanBallFilter state machine
  - BallDetector._is_plausible (via monkeypatching the YOLO import)
  - PlayerDetector._parse_results (static method — no model needed)

Tests that require actual model weights are skipped unless the weights exist.
Run with:  pytest tests/test_detection.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.detection.postprocess import (
    BALL_CLASS_ID,
    Detection,
    DetectionResult,
    compute_iou,
    filter_by_area,
    filter_by_confidence,
)
from src.detection.ball_detector import KalmanBallFilter, BallDetector
from src.detection.player_detector import PlayerDetector


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_detection(
    x1=10.0, y1=20.0, x2=50.0, y2=120.0,
    conf=0.8,
    cls_id=0,
    cls_name="person",
) -> Detection:
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=conf,
        class_id=cls_id,
        class_name=cls_name,
    )


def make_frame_bgr(h=1080, w=1920) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


# ── Detection dataclass ───────────────────────────────────────────────────────

class TestDetection:
    def test_center(self):
        d = make_detection(x1=0, y1=0, x2=100, y2=200)
        center = d.center
        assert center[0] == pytest.approx(50.0)
        assert center[1] == pytest.approx(100.0)

    def test_width_height(self):
        d = make_detection(x1=10, y1=20, x2=50, y2=120)
        assert d.width  == pytest.approx(40.0)
        assert d.height == pytest.approx(100.0)

    def test_area(self):
        d = make_detection(x1=0, y1=0, x2=40, y2=100)
        assert d.area == pytest.approx(4000.0)

    def test_to_dict_keys(self):
        d = make_detection()
        out = d.to_dict()
        assert set(out.keys()) == {"bbox", "confidence", "class_id", "class_name", "track_id"}

    def test_to_dict_bbox_is_list(self):
        d = make_detection()
        assert isinstance(d.to_dict()["bbox"], list)

    def test_track_id_defaults_none(self):
        d = make_detection()
        assert d.track_id is None

    def test_track_id_assignable(self):
        d = make_detection()
        d.track_id = 7
        assert d.track_id == 7


# ── DetectionResult ───────────────────────────────────────────────────────────

class TestDetectionResult:
    def test_to_dict_structure(self):
        bgr = make_frame_bgr()
        result = DetectionResult(
            frame_idx=10,
            timestamp_sec=0.333,
            players=[make_detection()],
            ball=make_detection(cls_id=BALL_CLASS_ID, cls_name="ball"),
            ball_is_predicted=False,
            raw_bgr=bgr,
        )
        d = result.to_dict()
        assert d["frame_idx"] == 10
        assert len(d["players"]) == 1
        assert d["ball"] is not None
        assert d["ball_is_predicted"] is False

    def test_to_dict_no_ball(self):
        result = DetectionResult(
            frame_idx=0,
            timestamp_sec=0.0,
            players=[],
            ball=None,
            ball_is_predicted=False,
            raw_bgr=make_frame_bgr(),
        )
        assert result.to_dict()["ball"] is None


# ── filter_by_confidence ──────────────────────────────────────────────────────

class TestFilterByConfidence:
    def test_keeps_above_threshold(self):
        dets = [make_detection(conf=0.9), make_detection(conf=0.3)]
        kept = filter_by_confidence(dets, min_conf=0.5)
        assert len(kept) == 1
        assert kept[0].confidence == pytest.approx(0.9)

    def test_keeps_at_threshold(self):
        dets = [make_detection(conf=0.5)]
        assert len(filter_by_confidence(dets, 0.5)) == 1

    def test_empty_input(self):
        assert filter_by_confidence([], 0.5) == []

    def test_all_filtered(self):
        dets = [make_detection(conf=0.1), make_detection(conf=0.2)]
        assert filter_by_confidence(dets, 0.5) == []


# ── filter_by_area ────────────────────────────────────────────────────────────

class TestFilterByArea:
    def test_removes_tiny(self):
        tiny = make_detection(x1=0, y1=0, x2=5, y2=5)   # area = 25
        large = make_detection(x1=0, y1=0, x2=100, y2=200)
        kept = filter_by_area([tiny, large], min_area=100.0)
        assert len(kept) == 1
        assert kept[0] is large

    def test_removes_huge(self):
        huge = make_detection(x1=0, y1=0, x2=5000, y2=5000)
        small = make_detection(x1=0, y1=0, x2=50, y2=100)
        kept = filter_by_area([huge, small], max_area=10000.0)
        assert len(kept) == 1
        assert kept[0] is small


# ── compute_iou ───────────────────────────────────────────────────────────────

class TestComputeIoU:
    def test_identical_boxes_iou_one(self):
        box = np.array([0, 0, 100, 100], dtype=float)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_iou_zero(self):
        a = np.array([0,   0,  50,  50], dtype=float)
        b = np.array([100, 100, 200, 200], dtype=float)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([0, 0, 100, 100], dtype=float)
        b = np.array([50, 50, 150, 150], dtype=float)
        iou = compute_iou(a, b)
        # intersection = 50*50=2500, union = 2*10000-2500=17500
        assert iou == pytest.approx(2500 / 17500, rel=1e-5)

    def test_touching_edges_iou_zero(self):
        a = np.array([0, 0, 50, 50], dtype=float)
        b = np.array([50, 0, 100, 50], dtype=float)
        assert compute_iou(a, b) == pytest.approx(0.0)


# ── KalmanBallFilter ──────────────────────────────────────────────────────────

class TestKalmanBallFilter:
    def test_not_initialized_predict_returns_none(self):
        kf = KalmanBallFilter()
        assert kf.predict() is None

    def test_position_none_before_init(self):
        assert KalmanBallFilter().position is None

    def test_initialize_sets_state(self):
        kf = KalmanBallFilter()
        kf.initialize(100.0, 200.0)
        assert kf.initialized is True
        pos = kf.position
        assert pos[0] == pytest.approx(100.0)
        assert pos[1] == pytest.approx(200.0)

    def test_update_initializes_if_first_call(self):
        kf = KalmanBallFilter()
        state = kf.update(50.0, 75.0)
        assert kf.initialized
        assert state[0] == pytest.approx(50.0)
        assert state[1] == pytest.approx(75.0)

    def test_update_resets_frames_since_update(self):
        kf = KalmanBallFilter()
        kf.update(0.0, 0.0)
        kf.predict()
        kf.predict()
        assert kf.frames_since_update == 2
        kf.update(10.0, 10.0)
        assert kf.frames_since_update == 0

    def test_predict_increments_frames_since_update(self):
        kf = KalmanBallFilter()
        kf.initialize(0.0, 0.0)
        for i in range(5):
            kf.predict()
        assert kf.frames_since_update == 5

    def test_constant_velocity_extrapolation(self):
        """With constant observations, predicted position should follow trajectory."""
        kf = KalmanBallFilter()
        # Feed observations moving right at 10px/frame
        for i in range(20):
            kf.update(float(i * 10), 0.0)
        # After convergence, one more predict should be close to i*10 + 10
        state = kf.predict()
        assert state[0] == pytest.approx(200.0, abs=15.0)  # ~200px ± some filter lag

    def test_reset_clears_state(self):
        kf = KalmanBallFilter()
        kf.initialize(100.0, 100.0)
        kf.reset()
        assert not kf.initialized
        assert kf.position is None
        assert kf.frames_since_update == 0

    def test_state_vector_shape(self):
        kf = KalmanBallFilter()
        state = kf.update(50.0, 50.0)
        assert state.shape == (4,)  # [cx, cy, vx, vy]


# ── BallDetector._is_plausible ────────────────────────────────────────────────

class TestBallDetectorIsPlausible:
    """
    Test the size/shape filter without needing YOLO weights.
    We reach the static method through a mocked BallDetector instance.
    """

    @pytest.fixture
    def detector(self):
        # Construct BallDetector without loading any YOLO model
        with patch("src.detection.ball_detector.BallDetector.__init__", return_value=None):
            d = BallDetector.__new__(BallDetector)
            d._MIN_DIM_PX    = BallDetector._MIN_DIM_PX
            d._MAX_FRAME_PCT = BallDetector._MAX_FRAME_PCT
            d._MIN_ASPECT    = BallDetector._MIN_ASPECT
            return d

    def test_normal_ball_accepted(self, detector):
        ball = make_detection(x1=900, y1=400, x2=930, y2=430)  # 30×30 px
        assert detector._is_plausible(ball, 1920, 1080)

    def test_too_small_rejected(self, detector):
        tiny = make_detection(x1=0, y1=0, x2=5, y2=5)   # 5px — below _MIN_DIM_PX
        assert not detector._is_plausible(tiny, 1920, 1080)

    def test_too_large_rejected(self, detector):
        # 400px wide on a 1920 frame = 20% > 15% threshold
        huge = make_detection(x1=0, y1=0, x2=400, y2=400)
        assert not detector._is_plausible(huge, 1920, 1080)

    def test_elongated_rejected(self, detector):
        # 10×80 px — aspect ratio 0.125 < 0.4 threshold
        elongated = make_detection(x1=0, y1=0, x2=10, y2=80)
        assert not detector._is_plausible(elongated, 1920, 1080)


# ── PlayerDetector._parse_results (static method) ────────────────────────────

class TestPlayerDetectorParseResults:
    def _make_mock_result(self, bboxes, confs, classes):
        """Build a mock ultralytics result object."""
        r = MagicMock()
        if bboxes:
            r.boxes.xyxy.cpu().numpy.return_value = np.array(bboxes, dtype=np.float32)
            r.boxes.conf.cpu().numpy.return_value = np.array(confs, dtype=np.float32)
            r.boxes.cls.cpu().numpy.return_value  = np.array(classes, dtype=np.float32)
            r.boxes.__len__ = lambda self: len(bboxes)
        else:
            r.boxes = None
        return r

    def test_parses_single_person(self):
        bbox = [[10, 20, 50, 120]]
        mock_r = self._make_mock_result(bbox, [0.85], [0])
        dets = PlayerDetector._parse_results([mock_r])
        assert len(dets) == 1
        assert dets[0].class_name == "person"
        assert dets[0].confidence == pytest.approx(0.85, abs=1e-4)

    def test_sorted_by_confidence_descending(self):
        bboxes = [[0, 0, 50, 50], [100, 100, 150, 200]]
        mock_r = self._make_mock_result(bboxes, [0.5, 0.9], [0, 0])
        dets = PlayerDetector._parse_results([mock_r])
        assert dets[0].confidence > dets[1].confidence

    def test_empty_result_returns_empty_list(self):
        mock_r = self._make_mock_result([], [], [])
        dets = PlayerDetector._parse_results([mock_r])
        assert dets == []

    def test_bbox_dtype_float32(self):
        bbox = [[10, 20, 50, 120]]
        mock_r = self._make_mock_result(bbox, [0.7], [0])
        dets = PlayerDetector._parse_results([mock_r])
        assert dets[0].bbox.dtype == np.float32
