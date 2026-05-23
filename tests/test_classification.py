"""
tests/test_classification.py

Unit tests for src/classification/team_classifier.py

Covers:
  - _extract_jersey_color: output shape, normalization, edge cases
  - _get_crop: valid / clipped / degenerate bounding boxes
  - process_frame before / after fitting
  - fit(): referee separation, team assignment structure
  - flip_teams(): swaps home ↔ away correctly
  - get_assignment(): cached lookup
  - reset(): clears all state
  - get_cluster_colors_bgr(): output format

Run with:  pytest tests/test_classification.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.classification.team_classifier import (
    TEAM_AWAY, TEAM_HOME, TEAM_REFEREE, TEAM_UNKNOWN,
    TeamClassifier,
)
from src.tracking.multi_tracker import Track


# ── Helpers ───────────────────────────────────────────────────────────────────

def solid_bgr(h: int, w: int, b: int, g: int, r: int) -> np.ndarray:
    """Create a solid-color BGR image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (b, g, r)
    return img


def make_track(
    track_id: int = 1,
    x1=10, y1=10, x2=60, y2=130,
) -> Track:
    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return Track(
        track_id=track_id,
        bbox=bbox,
        center=np.array([cx, cy], dtype=np.float32),
        confidence=0.9,
        class_id=0,
        frame_idx=0,
        timestamp_sec=0.0,
    )


def make_frame_with_players(
    colors: list,          # list of (B,G,R) tuples for each player's shirt
    player_w: int = 50,
    player_h: int = 120,
    frame_w: int = 800,
    frame_h: int = 480,
):
    """
    Create a synthetic frame with colored player rectangles.
    Returns (bgr, tracks).
    """
    frame = np.full((frame_h, frame_w, 3), 80, dtype=np.uint8)  # gray bg
    tracks = []
    for i, (b, g, r) in enumerate(colors):
        x1 = 50 + i * (player_w + 40)
        y1 = 50
        x2 = x1 + player_w
        y2 = y1 + player_h
        frame[y1:y2, x1:x2] = (b, g, r)
        tracks.append(make_track(
            track_id=i + 1,
            x1=x1, y1=y1, x2=x2, y2=y2,
        ))
    return frame, tracks


# ── _extract_jersey_color ─────────────────────────────────────────────────────

class TestExtractJerseyColor:
    def test_returns_array(self):
        crop = solid_bgr(80, 40, 0, 100, 200)
        color = TeamClassifier._extract_jersey_color(crop)
        assert isinstance(color, np.ndarray)

    def test_returns_shape_3(self):
        crop = solid_bgr(80, 40, 0, 100, 200)
        color = TeamClassifier._extract_jersey_color(crop)
        assert color.shape == (3,)

    def test_normalized_range(self):
        crop = solid_bgr(80, 40, 200, 50, 30)
        color = TeamClassifier._extract_jersey_color(crop)
        assert np.all(color >= 0.0)
        assert np.all(color <= 1.0)

    def test_none_on_none_input(self):
        assert TeamClassifier._extract_jersey_color(None) is None

    def test_none_on_empty_array(self):
        assert TeamClassifier._extract_jersey_color(np.zeros((0, 0, 3), dtype=np.uint8)) is None

    def test_none_on_tiny_crop(self):
        tiny = solid_bgr(5, 5, 0, 0, 255)
        assert TeamClassifier._extract_jersey_color(tiny) is None

    def test_different_colors_give_different_vectors(self):
        red_crop   = solid_bgr(80, 40, 0,   0, 200)
        blue_crop  = solid_bgr(80, 40, 200, 0, 0)
        green_crop = solid_bgr(80, 40, 0, 200, 0)
        r = TeamClassifier._extract_jersey_color(red_crop)
        b = TeamClassifier._extract_jersey_color(blue_crop)
        g = TeamClassifier._extract_jersey_color(green_crop)
        assert not np.allclose(r, b, atol=0.05)
        assert not np.allclose(r, g, atol=0.05)

    def test_gray_gives_low_saturation(self):
        gray_crop = solid_bgr(80, 40, 150, 150, 150)
        color = TeamClassifier._extract_jersey_color(gray_crop)
        # Saturation (index 1) should be low for gray
        assert color[1] < 0.15


# ── _get_crop ─────────────────────────────────────────────────────────────────

class TestGetCrop:
    def test_valid_crop(self):
        frame = solid_bgr(480, 640, 50, 50, 50)
        bbox  = np.array([10, 10, 60, 130], dtype=np.float32)
        crop  = TeamClassifier._get_crop(frame, bbox)
        assert crop is not None
        assert crop.shape == (120, 50, 3)

    def test_clamps_to_frame_bounds(self):
        frame = solid_bgr(100, 100, 0, 0, 0)
        bbox  = np.array([-10, -10, 200, 200], dtype=np.float32)
        crop  = TeamClassifier._get_crop(frame, bbox)
        assert crop is not None
        assert crop.shape == (100, 100, 3)

    def test_returns_none_for_degenerate_box(self):
        frame = solid_bgr(480, 640, 0, 0, 0)
        bbox  = np.array([50, 50, 50, 50], dtype=np.float32)   # zero size
        assert TeamClassifier._get_crop(frame, bbox) is None

    def test_returns_none_when_fully_outside(self):
        frame = solid_bgr(100, 100, 0, 0, 0)
        bbox  = np.array([200, 200, 300, 300], dtype=np.float32)  # beyond frame
        assert TeamClassifier._get_crop(frame, bbox) is None


# ── process_frame before fitting ─────────────────────────────────────────────

class TestProcessFrameBeforeFit:
    def test_returns_empty_before_fitting(self):
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False)
        frame, tracks = make_frame_with_players([(0, 0, 200), (200, 0, 0)])
        result = clf.process_frame(tracks, frame)
        assert result == {}

    def test_accumulates_samples(self):
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False)
        frame, tracks = make_frame_with_players([(0, 0, 200)])
        for _ in range(5):
            clf.process_frame(tracks, frame)
        assert clf.n_samples_collected == 5

    def test_auto_fit_triggers(self):
        # 2 players, 5 frames each = 10 samples; min=8 → should auto-fit
        clf = TeamClassifier(min_samples_to_fit=8, auto_fit=True)
        frame, tracks = make_frame_with_players(
            [(0, 0, 200), (0, 200, 0), (0, 0, 100)]  # 3 players
        )
        for _ in range(4):
            clf.process_frame(tracks, frame)
        assert clf.is_fitted


# ── fit() ─────────────────────────────────────────────────────────────────────

class TestFit:
    def _build_fitted_classifier(self, team_a_color, team_b_color, n_frames=15):
        """Create and fit a classifier on two distinct-colored teams."""
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False, sat_thresh=40)
        frame, tracks = make_frame_with_players([team_a_color, team_b_color])
        for _ in range(n_frames):
            clf.process_frame(tracks, frame)
        success = clf.fit()
        return clf, tracks, success

    def test_fit_returns_true_with_enough_data(self):
        clf, _, success = self._build_fitted_classifier((0, 0, 200), (200, 0, 0))
        assert success is True
        assert clf.is_fitted

    def test_assignments_are_valid_team_ids(self):
        clf, tracks, _ = self._build_fitted_classifier((0, 0, 200), (200, 0, 0))
        for tid in [t.track_id for t in tracks]:
            assignment = clf.get_assignment(tid)
            assert assignment in (TEAM_HOME, TEAM_AWAY, TEAM_REFEREE)

    def test_two_distinct_players_get_different_teams(self):
        clf, tracks, _ = self._build_fitted_classifier((0, 0, 200), (0, 200, 0))
        t1 = clf.get_assignment(tracks[0].track_id)
        t2 = clf.get_assignment(tracks[1].track_id)
        # Different-colored jerseys should land in different clusters
        assert t1 != t2

    def test_returns_false_with_insufficient_tracks(self):
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False)
        # Only 1 track → can't form 2 clusters
        frame, tracks = make_frame_with_players([(0, 0, 200)])
        for _ in range(10):
            clf.process_frame(tracks, frame)
        assert clf.fit() is False

    def test_referee_detected_by_gray_jersey(self):
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False, sat_thresh=50)
        # gray, red, blue
        frame, tracks = make_frame_with_players([
            (150, 150, 150),   # gray → referee
            (0, 0, 200),       # red  → team
            (200, 0, 0),       # blue → team
        ])
        for _ in range(15):
            clf.process_frame(tracks, frame)
        clf.fit()
        gray_tid = tracks[0].track_id
        assert clf.get_assignment(gray_tid) == TEAM_REFEREE


# ── flip_teams ────────────────────────────────────────────────────────────────

class TestFlipTeams:
    def test_flip_swaps_home_and_away(self):
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False)
        frame, tracks = make_frame_with_players([(0, 0, 200), (200, 0, 0)])
        for _ in range(15):
            clf.process_frame(tracks, frame)
        clf.fit()

        t1_before = clf.get_assignment(tracks[0].track_id)
        t2_before = clf.get_assignment(tracks[1].track_id)

        clf.flip_teams()

        t1_after = clf.get_assignment(tracks[0].track_id)
        t2_after = clf.get_assignment(tracks[1].track_id)

        # Must have swapped
        if t1_before in (TEAM_HOME, TEAM_AWAY):
            assert t1_after != t1_before
        if t2_before in (TEAM_HOME, TEAM_AWAY):
            assert t2_after != t2_before

    def test_flip_leaves_referee_unchanged(self):
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False, sat_thresh=50)
        frame, tracks = make_frame_with_players([
            (150, 150, 150), (0, 0, 200), (200, 0, 0)
        ])
        for _ in range(15):
            clf.process_frame(tracks, frame)
        clf.fit()
        ref_tid = tracks[0].track_id
        clf.flip_teams()
        assert clf.get_assignment(ref_tid) == TEAM_REFEREE


# ── get_assignment ────────────────────────────────────────────────────────────

class TestGetAssignment:
    def test_unknown_before_fitting(self):
        clf = TeamClassifier()
        assert clf.get_assignment(999) == TEAM_UNKNOWN

    def test_returns_int(self):
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False)
        frame, tracks = make_frame_with_players([(0, 0, 200), (200, 0, 0)])
        for _ in range(15):
            clf.process_frame(tracks, frame)
        clf.fit()
        result = clf.get_assignment(tracks[0].track_id)
        assert isinstance(result, int)


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_fit(self):
        clf = TeamClassifier(min_samples_to_fit=5, auto_fit=True)
        frame, tracks = make_frame_with_players([(0, 0, 200), (200, 0, 0)])
        for _ in range(10):
            clf.process_frame(tracks, frame)
        assert clf.is_fitted
        clf.reset()
        assert not clf.is_fitted

    def test_reset_clears_samples(self):
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False)
        frame, tracks = make_frame_with_players([(0, 0, 200)])
        clf.process_frame(tracks, frame)
        clf.reset()
        assert clf.n_samples_collected == 0

    def test_reset_clears_assignments(self):
        clf = TeamClassifier(min_samples_to_fit=5, auto_fit=True)
        frame, tracks = make_frame_with_players([(0, 0, 200), (200, 0, 0)])
        for _ in range(10):
            clf.process_frame(tracks, frame)
        clf.reset()
        assert clf.get_assignment(tracks[0].track_id) == TEAM_UNKNOWN


# ── get_cluster_colors_bgr ────────────────────────────────────────────────────

class TestClusterColors:
    def test_returns_dict_after_fit(self):
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False)
        frame, tracks = make_frame_with_players([(0, 0, 200), (200, 0, 0)])
        for _ in range(15):
            clf.process_frame(tracks, frame)
        clf.fit()
        colors = clf.get_cluster_colors_bgr()
        assert isinstance(colors, dict)
        assert len(colors) == 2

    def test_returns_empty_before_fit(self):
        clf = TeamClassifier()
        assert clf.get_cluster_colors_bgr() == {}

    def test_color_values_are_uint8_range(self):
        clf = TeamClassifier(min_samples_to_fit=999, auto_fit=False)
        frame, tracks = make_frame_with_players([(0, 0, 200), (200, 0, 0)])
        for _ in range(15):
            clf.process_frame(tracks, frame)
        clf.fit()
        for bgr in clf.get_cluster_colors_bgr().values():
            for v in bgr:
                assert 0 <= v <= 255
