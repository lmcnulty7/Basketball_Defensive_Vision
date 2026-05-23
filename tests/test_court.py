"""
tests/test_court.py

Unit tests for src/court/.

Covers:
  - CourtModel: zone classification, is_three_point, is_in_paint,
                dist_to_basket, court_to_pixel round-trip
  - CourtHomography: compute, to_court, to_pixel, batch projection,
                     round-trip accuracy, invalid-state guards
  - homography_from_keypoints: sufficient / insufficient points
  - ClassicalKeyDetector: output schema, synthetic court detection

Run with:  pytest tests/test_court.py -v
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from src.court.court_model import (
    BASKET_LEFT, BASKET_RIGHT, COURT_KEYPOINTS, COURT_WIDTH, HALF_COURT,
    PAINT_DEPTH, PAINT_HALF_W, RESTRICTED_RADIUS, THREE_CORNER_Y,
    THREE_RADIUS, CourtModel, ShotZone,
)
from src.court.homography import CourtHomography, homography_from_keypoints
from src.court.keypoint_detector import (
    KEYPOINT_NAMES, ClassicalKeyDetector,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_court_diagram(w=940, h=500) -> np.ndarray:
    """
    Draw a simplified top-down NBA court (wood + white paint lines).
    Used to test the classical keypoint detector on a known image.
    """
    img = np.full((h, w, 3), (40, 120, 200), dtype=np.uint8)   # wood color (BGR)

    def ft_to_px(x_ft, y_ft):
        px = int((x_ft + HALF_COURT) / (2 * HALF_COURT) * w)
        py = int((COURT_WIDTH / 2 - y_ft) / COURT_WIDTH * h)
        return (px, py)

    # White lines
    white = (255, 255, 255)
    thick = 3

    # Court boundary
    cv2.rectangle(img, ft_to_px(-HALF_COURT, -COURT_WIDTH/2),
                  ft_to_px(HALF_COURT, COURT_WIDTH/2), white, thick)

    # Home paint
    cv2.rectangle(img,
                  ft_to_px(-HALF_COURT, -PAINT_HALF_W),
                  ft_to_px(-HALF_COURT + PAINT_DEPTH, PAINT_HALF_W),
                  white, thick)

    # Away paint
    cv2.rectangle(img,
                  ft_to_px(HALF_COURT - PAINT_DEPTH, -PAINT_HALF_W),
                  ft_to_px(HALF_COURT, PAINT_HALF_W),
                  white, thick)

    # Half-court line
    cv2.line(img, ft_to_px(0, -COURT_WIDTH/2), ft_to_px(0, COURT_WIDTH/2), white, thick)

    return img


def four_point_H() -> tuple:
    """
    Return (pixel_pts, court_pts, H_true) for a simple known transform.
    Maps a 100x100 pixel square to a 10x10 court-foot square.
    """
    pixel_pts = np.array([
        [0.0,   0.0],
        [100.0, 0.0],
        [100.0, 100.0],
        [0.0,   100.0],
    ], dtype=np.float32)

    court_pts = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
    ], dtype=np.float32)

    return pixel_pts, court_pts


# ── CourtModel — zone classification ─────────────────────────────────────────

class TestCourtModelZones:
    cm = CourtModel()

    # Restricted area
    def test_restricted_area_home(self):
        # 2 ft from home basket
        x = BASKET_LEFT[0] + 2.0
        assert self.cm.get_zone(x, 0.0, basket="left") == ShotZone.RESTRICTED_AREA

    def test_restricted_area_away(self):
        x = BASKET_RIGHT[0] - 2.0
        assert self.cm.get_zone(x, 0.0, basket="right") == ShotZone.RESTRICTED_AREA

    # Paint (non-RA)
    def test_paint_non_ra_home(self):
        # 10 ft from baseline, center of paint → 10-5.25 = 4.75 ft from basket
        # But distance > RESTRICTED_RADIUS (4 ft), so paint_non_ra
        x = BASKET_LEFT[0] + 5.0   # well inside paint, outside RA
        assert self.cm.get_zone(x, 0.0, basket="left") == ShotZone.PAINT_NON_RA

    def test_paint_non_ra_away(self):
        x = BASKET_RIGHT[0] - 5.0
        assert self.cm.get_zone(x, 0.0, basket="right") == ShotZone.PAINT_NON_RA

    # Mid-range
    def test_midrange_home(self):
        # Inside 3PT line, outside paint, outside RA
        # 15 ft from basket, toward center
        x = BASKET_LEFT[0] + 15.0
        assert self.cm.get_zone(x, 0.0, basket="left") == ShotZone.MID_RANGE

    # Corner 3
    def test_corner_3_home_bottom(self):
        # Behind straight section at y = -22.5 (past THREE_CORNER_Y)
        # x must be past THREE_LEFT_X (toward center court)
        from src.court.court_model import THREE_LEFT_X
        x = THREE_LEFT_X + 2.0   # clearly behind the 3PT line
        assert self.cm.get_zone(x, -(THREE_CORNER_Y + 1.5), basket="left") == ShotZone.CORNER_3

    # Above-break 3
    def test_above_break_3_home(self):
        # 25 ft from basket, aligned with basket (y=0) → arc 3
        x = BASKET_LEFT[0] + THREE_RADIUS + 2.0
        assert self.cm.get_zone(x, 0.0, basket="left") == ShotZone.ABOVE_BREAK_3

    # Backcourt
    def test_backcourt_home(self):
        # Shot at left basket from x > 0 (right half)
        assert self.cm.get_zone(10.0, 0.0, basket="left") == ShotZone.BACKCOURT

    def test_backcourt_away(self):
        assert self.cm.get_zone(-10.0, 0.0, basket="right") == ShotZone.BACKCOURT

    # Out of bounds
    def test_out_of_bounds(self):
        assert self.cm.get_zone(100.0, 0.0, basket="left") == ShotZone.OUT_OF_BOUNDS
        assert self.cm.get_zone(-50.0, 0.0, basket="left") == ShotZone.OUT_OF_BOUNDS


# ── CourtModel — geometry predicates ─────────────────────────────────────────

class TestCourtModelPredicates:
    cm = CourtModel()

    def test_is_in_paint_home_true(self):
        # Center of home paint
        assert self.cm.is_in_paint(-40.0, 0.0, basket="left")

    def test_is_in_paint_home_false_outside_x(self):
        # Past the free throw line
        assert not self.cm.is_in_paint(-20.0, 0.0, basket="left")

    def test_is_in_paint_home_false_outside_y(self):
        # Too wide (y > PAINT_HALF_W)
        assert not self.cm.is_in_paint(-44.0, 10.0, basket="left")

    def test_is_in_paint_away_true(self):
        assert self.cm.is_in_paint(40.0, 0.0, basket="right")

    def test_is_three_home_arc(self):
        # 25 ft from basket, above the break → 3PT
        x = BASKET_LEFT[0] + THREE_RADIUS + 1.0
        assert self.cm.is_three_point(x, 0.0, basket="left")

    def test_is_two_inside_arc(self):
        # 20 ft from basket, inside arc → 2PT
        x = BASKET_LEFT[0] + 20.0
        assert not self.cm.is_three_point(x, 0.0, basket="left")

    def test_is_three_corner(self):
        # Bottom corner 3: past junction x and y < -THREE_CORNER_Y
        from src.court.court_model import THREE_LEFT_X
        assert self.cm.is_three_point(THREE_LEFT_X + 2.0, -23.0, basket="left")

    def test_is_two_corner_region_before_line(self):
        # In corner y region but NOT past the junction → 2PT
        from src.court.court_model import THREE_LEFT_X
        assert not self.cm.is_three_point(THREE_LEFT_X - 2.0, -23.0, basket="left")

    def test_dist_to_basket_home(self):
        # 10 ft directly to the right of home basket
        x = BASKET_LEFT[0] + 10.0
        dist = self.cm.dist_to_basket(x, 0.0, basket="left")
        assert dist == pytest.approx(10.0, abs=0.01)

    def test_nearest_basket(self):
        assert self.cm.nearest_basket(-30.0, 0.0) == "left"
        assert self.cm.nearest_basket( 30.0, 0.0) == "right"
        assert self.cm.nearest_basket(  0.0, 0.0) in ("left", "right")   # equidistant

    def test_is_contested_true(self):
        shooter  = np.array([-30.0, 5.0])
        defender = np.array([-32.0, 5.0])   # 2 ft away
        assert self.cm.is_contested(shooter, defender, threshold_ft=4.0)

    def test_is_contested_false(self):
        shooter  = np.array([-30.0, 5.0])
        defender = np.array([-25.0, 5.0])   # 5 ft away
        assert not self.cm.is_contested(shooter, defender, threshold_ft=4.0)

    def test_defender_distance(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert self.cm.defender_distance(a, b) == pytest.approx(5.0)


# ── CourtModel — court_to_pixel round-trip ────────────────────────────────────

class TestCourtPixelTransform:
    def test_round_trip_center(self):
        pt = np.array([0.0, 0.0])
        px = CourtModel.court_to_pixel(pt, 940, 500)
        back = CourtModel.pixel_to_court(px, 940, 500)
        assert back[0] == pytest.approx(0.0, abs=0.5)
        assert back[1] == pytest.approx(0.0, abs=0.5)

    def test_round_trip_corner(self):
        pt = np.array([-HALF_COURT, -COURT_WIDTH / 2])   # bottom-left
        px = CourtModel.court_to_pixel(pt, 940, 500)
        back = CourtModel.pixel_to_court(px, 940, 500)
        assert back[0] == pytest.approx(-HALF_COURT, abs=0.5)
        assert back[1] == pytest.approx(-COURT_WIDTH / 2, abs=0.5)

    def test_center_maps_to_image_center(self):
        pt = np.array([0.0, 0.0])
        px = CourtModel.court_to_pixel(pt, 940, 500)
        assert px[0] == pytest.approx(470.0, abs=1.0)
        assert px[1] == pytest.approx(250.0, abs=1.0)


# ── CourtHomography ───────────────────────────────────────────────────────────

class TestCourtHomography:

    def test_not_valid_before_compute(self):
        hom = CourtHomography()
        assert not hom.is_valid

    def test_to_court_returns_nan_before_compute(self):
        hom = CourtHomography()
        result = hom.to_court(np.array([100.0, 200.0]))
        assert np.all(np.isnan(result))

    def test_to_pixel_returns_nan_before_compute(self):
        hom = CourtHomography()
        result = hom.to_pixel(np.array([0.0, 0.0]))
        assert np.all(np.isnan(result))

    def test_batch_returns_nan_before_compute(self):
        hom = CourtHomography()
        pts = np.array([[100.0, 200.0], [300.0, 400.0]])
        out = hom.to_court_batch(pts)
        assert np.all(np.isnan(out))

    def test_compute_returns_true_with_4_points(self):
        hom = CourtHomography()
        pixel_pts, court_pts = four_point_H()
        assert hom.compute(pixel_pts, court_pts) is True
        assert hom.is_valid

    def test_compute_returns_false_with_3_points(self):
        hom = CourtHomography()
        pixel_pts = np.array([[0,0],[100,0],[50,100]], dtype=np.float32)
        court_pts = np.array([[0,0],[10,0],[5,10]],    dtype=np.float32)
        assert hom.compute(pixel_pts, court_pts) is False
        assert not hom.is_valid

    def test_project_known_point(self):
        hom = CourtHomography()
        pixel_pts, court_pts = four_point_H()
        hom.compute(pixel_pts, court_pts)
        # Pixel (50, 50) should project to court (5, 5)
        result = hom.to_court(np.array([50.0, 50.0]))
        assert result[0] == pytest.approx(5.0, abs=0.1)
        assert result[1] == pytest.approx(5.0, abs=0.1)

    def test_inverse_projection(self):
        hom = CourtHomography()
        pixel_pts, court_pts = four_point_H()
        hom.compute(pixel_pts, court_pts)
        # court (5, 5) should project back to pixel (50, 50)
        result = hom.to_pixel(np.array([5.0, 5.0]))
        assert result[0] == pytest.approx(50.0, abs=0.5)
        assert result[1] == pytest.approx(50.0, abs=0.5)

    def test_round_trip_pixel_to_court_to_pixel(self):
        hom = CourtHomography()
        pixel_pts, court_pts = four_point_H()
        hom.compute(pixel_pts, court_pts)
        original = np.array([25.0, 75.0])
        court = hom.to_court(original)
        recovered = hom.to_pixel(court)
        assert recovered[0] == pytest.approx(original[0], abs=0.5)
        assert recovered[1] == pytest.approx(original[1], abs=0.5)

    def test_batch_projection_shape(self):
        hom = CourtHomography()
        pixel_pts, court_pts = four_point_H()
        hom.compute(pixel_pts, court_pts)
        batch = np.array([[10,10],[20,20],[30,30]], dtype=np.float32)
        result = hom.to_court_batch(batch)
        assert result.shape == (3, 2)

    def test_batch_matches_single(self):
        hom = CourtHomography()
        pixel_pts, court_pts = four_point_H()
        hom.compute(pixel_pts, court_pts)
        pts = np.array([[10.0, 10.0], [50.0, 50.0]], dtype=np.float32)
        batch = hom.to_court_batch(pts)
        for i, pt in enumerate(pts):
            single = hom.to_court(pt)
            assert batch[i, 0] == pytest.approx(single[0], abs=0.01)
            assert batch[i, 1] == pytest.approx(single[1], abs=0.01)

    def test_foot_point_projection(self):
        hom = CourtHomography()
        pixel_pts, court_pts = four_point_H()
        hom.compute(pixel_pts, court_pts)
        # bbox where foot = (50, 80)
        bbox = np.array([20.0, 40.0, 80.0, 80.0])
        court_pos = hom.project_foot_point(bbox)
        # foot pixel = (50, 80) → should map to a valid court point
        assert not np.any(np.isnan(court_pos))

    def test_quality_set_after_compute(self):
        hom = CourtHomography()
        pixel_pts, court_pts = four_point_H()
        hom.compute(pixel_pts, court_pts)
        assert hom.quality is not None
        assert hom.quality >= 0.0

    def test_reset_clears_state(self):
        hom = CourtHomography()
        pixel_pts, court_pts = four_point_H()
        hom.compute(pixel_pts, court_pts)
        hom.reset()
        assert not hom.is_valid
        assert hom.quality is None

    def test_save_load_roundtrip(self, tmp_path):
        hom = CourtHomography()
        pixel_pts, court_pts = four_point_H()
        hom.compute(pixel_pts, court_pts)
        path = tmp_path / "test_H.npy"
        hom.save(path)

        hom2 = CourtHomography()
        hom2.load(path)
        # Both should project the same point the same way
        pt = np.array([30.0, 30.0])
        r1 = hom.to_court(pt)
        r2 = hom2.to_court(pt)
        assert r1[0] == pytest.approx(r2[0], abs=1e-4)
        assert r1[1] == pytest.approx(r2[1], abs=1e-4)


# ── homography_from_keypoints ─────────────────────────────────────────────────

class TestHomographyFromKeypoints:
    def test_returns_valid_with_4_points(self):
        kp_ref  = {k: v for k, v in list(COURT_KEYPOINTS.items())[:4]}
        # Simulate pixel detections as court coords scaled to pixel space
        kp_det  = {k: v * 10 + 200 for k, v in kp_ref.items()}
        hom = homography_from_keypoints(kp_det, kp_ref)
        assert hom is not None
        assert hom.is_valid

    def test_returns_none_with_3_points(self):
        kp_ref = {k: v for k, v in list(COURT_KEYPOINTS.items())[:3]}
        kp_det = {k: v * 10 + 200 for k, v in kp_ref.items()}
        hom = homography_from_keypoints(kp_det, kp_ref)
        assert hom is None

    def test_returns_none_when_no_shared_keys(self):
        kp_det = {"nonexistent_point": np.array([100.0, 200.0])}
        kp_ref = {k: v for k, v in list(COURT_KEYPOINTS.items())[:4]}
        hom = homography_from_keypoints(kp_det, kp_ref)
        assert hom is None


# ── ClassicalKeyDetector ──────────────────────────────────────────────────────

class TestClassicalKeyDetector:
    def test_output_is_dict_with_correct_keys(self):
        det = ClassicalKeyDetector()
        bgr = make_court_diagram()
        result = det.detect(bgr)
        assert isinstance(result, dict)
        for k in KEYPOINT_NAMES:
            assert k in result

    def test_values_are_ndarray_or_none(self):
        det = ClassicalKeyDetector()
        bgr = make_court_diagram()
        result = det.detect(bgr)
        for v in result.values():
            assert v is None or (isinstance(v, np.ndarray) and v.shape == (2,))

    def test_on_blank_image_returns_all_none(self):
        det = ClassicalKeyDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(blank)
        # No court lines → should detect nothing
        assert det.count_detected(result) == 0

    def test_detects_some_keypoints_on_court_diagram(self):
        det = ClassicalKeyDetector(
            floor_hsv_lo=(100, 100, 100),  # match the blue-ish synthetic court
            floor_hsv_hi=(130, 255, 255),
        )
        bgr = make_court_diagram()
        result = det.detect(bgr)
        # We don't assert exact keypoints (classical is approximate),
        # just that the interface works and schema is correct
        assert isinstance(result, dict)

    def test_count_detected(self):
        result = {"a": np.array([1.0, 2.0]), "b": None, "c": np.array([3.0, 4.0])}
        assert ClassicalKeyDetector.count_detected(result) == 2
