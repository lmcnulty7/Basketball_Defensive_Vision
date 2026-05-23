"""
src/court/court_model.py

Encodes official NBA court geometry and provides zone / distance queries.

Coordinate system
─────────────────
  Origin : center of the court
  x-axis : -47 ft (left baseline) → 0 (center) → +47 ft (right baseline)
  y-axis : -25 ft (bottom sideline) → 0 (center) → +25 ft (top sideline)

  Left basket  : (-41.75,  0.0)
  Right basket : ( 41.75,  0.0)

This matches the coordinate system in configs/court.yaml.

Usage
─────
  from src.court.court_model import CourtModel, ShotZone
  court = CourtModel()
  zone  = court.get_zone(-35.0, -5.0, basket="left")   # → ShotZone.MID_RANGE
  is_3  = court.is_three_point(-35.0, -5.0, basket="left")
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

# ── Court constants (feet) ─────────────────────────────────────────────────────

COURT_LENGTH    = 94.0
COURT_WIDTH     = 50.0
HALF_COURT      = COURT_LENGTH / 2.0   # 47.0

BASKET_LEFT     = np.array([-41.75, 0.0])   # left  (home) basket
BASKET_RIGHT    = np.array([ 41.75, 0.0])   # right (away) basket

PAINT_DEPTH     = 19.0    # ft from baseline to free-throw line
PAINT_HALF_W    = 8.0     # half-width of paint (total = 16 ft)

THREE_RADIUS    = 23.75   # ft from basket center (arc)
THREE_CORNER_Y  = 22.0    # |y| value where corner straight-section starts
                          # (= 25 - 3 ft from sideline)

# x-position where the 3PT arc meets the corner straight section.
# Solve: sqrt((x - basket_x)^2 + THREE_CORNER_Y^2) = THREE_RADIUS
# → THREE_CORNER_DX = sqrt(THREE_RADIUS^2 - THREE_CORNER_Y^2)
THREE_CORNER_DX = math.sqrt(THREE_RADIUS ** 2 - THREE_CORNER_Y ** 2)  # ≈ 8.95 ft
THREE_LEFT_X    = BASKET_LEFT[0]  + THREE_CORNER_DX   # ≈ -32.80
THREE_RIGHT_X   = BASKET_RIGHT[0] - THREE_CORNER_DX   # ≈  32.80

RESTRICTED_RADIUS = 4.0   # restricted-area (no-charge) circle radius

# ── Canonical keypoints (name → [x, y] in court feet) ─────────────────────────
# These are the court landmarks detected by the keypoint detector and used to
# compute the homography matrix H.  Names match the anchors in court.yaml.

COURT_KEYPOINTS: Dict[str, np.ndarray] = {
    # Left half
    "home_paint_bl":   np.array([-HALF_COURT,            -PAINT_HALF_W]),  # (-47, -8)
    "home_paint_tl":   np.array([-HALF_COURT,             PAINT_HALF_W]),  # (-47, +8)
    "home_paint_br":   np.array([-HALF_COURT + PAINT_DEPTH, -PAINT_HALF_W]),  # (-28, -8)
    "home_paint_tr":   np.array([-HALF_COURT + PAINT_DEPTH,  PAINT_HALF_W]),  # (-28, +8)
    "home_three_bl":   np.array([THREE_LEFT_X,           -THREE_CORNER_Y]),  # (-32.8, -22)
    "home_three_tl":   np.array([THREE_LEFT_X,            THREE_CORNER_Y]),  # (-32.8, +22)
    "home_basket":     BASKET_LEFT.copy(),
    # Center
    "center":          np.array([0.0,  0.0]),
    "half_bot":        np.array([0.0, -COURT_WIDTH / 2]),
    "half_top":        np.array([0.0,  COURT_WIDTH / 2]),
    # Right half
    "away_paint_bl":   np.array([ HALF_COURT - PAINT_DEPTH, -PAINT_HALF_W]),  # (+28, -8)
    "away_paint_tl":   np.array([ HALF_COURT - PAINT_DEPTH,  PAINT_HALF_W]),  # (+28, +8)
    "away_paint_br":   np.array([ HALF_COURT,              -PAINT_HALF_W]),   # (+47, -8)
    "away_paint_tr":   np.array([ HALF_COURT,               PAINT_HALF_W]),   # (+47, +8)
    "away_three_bl":   np.array([THREE_RIGHT_X,            -THREE_CORNER_Y]), # (+32.8, -22)
    "away_three_tl":   np.array([THREE_RIGHT_X,             THREE_CORNER_Y]), # (+32.8, +22)
    "away_basket":     BASKET_RIGHT.copy(),
}


# ── Shot zones ────────────────────────────────────────────────────────────────

class ShotZone(str, Enum):
    RESTRICTED_AREA = "restricted_area"
    PAINT_NON_RA    = "paint_non_ra"
    MID_RANGE       = "mid_range"
    CORNER_3        = "corner_3"
    ABOVE_BREAK_3   = "above_break_3"
    BACKCOURT       = "backcourt"
    OUT_OF_BOUNDS   = "out_of_bounds"


# ── CourtModel ────────────────────────────────────────────────────────────────

class CourtModel:
    """
    NBA court geometry helper.

    All inputs and outputs are in feet using the center-origin coordinate system
    described at the top of this file.

    Example
    ───────
        court = CourtModel()
        zone = court.get_zone(-35.0, 0.0, basket="left")   # mid-range
        court.is_three_point(-33.0, -23.0, basket="left")  # True (corner 3)
    """

    # ── Zone queries ──────────────────────────────────────────────────────────

    def get_zone(self, x: float, y: float, basket: str = "left") -> ShotZone:
        """
        Classify a court position into a shot zone relative to a basket.

        Parameters
        ──────────
        x, y   : Court coordinates in feet (center-origin system).
        basket : "left" or "right" — which basket the shot is aimed at.
        """
        # Out of bounds
        if not (-HALF_COURT <= x <= HALF_COURT and
                -COURT_WIDTH / 2 <= y <= COURT_WIDTH / 2):
            return ShotZone.OUT_OF_BOUNDS

        # Backcourt (shot taken from the wrong half)
        if basket == "left"  and x > 0:
            return ShotZone.BACKCOURT
        if basket == "right" and x < 0:
            return ShotZone.BACKCOURT

        dist = self._dist_from_basket(x, y, basket)

        if dist <= RESTRICTED_RADIUS:
            return ShotZone.RESTRICTED_AREA

        if self.is_in_paint(x, y, basket):
            return ShotZone.PAINT_NON_RA

        if self.is_three_point(x, y, basket):
            if abs(y) >= THREE_CORNER_Y:
                return ShotZone.CORNER_3
            return ShotZone.ABOVE_BREAK_3

        return ShotZone.MID_RANGE

    def is_three_point(self, x: float, y: float, basket: str = "left") -> bool:
        """
        Return True if the court position is behind the 3PT line.

        In the corner region (|y| >= THREE_CORNER_Y = 22 ft), the boundary
        is the straight line at |y| = 22 ft.  Above the break, the boundary
        is the arc at THREE_RADIUS = 23.75 ft from the basket.
        """
        # In the corner zone: check if shooter is behind the straight section
        if abs(y) >= THREE_CORNER_Y:
            if basket == "left":
                return x >= THREE_LEFT_X       # must be toward center from junction
            else:
                return x <= THREE_RIGHT_X

        # Above the break: check arc distance
        return self._dist_from_basket(x, y, basket) >= THREE_RADIUS

    def is_in_paint(self, x: float, y: float, basket: str = "left") -> bool:
        """Return True if the position is inside the paint (key) for this basket."""
        if abs(y) > PAINT_HALF_W:
            return False
        if basket == "left":
            return -HALF_COURT <= x <= -HALF_COURT + PAINT_DEPTH
        else:
            return HALF_COURT - PAINT_DEPTH <= x <= HALF_COURT

    def dist_to_basket(self, x: float, y: float, basket: str = "left") -> float:
        """Euclidean distance in feet from (x, y) to the specified basket."""
        return self._dist_from_basket(x, y, basket)

    def nearest_basket(self, x: float, y: float) -> str:
        """Return "left" or "right" depending on which basket is closer."""
        dl = self._dist_from_basket(x, y, "left")
        dr = self._dist_from_basket(x, y, "right")
        return "left" if dl <= dr else "right"

    # ── Defensive utilities ───────────────────────────────────────────────────

    def defender_distance(
        self,
        shooter_pos: np.ndarray,
        defender_pos: np.ndarray,
    ) -> float:
        """
        Distance in feet between a shooter and defender.

        Both positions are court-coordinate [x, y] arrays.
        The threshold for "contested" in the NBA analytics community is 4 ft.
        """
        return float(np.linalg.norm(shooter_pos - defender_pos))

    def is_contested(
        self,
        shooter_pos: np.ndarray,
        defender_pos: np.ndarray,
        threshold_ft: float = 4.0,
    ) -> bool:
        """Return True if a defender is within threshold_ft of the shooter."""
        return self.defender_distance(shooter_pos, defender_pos) <= threshold_ft

    def matchup_pair(
        self,
        offensive_positions: Dict[int, np.ndarray],
        defensive_positions: Dict[int, np.ndarray],
        max_dist_ft: float = 10.0,
    ) -> Dict[int, Optional[int]]:
        """
        Assign each offensive player to their closest defender.

        Uses a greedy nearest-neighbor assignment (not Hungarian).
        The Hungarian version is in stats/matchup_tracker.py, which is
        called once per frame for the authoritative matchup assignment.
        This version is a fast approximation for event detectors.

        Returns {offensive_track_id: defensive_track_id or None}
        """
        assignments: Dict[int, Optional[int]] = {}
        used_defs: set = set()

        off_ids  = sorted(offensive_positions.keys())
        def_ids  = list(defensive_positions.keys())

        for oid in off_ids:
            op = offensive_positions[oid]
            best_dist = float("inf")
            best_did  = None
            for did in def_ids:
                if did in used_defs:
                    continue
                d = float(np.linalg.norm(op - defensive_positions[did]))
                if d < best_dist:
                    best_dist = d
                    best_did  = did
            if best_dist <= max_dist_ft:
                assignments[oid] = best_did
                if best_did is not None:
                    used_defs.add(best_did)
            else:
                assignments[oid] = None

        return assignments

    # ── Court-to-pixel helpers (for visualization) ───────────────────────────

    @staticmethod
    def court_to_pixel(
        court_pt: np.ndarray,
        img_w: int = 940,
        img_h: int = 500,
    ) -> np.ndarray:
        """
        Map a court coordinate [x, y] to pixel coordinates in a court diagram.

        The diagram is img_w × img_h pixels representing the full 94×50 ft court.
        Left edge = x=-47, right edge = x=+47, bottom = y=-25, top = y=+25.
        """
        px = (court_pt[0] + HALF_COURT) / COURT_LENGTH * img_w
        py = (COURT_WIDTH / 2 - court_pt[1]) / COURT_WIDTH * img_h   # y flipped
        return np.array([px, py], dtype=np.float32)

    @staticmethod
    def pixel_to_court(
        px_pt: np.ndarray,
        img_w: int = 940,
        img_h: int = 500,
    ) -> np.ndarray:
        """Inverse of court_to_pixel."""
        x = px_pt[0] / img_w * COURT_LENGTH - HALF_COURT
        y = COURT_WIDTH / 2 - px_pt[1] / img_h * COURT_WIDTH
        return np.array([x, y], dtype=np.float32)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dist_from_basket(x: float, y: float, basket: str) -> float:
        b = BASKET_LEFT if basket == "left" else BASKET_RIGHT
        return math.sqrt((x - b[0]) ** 2 + (y - b[1]) ** 2)
