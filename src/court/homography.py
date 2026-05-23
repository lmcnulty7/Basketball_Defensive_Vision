"""
src/court/homography.py

Computes the homography matrix H and projects coordinates between
video pixel space and real-world NBA court space.

What homography does
────────────────────
A broadcast camera looking at the court from an elevated angle creates a
perspective projection.  A homography H is a 3×3 matrix that captures this
projection and inverts it: given at least 4 matched point pairs
(pixel ↔ court-feet), H lets you map ANY pixel position to its real-world
court coordinate and vice versa.

Convention
──────────
  H maps:  pixel → court  (i.e. court_pt = H * pixel_pt  in homogeneous coords)
  H_inv :  court → pixel

Typical usage
─────────────
  hom = CourtHomography()
  hom.compute(pixel_pts, court_pts)          # given ≥4 matched pairs
  court_pos = hom.to_court(np.array([u, v])) # project one pixel point
  pixel_pos = hom.to_pixel(np.array([x, y])) # project back

  # Project an entire frame's player positions at once:
  foot_pts   = np.array([[u1,v1],[u2,v2],...])   # pixel foot-points
  court_pts  = hom.to_court_batch(foot_pts)       # Nx2 court coords
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum correspondences for a valid homography
MIN_POINTS = 4

# Reprojection-error threshold for RANSAC (pixels)
RANSAC_THRESHOLD = 5.0


class CourtHomography:
    """
    Manages the homography matrix between video pixels and court coordinates.

    Parameters
    ──────────
    ransac_threshold : Max reprojection error (pixels) for RANSAC inliers.

    Lifecycle
    ─────────
    1.  compute(pixel_pts, court_pts)  — call once per camera angle.
    2.  to_court() / to_pixel()        — call on every frame while the
                                         camera angle is stable.
    3.  reset()                        — call on camera cut; triggers
                                         recomputation at the next frame.
    """

    def __init__(self, ransac_threshold: float = RANSAC_THRESHOLD) -> None:
        self.ransac_threshold = ransac_threshold
        self._H:     Optional[np.ndarray] = None   # pixel → court
        self._H_inv: Optional[np.ndarray] = None   # court → pixel
        self._quality: Optional[float]    = None   # mean reprojection error (px)

    # ── Calibration ───────────────────────────────────────────────────────────

    def compute(
        self,
        pixel_pts: np.ndarray,
        court_pts: np.ndarray,
    ) -> bool:
        """
        Compute the homography from matched point pairs.

        Parameters
        ──────────
        pixel_pts : (N, 2) array of pixel coordinates  [u, v].
        court_pts : (N, 2) array of court coordinates  [x_ft, y_ft].
                    Must correspond 1-to-1 with pixel_pts.

        Returns
        ───────
        True if a valid H was computed; False if fewer than 4 points
        or RANSAC failed.
        """
        pixel_pts = np.asarray(pixel_pts, dtype=np.float32)
        court_pts = np.asarray(court_pts, dtype=np.float32)

        if len(pixel_pts) < MIN_POINTS:
            logger.warning(
                "Need ≥%d point pairs; got %d — H not computed",
                MIN_POINTS, len(pixel_pts),
            )
            return False

        H, mask = cv2.findHomography(
            pixel_pts, court_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )

        if H is None:
            logger.error("cv2.findHomography returned None — check point quality")
            return False

        self._H     = H
        self._H_inv = np.linalg.inv(H)

        # Compute mean reprojection error on inliers
        inlier_mask = mask.ravel().astype(bool)
        n_inliers   = int(inlier_mask.sum())
        reprojected = self.to_court_batch(pixel_pts[inlier_mask])
        errors = np.linalg.norm(reprojected - court_pts[inlier_mask], axis=1)
        self._quality = float(errors.mean()) if len(errors) > 0 else 0.0

        logger.info(
            "Homography computed: %d/%d inliers, mean reprojection error = %.3f ft",
            n_inliers, len(pixel_pts), self._quality,
        )
        return True

    def load(self, path: str | Path) -> None:
        """Load a previously saved H matrix from a .npy file."""
        arr = np.load(str(path))
        if arr.shape != (3, 3):
            raise ValueError(f"Expected (3,3) array, got {arr.shape}")
        self._H     = arr.astype(np.float64)
        self._H_inv = np.linalg.inv(self._H)
        logger.info("Homography loaded from %s", path)

    def save(self, path: str | Path) -> None:
        """Save H matrix to a .npy file for reuse."""
        if self._H is None:
            raise RuntimeError("No H matrix to save — call compute() first")
        np.save(str(path), self._H)
        logger.info("Homography saved to %s", path)

    def reset(self) -> None:
        """Clear the current H (e.g. on camera cut)."""
        self._H     = None
        self._H_inv = None
        self._quality = None
        logger.info("Homography reset")

    @property
    def is_valid(self) -> bool:
        return self._H is not None

    @property
    def quality(self) -> Optional[float]:
        """Mean reprojection error in feet (lower = better)."""
        return self._quality

    # ── Projection ────────────────────────────────────────────────────────────

    def to_court(self, pixel_pt: np.ndarray) -> np.ndarray:
        """
        Project a single pixel point to court coordinates.

        Parameters
        ──────────
        pixel_pt : [u, v] in pixels.

        Returns
        ───────
        [x_ft, y_ft] in court coordinates, or [nan, nan] if H is invalid.
        """
        if self._H is None:
            return np.array([np.nan, np.nan])
        return self._apply_H(self._H, pixel_pt)

    def to_pixel(self, court_pt: np.ndarray) -> np.ndarray:
        """
        Project a court coordinate back to pixel space.

        Parameters
        ──────────
        court_pt : [x_ft, y_ft].

        Returns
        ───────
        [u, v] in pixels, or [nan, nan] if H is invalid.
        """
        if self._H_inv is None:
            return np.array([np.nan, np.nan])
        return self._apply_H(self._H_inv, court_pt)

    def to_court_batch(self, pixel_pts: np.ndarray) -> np.ndarray:
        """
        Project (N, 2) pixel points to (N, 2) court coordinates.

        Uses cv2.perspectiveTransform for efficient batch projection.
        Returns array of NaNs if H is not valid.
        """
        if self._H is None:
            return np.full((len(pixel_pts), 2), np.nan)
        pts = pixel_pts.astype(np.float32).reshape(-1, 1, 2)
        result = cv2.perspectiveTransform(pts, self._H)
        return result.reshape(-1, 2)

    def to_pixel_batch(self, court_pts: np.ndarray) -> np.ndarray:
        """Project (N, 2) court coordinates to (N, 2) pixel points."""
        if self._H_inv is None:
            return np.full((len(court_pts), 2), np.nan)
        pts = court_pts.astype(np.float32).reshape(-1, 1, 2)
        result = cv2.perspectiveTransform(pts, self._H_inv)
        return result.reshape(-1, 2)

    def project_foot_point(self, bbox: np.ndarray) -> np.ndarray:
        """
        Project the foot-point of a bounding box to court coordinates.

        The foot-point is the bottom-center of the bounding box — the best
        approximation of where the player is touching the ground.

        Parameters
        ──────────
        bbox : [x1, y1, x2, y2] in pixels.

        Returns
        ───────
        [x_ft, y_ft] court coordinates.
        """
        foot_pixel = np.array([
            (bbox[0] + bbox[2]) / 2.0,   # cx
            bbox[3],                       # bottom of box
        ], dtype=np.float32)
        return self.to_court(foot_pixel)

    def project_foot_points_batch(self, bboxes: np.ndarray) -> np.ndarray:
        """
        Project foot-points for multiple bounding boxes.

        Parameters
        ──────────
        bboxes : (N, 4) array of [x1, y1, x2, y2].

        Returns
        ───────
        (N, 2) court coordinate array.
        """
        foot_pixels = np.stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2.0,
            bboxes[:, 3],
        ], axis=1).astype(np.float32)
        return self.to_court_batch(foot_pixels)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_H(H: np.ndarray, pt: np.ndarray) -> np.ndarray:
        """Apply a 3×3 homography to a single 2D point."""
        p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        r = H @ p
        return (r[:2] / r[2]).astype(np.float32)


# ── Convenience function ──────────────────────────────────────────────────────

def homography_from_keypoints(
    detected: Dict[str, np.ndarray],
    reference: Dict[str, np.ndarray],
    min_points: int = MIN_POINTS,
) -> Optional[CourtHomography]:
    """
    Build a CourtHomography from two dicts of matched named keypoints.

    Parameters
    ──────────
    detected  : {name: [u, v]}  — pixel coordinates from the detector.
    reference : {name: [x, y]}  — corresponding court coordinates in feet
                                  (typically from COURT_KEYPOINTS).

    Returns
    ───────
    A valid CourtHomography, or None if not enough shared keypoints.
    """
    shared = [k for k in detected if k in reference and detected[k] is not None]

    if len(shared) < min_points:
        logger.warning(
            "homography_from_keypoints: only %d shared points (need %d)",
            len(shared), min_points,
        )
        return None

    pixel_pts = np.array([detected[k]  for k in shared], dtype=np.float32)
    court_pts = np.array([reference[k] for k in shared], dtype=np.float32)

    hom = CourtHomography()
    success = hom.compute(pixel_pts, court_pts)
    return hom if success else None
