"""
src/classification/team_classifier.py

Classifies each tracked player as home (0), away (1), or referee (2).

Algorithm
─────────
Phase 1 — Accumulation (first ~30 frames):
  For every tracked player, extract the dominant HSV color of their torso
  region and store it against their track_id.

Phase 2 — Fitting (once ≥ min_samples_to_fit color vectors collected):
  Referees are separated first by a low-saturation heuristic (NBA refs wear
  black/white stripes → gray → S < sat_thresh in HSV).  The remaining players
  are clustered into two groups with KMeans(k=2).  Each track is assigned to
  the cluster whose centroid is closest to its mean jersey color.

Phase 3 — Real-time inference:
  New tracks: referee heuristic → KMeans.predict().
  Known tracks: cached assignment (re-checked every reassign_interval frames).

Why jersey torso, not full crop?
─────────────────────────────────
  Head/face and legs contain noise (skin tones, shorts vary by player).
  The torso region (rows 25–65%, cols 20–80% of the bbox) is almost entirely
  jersey fabric and gives the cleanest color signal.

Team 0 vs 1 ambiguity
─────────────────────
  KMeans cluster labels are arbitrary.  Call flip_teams() to swap 0↔1 if
  the pipeline assigns them backwards.  A future enhancement would use
  court-side position (home/away half) to auto-resolve this.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.tracking.multi_tracker import Track


# ── Pure-numpy K-means (replaces sklearn.KMeans) ─────────────────────────────
# sklearn links Intel MKL/OpenMP which conflicts with PyTorch + FFmpeg on macOS,
# causing a segfault on first YOLO inference.  This numpy implementation is
# sufficient for K=2 clusters on 30-50 small color vectors.

class _NumpyKMeans:
    def __init__(self, n_clusters: int = 2, n_init: int = 10, max_iter: int = 300,
                 random_state: int = 42) -> None:
        self.n_clusters     = n_clusters
        self.n_init         = n_init
        self.max_iter       = max_iter
        self.random_state   = random_state
        self.cluster_centers_: np.ndarray = None
        self.labels_: np.ndarray          = None

    def fit(self, X: np.ndarray) -> "_NumpyKMeans":
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        best_inertia, best_centers, best_labels = float("inf"), None, None
        for _ in range(self.n_init):
            idx     = rng.choice(len(X), self.n_clusters, replace=False)
            centers = X[idx].copy()
            labels  = np.zeros(len(X), dtype=int)
            for _ in range(self.max_iter):
                # Assignment
                dists  = np.linalg.norm(X[:, None] - centers[None], axis=2)
                new_lbl = np.argmin(dists, axis=1)
                if np.array_equal(new_lbl, labels):
                    break
                labels = new_lbl
                # Update
                for k in range(self.n_clusters):
                    mask = labels == k
                    if mask.any():
                        centers[k] = X[mask].mean(axis=0)
            inertia = sum(
                float(np.linalg.norm(X[i] - centers[labels[i]]) ** 2)
                for i in range(len(X))
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels  = labels.copy()
        self.cluster_centers_ = best_centers
        self.labels_          = best_labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dists = np.linalg.norm(X[:, None] - self.cluster_centers_[None], axis=2)
        return np.argmin(dists, axis=1).astype(int)

logger = logging.getLogger(__name__)

TEAM_HOME     = 0
TEAM_AWAY     = 1
TEAM_REFEREE  = 2
TEAM_UNKNOWN  = -1

# Visualization colors (BGR) for each team label
TEAM_BGR: Dict[int, Tuple[int, int, int]] = {
    TEAM_HOME:    (0,  200,   0),   # green
    TEAM_AWAY:    (0,  100, 255),   # orange
    TEAM_REFEREE: (200, 200, 200),  # gray
    TEAM_UNKNOWN: (128, 128, 128),  # dark gray
}


class TeamClassifier:
    """
    Online jersey-color-based team classifier.

    Parameters
    ──────────
    n_teams              : Number of playing teams (always 2 for NBA).
    min_samples_to_fit   : Total color vectors to collect before fitting.
                           Lower = faster cold start but less accurate.
                           30–50 is a good range for 10fps processing.
    sat_thresh           : HSV-S threshold below which a player is called
                           a referee.  NBA refs wear black/white stripes
                           → mean saturation is low (< 50 in HSV [0-255]).
    auto_fit             : If True, fit() is triggered automatically once
                           min_samples_to_fit samples are accumulated.
    reassign_interval    : Re-run predict for a known track every N frames
                           to handle lighting changes or jersey number swaps.
    """

    def __init__(
        self,
        n_teams: int = 2,
        min_samples_to_fit: int = 30,
        sat_thresh: int = 50,
        auto_fit: bool = True,
        reassign_interval: int = 90,
    ) -> None:
        self.n_teams           = n_teams
        self.min_samples       = min_samples_to_fit
        self.sat_thresh        = sat_thresh
        self.auto_fit          = auto_fit
        self.reassign_interval = reassign_interval

        # track_id → list of (3,) HSV color vectors [0–1 normalized]
        self._gallery: Dict[int, List[np.ndarray]] = defaultdict(list)
        # track_id → assigned team_id
        self._assignments: Dict[int, int] = {}
        # track_id → frame_idx of last assignment (for periodic re-check)
        self._last_assigned: Dict[int, int] = {}

        self._kmeans: Optional[_NumpyKMeans] = None
        self._fitted = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_samples_collected(self) -> int:
        return sum(len(v) for v in self._gallery.values())

    # ── Main API ──────────────────────────────────────────────────────────────

    def process_frame(
        self,
        tracks: List[Track],
        bgr: np.ndarray,
        frame_idx: int = 0,
    ) -> Dict[int, int]:
        """
        Ingest one frame: extract jersey colors, auto-fit, predict.

        Parameters
        ──────────
        tracks    : Active player tracks from MultiTracker.
        bgr       : Raw BGR frame.
        frame_idx : Used for the periodic reassignment schedule.

        Returns
        ───────
        {track_id: team_id} for all tracks in this frame.
        Returns {} while still accumulating (before fitting).
        """
        for track in tracks:
            crop = self._get_crop(bgr, track.bbox)
            color = self._extract_jersey_color(crop)
            if color is not None:
                self._gallery[track.track_id].append(color)

        if self.auto_fit and not self._fitted:
            if self.n_samples_collected >= self.min_samples:
                self.fit()

        if not self._fitted:
            return {}

        result: Dict[int, int] = {}
        for track in tracks:
            tid = track.track_id
            needs_assign = (
                tid not in self._assignments
                or (frame_idx - self._last_assigned.get(tid, 0)) >= self.reassign_interval
            )
            if needs_assign:
                crop = self._get_crop(bgr, track.bbox)
                self._assignments[tid] = self._predict_one(tid, crop)
                self._last_assigned[tid] = frame_idx
            result[tid] = self._assignments[tid]

        return result

    def fit(self) -> bool:
        """
        Cluster the accumulated jersey-color gallery into team assignments.

        Returns True if fitting succeeded, False if insufficient data.
        """
        # Compute mean color per track (one representative vector per player)
        mean_colors: Dict[int, np.ndarray] = {}
        for tid, colors in self._gallery.items():
            if len(colors) >= 3:
                mean_colors[tid] = np.mean(colors, axis=0)

        if len(mean_colors) < self.n_teams:
            logger.warning(
                "TeamClassifier.fit(): need ≥%d tracks with samples; have %d",
                self.n_teams, len(mean_colors),
            )
            return False

        track_ids = list(mean_colors.keys())
        X = np.array([mean_colors[tid] for tid in track_ids])   # (N, 3)

        # Separate referees: low saturation in normalized HSV → S < sat_thresh/255
        sat_thresh_norm = self.sat_thresh / 255.0
        is_ref = X[:, 1] < sat_thresh_norm

        non_ref_X   = X[~is_ref]
        non_ref_ids = [tid for tid, r in zip(track_ids, is_ref) if not r]

        k = min(self.n_teams, max(1, len(non_ref_X)))
        self._kmeans = _NumpyKMeans(n_clusters=k, n_init=10, random_state=42)

        if len(non_ref_X) >= k:
            self._kmeans.fit(non_ref_X)
        else:
            # Fallback: cluster everyone together
            self._kmeans.fit(X)

        # Assign team labels to all tracks
        for tid, ref_flag, color in zip(track_ids, is_ref, X):
            if ref_flag:
                self._assignments[tid] = TEAM_REFEREE
            else:
                label = int(self._kmeans.predict(color.reshape(1, -1))[0])
                self._assignments[tid] = label

        self._fitted = True
        logger.info(
            "TeamClassifier fitted: %d tracks → %s",
            len(mean_colors),
            {tid: self._assignments[tid] for tid in track_ids},
        )
        return True

    def flip_teams(self) -> None:
        """
        Swap home (0) and away (1) labels everywhere.

        Call this when K-means assigns the teams backwards relative to
        which team is "home" vs "away" in your convention.
        """
        self._assignments = {
            tid: (TEAM_AWAY if v == TEAM_HOME else
                  TEAM_HOME if v == TEAM_AWAY else v)
            for tid, v in self._assignments.items()
        }
        if self._kmeans is not None and len(self._kmeans.cluster_centers_) == 2:
            self._kmeans.cluster_centers_ = self._kmeans.cluster_centers_[::-1]
        logger.info("TeamClassifier: teams flipped (0↔1)")

    def get_assignment(self, track_id: int) -> int:
        """Return the cached team assignment for a track_id, or TEAM_UNKNOWN."""
        return self._assignments.get(track_id, TEAM_UNKNOWN)

    def get_cluster_colors_bgr(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Return the mean jersey color per team as a BGR tuple for visualization.

        Each color represents the centroid of the K-means cluster.
        """
        if self._kmeans is None:
            return {}
        result = {}
        for i, center in enumerate(self._kmeans.cluster_centers_):
            # center is normalized [0-1] HSV
            hsv_pixel = np.uint8([[center * np.array([180, 255, 255])]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
            result[i] = tuple(int(v) for v in bgr_pixel)
        return result

    def reset(self) -> None:
        """Clear all state (call at halftime or on scene change)."""
        self._gallery.clear()
        self._assignments.clear()
        self._last_assigned.clear()
        self._kmeans  = None
        self._fitted  = False
        logger.info("TeamClassifier reset")

    # ── Private ───────────────────────────────────────────────────────────────

    def _predict_one(self, track_id: int, crop: Optional[np.ndarray]) -> int:
        """Predict team for a single track (called after fitting)."""
        # Use cached gallery mean if available
        if track_id in self._gallery and len(self._gallery[track_id]) >= 3:
            color = np.mean(self._gallery[track_id], axis=0)
        elif crop is not None:
            color = self._extract_jersey_color(crop)
            if color is None:
                return TEAM_UNKNOWN
        else:
            return TEAM_UNKNOWN

        if color[1] < self.sat_thresh / 255.0:
            return TEAM_REFEREE

        return int(self._kmeans.predict(color.reshape(1, -1))[0])

    @staticmethod
    def _extract_jersey_color(
        crop_bgr: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Extract the dominant jersey color from a player crop as a
        normalized (3,) HSV vector [H/180, S/255, V/255].

        Returns None if the crop is too small or empty.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 12 or w < 8:
            return None

        # Torso region: avoids head (noise) and legs (shorts differ from jersey)
        r1, r2 = int(h * 0.25), int(h * 0.65)
        c1, c2 = int(w * 0.20), int(w * 0.80)
        torso = crop_bgr[r1:r2, c1:c2]
        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Remove very dark pixels (shadows) and very bright pixels (specular)
        valid = pixels[(pixels[:, 2] > 25) & (pixels[:, 2] < 245)]
        if len(valid) == 0:
            valid = pixels

        mean_hsv = valid.mean(axis=0)
        return mean_hsv / np.array([180.0, 255.0, 255.0], dtype=np.float32)

    @staticmethod
    def _get_crop(
        bgr: np.ndarray,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Safe crop from frame using a [x1,y1,x2,y2] bounding box."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        fh, fw = bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return bgr[y1:y2, x1:x2]
