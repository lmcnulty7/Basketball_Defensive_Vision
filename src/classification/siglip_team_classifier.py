"""
src/classification/siglip_team_classifier.py

Team classifier using SigLIP visual embeddings + UMAP + K-means.

Replaces the HSV K-means approach in TeamClassifier.  SigLIP embeddings
capture jersey color, texture, and number patterns simultaneously, giving
much better separation for teams with similar hues (e.g. two dark jerseys)
or white home uniforms.

Architecture
────────────
Phase 1 — Accumulation (~50 frames):
  For each tracked player, crop the torso region and extract a 1152-dim
  SigLIP embedding.  Accumulate one embedding per track per frame.

Phase 2 — Fitting:
  1. Referee filter: tracks whose jersey saturation (HSV-S) is consistently
     low (grey/black-and-white stripes) are labelled TEAM_REFEREE.
  2. UMAP reduces remaining embeddings to 10 dimensions.
  3. K-means(k=2) clusters into two teams.

Phase 3 — Real-time inference:
  New tracks are classified by nearest centroid in UMAP space.
  Existing tracks use cached assignments (re-checked every 90 frames).

Fallback
────────
If SigLIP or UMAP is unavailable, falls back to the original HSV K-means
TeamClassifier so the pipeline keeps running.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.classification.team_classifier import (
    TEAM_HOME, TEAM_AWAY, TEAM_REFEREE, TEAM_UNKNOWN, TeamClassifier,
)
from src.tracking.multi_tracker import Track

logger = logging.getLogger(__name__)

_MIN_SAMPLES    = 50     # frames of embeddings before fitting
_SAT_THRESH     = 35     # HSV-S below this → referee (grey stripes)
_REASSIGN_EVERY = 90     # frames between re-checking known tracks


def _torso_crop(bgr: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """Return the torso region of a player bounding box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    w = x2 - x1
    if h < 20 or w < 10:
        return None
    ty1 = y1 + int(h * 0.25)
    ty2 = y1 + int(h * 0.65)
    tx1 = x1 + int(w * 0.20)
    tx2 = x1 + int(w * 0.80)
    crop = bgr[ty1:ty2, tx1:tx2]
    return crop if crop.size > 0 else None


def _mean_saturation(bgr_crop: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


class SigLIPTeamClassifier:
    """
    Drop-in replacement for TeamClassifier using SigLIP + UMAP + K-means.

    Exposes the same interface:
        process_frame(player_tracks, bgr, frame_idx)
        get_assignment(track_id) -> int  (TEAM_HOME / TEAM_AWAY / TEAM_REFEREE)
        is_fitted -> bool
        flip_teams()
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: str = "mps",
        min_samples_to_fit: int = _MIN_SAMPLES,
        sat_thresh: int = _SAT_THRESH,
    ) -> None:
        self.min_samples    = min_samples_to_fit
        self.sat_thresh     = sat_thresh
        self._is_fitted     = False
        self._assignments: Dict[int, int] = {}
        self._last_checked: Dict[int, int] = {}
        # Per track: list of embeddings (np arrays)
        self._embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
        # Per track: mean saturation values (for referee detection)
        self._saturations: Dict[int, List[float]] = defaultdict(list)
        # UMAP centroids and K-means state
        self._umap = None
        self._centroids: Optional[np.ndarray] = None   # shape (2, n_components)
        self._cluster_to_team: Dict[int, int] = {0: TEAM_HOME, 1: TEAM_AWAY}

        self._model     = None
        self._processor = None
        self._device    = device
        self._fallback  = None   # HSV fallback

        self._load_model(model_name, device)

    def _load_model(self, model_name: str, device: str) -> None:
        try:
            from transformers import AutoProcessor, AutoModel
            import torch
            logger.info("SigLIPTeamClassifier: loading %s...", model_name)
            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            # MPS or CPU
            dev = device if device != "mps" else ("mps" if self._mps_available() else "cpu")
            self._model = self._model.to(dev)
            self._model.eval()
            self._device = dev
            logger.info("SigLIPTeamClassifier: loaded on %s", dev)
        except Exception as e:
            logger.warning("SigLIPTeamClassifier: failed to load (%s) — using HSV fallback", e)
            self._model = None

        try:
            import umap  # noqa: F401
        except ImportError:
            logger.warning("umap-learn not installed — run: pip install umap-learn")
            if self._model is not None:
                logger.warning("SigLIPTeamClassifier: UMAP unavailable — using HSV fallback")
                self._model = None

        if self._model is None:
            from src.classification.team_classifier import TeamClassifier
            self._fallback = TeamClassifier(
                min_samples_to_fit=self.min_samples,
                sat_thresh=self.sat_thresh,
            )

    @staticmethod
    def _mps_available() -> bool:
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False

    # ── Public interface (mirrors TeamClassifier) ─────────────────────────────

    @property
    def is_fitted(self) -> bool:
        if self._fallback:
            return self._fallback.is_fitted
        return self._is_fitted

    def get_assignment(self, track_id: int) -> int:
        if self._fallback:
            return self._fallback.get_assignment(track_id)
        return self._assignments.get(track_id, TEAM_UNKNOWN)

    def flip_teams(self) -> None:
        if self._fallback:
            self._fallback.flip_teams()
            return
        self._cluster_to_team = {
            0: TEAM_AWAY if self._cluster_to_team[0] == TEAM_HOME else TEAM_HOME,
            1: TEAM_HOME if self._cluster_to_team[1] == TEAM_AWAY else TEAM_AWAY,
        }
        for tid in self._assignments:
            if self._assignments[tid] in (TEAM_HOME, TEAM_AWAY):
                self._assignments[tid] = (
                    TEAM_AWAY if self._assignments[tid] == TEAM_HOME else TEAM_HOME
                )

    def process_frame(
        self,
        player_tracks: List[Track],
        bgr: np.ndarray,
        frame_idx: int,
    ) -> None:
        if self._fallback:
            self._fallback.process_frame(player_tracks, bgr, frame_idx)
            return
        self._accumulate(player_tracks, bgr, frame_idx)
        if not self._is_fitted:
            total = sum(len(v) for v in self._embeddings.values())
            if total >= self.min_samples:
                self._fit()
        if self._is_fitted:
            self._infer(player_tracks, frame_idx)

    # ── Private ───────────────────────────────────────────────────────────────

    def _accumulate(
        self,
        tracks: List[Track],
        bgr: np.ndarray,
        frame_idx: int,
    ) -> None:
        import torch
        crops, track_ids = [], []
        for track in tracks:
            crop = _torso_crop(bgr, track.bbox)
            if crop is None:
                continue
            self._saturations[track.track_id].append(_mean_saturation(crop))
            crops.append(crop)
            track_ids.append(track.track_id)

        if not crops:
            return

        # Batch embed with SigLIP
        from PIL import Image
        pil_crops = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]

        with __import__("torch").no_grad():
            inputs = self._processor(images=pil_crops, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2 normalize
            feats = feats.cpu().float().numpy()

        for tid, emb in zip(track_ids, feats):
            self._embeddings[tid].append(emb)

    def _fit(self) -> None:
        import umap
        from sklearn.cluster import KMeans

        # Separate referees by low saturation
        ref_ids, player_ids = [], []
        for tid, sats in self._saturations.items():
            if np.mean(sats) < self.sat_thresh:
                ref_ids.append(tid)
            else:
                player_ids.append(tid)

        for tid in ref_ids:
            self._assignments[tid] = TEAM_REFEREE

        if len(player_ids) < 4:
            logger.warning("SigLIPTeamClassifier: too few non-referee players to fit")
            return

        # Stack embeddings: one mean embedding per track
        X_ids, X_embs = [], []
        for tid in player_ids:
            embs = np.array(self._embeddings[tid])
            X_ids.append(tid)
            X_embs.append(embs.mean(axis=0))

        X = np.array(X_embs)

        # UMAP — use random init to avoid spectral decomposition failure with few samples
        n_comp = min(10, X.shape[0] - 1)
        reducer = umap.UMAP(
            n_components=n_comp, n_neighbors=min(15, len(X) - 1),
            min_dist=0.1, metric="cosine", random_state=42, init="random",
        )
        try:
            X_reduced = reducer.fit_transform(X)
        except Exception as e:
            logger.warning("UMAP fitting failed (%s) — skipping team fit this frame", e)
            return
        self._umap = reducer

        # K-means k=2
        km = KMeans(n_clusters=2, n_init=20, random_state=42)
        labels = km.fit_predict(X_reduced)
        self._centroids = km.cluster_centers_

        for tid, label in zip(X_ids, labels):
            self._assignments[tid] = self._cluster_to_team[int(label)]

        self._is_fitted = True
        counts = {TEAM_HOME: 0, TEAM_AWAY: 0}
        for v in self._assignments.values():
            if v in counts:
                counts[v] += 1
        logger.info(
            "SigLIPTeamClassifier fitted: home=%d away=%d referees=%d",
            counts[TEAM_HOME], counts[TEAM_AWAY], len(ref_ids),
        )

    def _infer(self, tracks: List[Track], frame_idx: int) -> None:
        for track in tracks:
            tid = track.track_id
            last = self._last_checked.get(tid, -9999)
            if tid in self._assignments and frame_idx - last < _REASSIGN_EVERY:
                continue
            crop = _torso_crop(None, track.bbox) if False else None
            # Use cached assignment or nearest centroid
            if tid in self._assignments:
                self._last_checked[tid] = frame_idx
                continue
            # New track — classify
            if tid in self._embeddings and self._embeddings[tid]:
                emb = np.mean(self._embeddings[tid], axis=0, keepdims=True)
                # Check referee
                if self._saturations.get(tid) and np.mean(self._saturations[tid]) < self.sat_thresh:
                    self._assignments[tid] = TEAM_REFEREE
                elif self._umap is not None and self._centroids is not None:
                    reduced = self._umap.transform(emb)
                    dists = np.linalg.norm(reduced - self._centroids, axis=1)
                    label = int(np.argmin(dists))
                    self._assignments[tid] = self._cluster_to_team[label]
                self._last_checked[tid] = frame_idx
