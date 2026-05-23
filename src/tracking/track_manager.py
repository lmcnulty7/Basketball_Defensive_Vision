"""
src/tracking/track_manager.py

Maintains rolling history buffers for all active tracks and exposes
per-track analytics (velocity, time on court, last-seen frame).

Why this exists separately from MultiTracker
────────────────────────────────────────────
MultiTracker emits a fresh List[Track] each frame but has no memory of
prior frames.  TrackManager is the persistent layer that accumulates those
snapshots into per-track deques and answers questions like:

  "What was player #7's velocity over the last 10 frames?"
  "How long has track #3 been active?"
  "Which tracks have gone stale and should be flagged?"

Every event detector reads from TrackManager, not from MultiTracker directly.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.tracking.multi_tracker import BALL_TRACK_ID, Track

logger = logging.getLogger(__name__)


# ── TrackState ────────────────────────────────────────────────────────────────

@dataclass
class TrackState:
    """
    One frame-snapshot of a track, stored in the history buffer.

    Lighter than a full Track — omits the raw_bgr reference and only
    keeps the fields needed by event detectors and the stats aggregator.

    court_pos is filled in after the court/ module runs homography.
    """
    track_id: int
    center: np.ndarray      # [cx, cy] pixel coords, float32
    bbox: np.ndarray        # [x1, y1, x2, y2] pixel coords, float32
    frame_idx: int
    timestamp_sec: float
    team_id: Optional[int] = None
    court_pos: Optional[np.ndarray] = None   # [x_ft, y_ft] once filled


# ── TrackManager ──────────────────────────────────────────────────────────────

class TrackManager:
    """
    Per-track rolling history and lifecycle management.

    Parameters
    ──────────
    history_len         : Max frames of history per track (default 30).
                          At stride=3 on 30fps, 30 frames = ~3 real seconds.
    camera_cut_threshold: If the fraction of previously-active tracks that
                          vanish in a single frame exceeds this value, a
                          camera cut is flagged.  Caller should then call
                          MultiTracker.reset().

    Typical usage
    ─────────────
        manager = TrackManager()
        for frame in reader:
            tracks = multi_tracker.update(frame.data, frame.frame_idx, ...)
            ball, is_pred, poss = ball_tracker.update(...)
            manager.update(tracks, ball, frame.frame_idx, frame.timestamp_sec)

            vel = manager.get_velocity(track_id=3)
            hist = manager.get_history(track_id=3, n=10)
    """

    def __init__(
        self,
        history_len: int = 30,
        camera_cut_threshold: float = 0.80,
        min_tracks_for_cut: int = 4,
    ) -> None:
        self.history_len = history_len
        self.camera_cut_threshold = camera_cut_threshold
        self.min_tracks_for_cut = min_tracks_for_cut

        # track_id → deque of TrackState (capped at history_len)
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.history_len)
        )
        # Currently active tracks (present in the most recent frame)
        self._active: Dict[int, Track] = {}
        # track_id → frame_idx of first appearance
        self._born: Dict[int, int] = {}
        # track_id → frame_idx of most recent appearance
        self._last_seen: Dict[int, int] = {}

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        player_tracks: List[Track],
        ball_track: Optional[Track],
        frame_idx: int,
        timestamp_sec: float,
    ) -> bool:
        """
        Ingest one frame of tracking output.

        Parameters
        ──────────
        player_tracks : Output of MultiTracker.update() for this frame.
        ball_track    : Output of BallTracker.update() for this frame (or None).
        frame_idx     : Absolute frame index.
        timestamp_sec : Frame timestamp in seconds.

        Returns
        ───────
        True if a camera cut is detected (large fraction of tracks vanished).
        """
        prev_active_ids = set(self._active.keys())
        new_active_ids  = {t.track_id for t in player_tracks}

        # ── Detect camera cut ─────────────────────────────────────────────────
        # Require a minimum number of previous tracks so that a single
        # close-up (1-2 players visible) doesn't trigger a reset that
        # discards all multi-track state.
        camera_cut = False
        if len(prev_active_ids) >= self.min_tracks_for_cut:
            vanished = prev_active_ids - new_active_ids
            frac_vanished = len(vanished) / len(prev_active_ids)
            if frac_vanished >= self.camera_cut_threshold:
                logger.warning(
                    "Camera cut detected at frame %d (%.0f%% of tracks vanished)",
                    frame_idx, frac_vanished * 100,
                )
                camera_cut = True

        # ── Register and update active player tracks ──────────────────────────
        self._active = {}
        for track in player_tracks:
            tid = track.track_id
            self._active[tid] = track

            if tid not in self._born:
                self._born[tid] = frame_idx

            self._last_seen[tid] = frame_idx

            self._history[tid].append(TrackState(
                track_id=tid,
                center=track.center.copy(),
                bbox=track.bbox.copy(),
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                team_id=track.team_id,
                court_pos=track.court_pos.copy() if track.court_pos is not None else None,
            ))

        # ── Register ball track ───────────────────────────────────────────────
        if ball_track is not None:
            self._history[BALL_TRACK_ID].append(TrackState(
                track_id=BALL_TRACK_ID,
                center=ball_track.center.copy(),
                bbox=ball_track.bbox.copy(),
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
            ))

        return camera_cut

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_history(
        self,
        track_id: int,
        n: Optional[int] = None,
    ) -> List[TrackState]:
        """
        Return the last n frames of history for track_id.

        Returns all history if n is None.  Returns an empty list if the
        track has never been seen (no KeyError — safe to call speculatively).
        """
        hist = list(self._history[track_id])
        return hist[-n:] if n is not None else hist

    def get_velocity(
        self,
        track_id: int,
        n_frames: int = 5,
    ) -> Optional[np.ndarray]:
        """
        Estimate average velocity in pixel-coordinates over the last n_frames.

        Returns [vx, vy] in px/frame, or None if fewer than 2 frames are
        available.

        Divide by stride to get px/original-frame; multiply by fps to get
        px/sec; apply homography to get ft/sec for real-world speed.
        """
        hist = self.get_history(track_id, n=n_frames)
        if len(hist) < 2:
            return None
        delta_pos = hist[-1].center - hist[0].center
        delta_frames = max(1, hist[-1].frame_idx - hist[0].frame_idx)
        return delta_pos / float(delta_frames)

    def get_active_tracks(self) -> Dict[int, Track]:
        """Snapshot of all currently active player tracks."""
        return dict(self._active)

    def n_active(self) -> int:
        return len(self._active)

    def frames_active(self, track_id: int) -> int:
        """
        How many frames has this track been alive?
        Returns 0 if the track_id has never been seen.
        """
        if track_id not in self._born or track_id not in self._last_seen:
            return 0
        return self._last_seen[track_id] - self._born[track_id] + 1

    def is_active(self, track_id: int) -> bool:
        return track_id in self._active

    def update_court_pos(
        self,
        track_id: int,
        court_pos: np.ndarray,
    ) -> None:
        """
        Back-fill the court_pos on the most recent TrackState for track_id.
        Called by the court/ module after homography projection.
        """
        if self._history[track_id]:
            self._history[track_id][-1].court_pos = court_pos.copy()
        if track_id in self._active:
            self._active[track_id].court_pos = court_pos.copy()

    def update_team_id(self, track_id: int, team_id: int) -> None:
        """Back-fill team assignment on the most recent TrackState."""
        if self._history[track_id]:
            self._history[track_id][-1].team_id = team_id
        if track_id in self._active:
            self._active[track_id].team_id = team_id
