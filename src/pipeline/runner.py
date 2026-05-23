"""
src/pipeline/runner.py

PipelineRunner — the top-level orchestrator.

Connects all six pipeline stages into a single frame loop:

  VideoReader → MultiTracker → BallTracker → TeamClassifier
      → KeypointDetector → CourtHomography → TrackManager
      → EventOrchestrator → MatchupTracker → SpeedCalculator
      → StatsAggregator

Every stage reads its inputs from the previous stage's outputs via
PipelineState, which is assembled fresh each frame.

Camera-cut handling
───────────────────
TrackManager.update() returns True when >80% of tracks vanish in a single
frame (camera cut).  The runner responds by resetting the tracker and H
matrix so fresh IDs and court calibration start for the new angle.

Homography bootstrapping
────────────────────────
H is not available at frame 0.  The keypoint detector runs every frame
until enough keypoints are found to compute H.  Once valid, H is cached
and only recomputed if a camera cut is detected.  Stats that require real-
world distances (contested shot, matchup time, defensive speed) are simply
skipped until H becomes valid.

Team-classification cold start
───────────────────────────────
TeamClassifier accumulates jersey-color samples for the first
min_samples_to_fit frames before fitting KMeans.  During the cold-start
window, event detectors still run (using team_id=TEAM_UNKNOWN), but
matchup and speed stats are skipped because offense/defense is unknown.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.classification.team_classifier import (
    TEAM_UNKNOWN, TeamClassifier, TEAM_BGR,
)
from src.classification.siglip_team_classifier import SigLIPTeamClassifier
from src.detection.jersey_reader import JerseyReader
from src.tracking.player_identity import PlayerIdentityResolver
from src.court.court_model import COURT_KEYPOINTS, CourtModel
from src.court.homography import CourtHomography, homography_from_keypoints
from src.court.keypoint_detector import NeuralKeyDetector
from src.detection.ball_detector import BallDetector
from src.events.assist_detector import AssistDetector
from src.events.block_detector import BlockDetector
from src.events.contest_detector import ContestDetector
from src.events.deflection_detector import DeflectionDetector
from src.events.event import PipelineState, PossessionState
from src.events.event_detector import EventOrchestrator
from src.events.rebound_detector import ReboundDetector
from src.events.shot_detector import ShotDetector
from src.events.steal_detector import StealDetector
from src.events.turnover_detector import TurnoverDetector
from src.tracking.possession_tracker import PossessionTracker
from src.ingestion.video_reader import VideoReader
from src.pipeline.state import PipelineConfig, PipelineOutput
from src.stats.aggregator import StatsAggregator
from src.stats.matchup_tracker import MatchupTracker
from src.stats.speed_calculator import SpeedCalculator
from src.tracking.ball_tracker import BallTracker
from src.tracking.multi_tracker import MultiTracker
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class PipelineRunner:
    """
    End-to-end NBA defensive statistics pipeline.

    Parameters
    ──────────
    config : PipelineConfig controlling all thresholds and paths.

    Example
    ───────
        config = PipelineConfig.from_yaml("configs/pipeline.yaml",
                                           "configs/models.yaml")
        runner = PipelineRunner(config)
        output = runner.run("data/raw/game_clip.mp4")
        output.print_summary()
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._build_modules()
        self._reset_frame_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, video_path: str) -> PipelineOutput:
        """
        Process an entire video clip and return defensive stats.

        Parameters
        ──────────
        video_path : Path to .mp4 (or any OpenCV-supported format).

        Returns
        ───────
        PipelineOutput containing event log, per-player stats, and summary.
        """
        vpath = Path(video_path)
        if not vpath.exists():
            raise FileNotFoundError(f"Video not found: {vpath}")

        logger.info("PipelineRunner.run() → %s", vpath.name)
        self._reset_frame_state()

        # Try to init identity resolver from game_id embedded in filename
        # e.g. clip_10m00_18m00_201206070BOS.mp4 or just use pbp.db lookup
        self._identity = self._init_identity(vpath)

        writer = None
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with VideoReader(str(vpath), stride=self.config.stride) as reader:
            meta = reader.metadata
            duration_sec = meta.duration_sec
            self._total_frames = meta.total_frames
            logger.info("Video: %s", meta)

            if self.config.save_annotated_video:
                out_path = output_dir / f"{vpath.stem}_annotated.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(out_path), fourcc,
                    self.config.annotated_video_fps,
                    (meta.width, meta.height),
                )

            for frame in reader:
                events = self._process_frame(frame.data, frame.frame_idx, frame.timestamp_sec)

                if writer is not None:
                    annotated = self._annotate(frame.data, frame.frame_idx, events)
                    writer.write(annotated)

        if writer is not None:
            writer.release()

        return self._build_output(duration_sec)

    def process_frame(
        self,
        bgr: np.ndarray,
        frame_idx: int = 0,
        timestamp_sec: float = 0.0,
    ) -> list:
        """
        Process a single frame (streaming / real-time use).

        Returns the list of events that fired this frame.
        """
        return self._process_frame(bgr, frame_idx, timestamp_sec)

    def get_output(self, duration_sec: float = 0.0) -> PipelineOutput:
        """Return current output (call after streaming finishes)."""
        return self._build_output(duration_sec)

    def reset(self) -> None:
        """Hard reset all state (call between clips)."""
        self._build_modules()
        self._reset_frame_state()

    # ── Module initialization ─────────────────────────────────────────────────

    def _build_modules(self) -> None:
        cfg = self.config

        self._tracker = MultiTracker(
            weights_path=cfg.player_weights,
            conf_thresh=cfg.player_conf,
            iou_thresh=cfg.iou_threshold,
            device=cfg.device,
            tracker_config=cfg.tracker_config,
        )

        tracknet_path = Path(cfg.tracknet_weights)
        if tracknet_path.exists():
            from src.detection.ball_detector import TrackNetDetector
            ball_det = TrackNetDetector(
                weights_path=tracknet_path,
                conf_thresh=cfg.tracknet_conf,
                seq_len=cfg.tracknet_seq_len,
                device=cfg.device,
            )
        else:
            ball_det = BallDetector(
                weights_path=cfg.ball_weights,
                fallback_weights=cfg.player_weights,
                conf_thresh=cfg.ball_conf,
                device=cfg.device,
            )
            if ball_det.mode == "disabled":
                ball_det.enable_color_detection()
        self._ball_tracker = BallTracker(
            ball_det,
            seq_len=cfg.tracknet_seq_len,
        )

        self._team_clf = SigLIPTeamClassifier(
            device=cfg.device,
            min_samples_to_fit=cfg.min_samples_to_fit,
            sat_thresh=cfg.sat_thresh,
        )
        self._jersey_reader  = JerseyReader(device=cfg.device)
        self._identity: Optional[PlayerIdentityResolver] = None

        self._kp_detector = NeuralKeyDetector(
            weights_path=cfg.court_kp_weights,
            device=cfg.device,
        )
        self._homography = CourtHomography()
        self._court      = CourtModel()
        self._manager    = TrackManager(history_len=60)

        # Wire detectors together
        steal_det   = StealDetector()
        contest_det = ContestDetector()
        rebound_det = ReboundDetector()
        shot_det    = ShotDetector()
        assist_det  = AssistDetector()
        tov_det     = TurnoverDetector()

        contest_det.set_steal_detector(steal_det)
        shot_det.set_rebound_detector(rebound_det)
        assist_det.set_shot_detector(shot_det)
        tov_det.set_steal_detector(steal_det)

        # Keep references for possession tracker access
        self._shot_det   = shot_det
        self._assist_det = assist_det

        self._orchestrator = EventOrchestrator([
            steal_det,
            tov_det,
            BlockDetector(),
            DeflectionDetector(),
            contest_det,
            rebound_det,
            shot_det,
            assist_det,
        ])

        self._possession = PossessionTracker()

        self._matchup    = MatchupTracker(stride=cfg.stride, fps=cfg.source_fps)
        self._speed      = SpeedCalculator(stride=cfg.stride, fps=cfg.source_fps)
        self._aggregator = StatsAggregator()

    def _reset_frame_state(self) -> None:
        self._n_frames        = 0
        self._skipped_frames  = 0
        self._total_frames    = 0
        self._all_track_ids: set = set()
        self._last_tracks     = []
        self._last_assignments: dict = {}

    # ── Per-frame logic ───────────────────────────────────────────────────────

    def _process_frame(
        self,
        bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
    ) -> list:
        # ── 0. Court-visibility gate ──────────────────────────────────────────
        # Runs in <1 ms.  Skips close-ups, replays, ads, and timeouts before
        # any expensive model inference touches the frame.
        if not self._is_court_visible(bgr):
            self._skipped_frames += 1
            if self._skipped_frames % 30 == 1:
                logger.debug(
                    "Frame %d skipped — court not visible (%d total skipped)",
                    frame_idx, self._skipped_frames,
                )
            return []

        # ── 1. Detect + track players ─────────────────────────────────────────
        player_tracks = self._tracker.update(bgr, frame_idx, timestamp_sec)
        self._last_tracks = player_tracks

        # ── 2. Track ball ─────────────────────────────────────────────────────
        ball_track, ball_is_pred, possessor_id = self._ball_tracker.update(
            bgr, player_tracks, frame_idx, timestamp_sec,
        )

        # ── 3. Team classification ────────────────────────────────────────────
        self._team_clf.process_frame(player_tracks, bgr, frame_idx)
        assignments: dict = {}
        for t in player_tracks:
            team = self._team_clf.get_assignment(t.track_id)
            if team != TEAM_UNKNOWN:
                assignments[t.track_id] = team
                t.team_id = team
        self._last_assignments = assignments

        # ── 3b. Jersey reading + player identity ─────────────────────────────
        self._jersey_reader.update(player_tracks, bgr, frame_idx)
        for t in player_tracks:
            jersey = self._jersey_reader.get_number(t.track_id)
            if jersey:
                t.jersey_number = jersey
                if self._identity:
                    team_side = {0: "home", 1: "away"}.get(
                        assignments.get(t.track_id)
                    )
                    name = self._identity.resolve(t.track_id, jersey, team_side)
                    if name:
                        t.player_name = name

        # ── 4. Court homography ───────────────────────────────────────────────
        if not self._homography.is_valid:
            kps = self._kp_detector.detect(bgr)
            n_kp = self._kp_detector.count_detected(kps)
            if n_kp >= self.config.min_keypoints:
                hom = homography_from_keypoints(kps, COURT_KEYPOINTS)
                if hom is not None:
                    self._homography = hom
                    logger.info(
                        "Homography computed at frame %d (%d KPs, err=%.3f ft)",
                        frame_idx, n_kp, hom.quality or 0.0,
                    )

        # ── 5. Project foot-points to court coords ────────────────────────────
        if self._homography.is_valid:
            for t in player_tracks:
                t.court_pos = self._homography.project_foot_point(t.bbox)
            if ball_track is not None:
                ball_track.court_pos = self._homography.to_court(ball_track.center)

        # ── 6. Update TrackManager ────────────────────────────────────────────
        camera_cut = self._manager.update(
            player_tracks, ball_track, frame_idx, timestamp_sec
        )
        for t in player_tracks:
            if t.team_id is not None:
                self._manager.update_team_id(t.track_id, t.team_id)
            if t.court_pos is not None:
                self._manager.update_court_pos(t.track_id, t.court_pos)

        if camera_cut:
            logger.warning("Camera cut at frame %d — resetting tracker + H", frame_idx)
            self._tracker.reset()
            self._homography.reset()
            self._possession.reset_for_camera_cut()

        # ── 7. Update possession state machine ────────────────────────────────
        ball_court = (ball_track.court_pos
                      if ball_track is not None else None)
        poss_state, attacking_basket = self._possession.update(
            possessor_id, assignments, ball_court, frame_idx,
        )

        # ── 8. Build PipelineState and run event detectors ────────────────────
        ball_history = self._ball_tracker.get_history()
        state = PipelineState(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            raw_bgr=bgr,
            player_tracks=player_tracks,
            ball_track=ball_track,
            ball_is_predicted=ball_is_pred,
            ball_possessor_id=possessor_id,
            ball_track_history=list(ball_history),
            team_assignments=assignments,
            possession_state=poss_state,
            attacking_basket=attacking_basket,
        )
        events = self._orchestrator.update(state, self._manager)

        # Expose fired events to assist detector via state (lightweight coupling)
        state._frame_events = events

        # ── 9. Matchup + speed (requires teams + court positions) ────────────
        self._all_track_ids.update(t.track_id for t in player_tracks)
        if self._team_clf.is_fitted:
            off_pos = {
                t.track_id: t.court_pos
                for t in state.get_offensive_tracks()
                if t.court_pos is not None
            }
            def_pos = {
                t.track_id: t.court_pos
                for t in state.get_defensive_tracks()
                if t.court_pos is not None
            }
            if off_pos or def_pos:
                matchups = self._matchup.update(off_pos, def_pos)
                self._speed.update(self._manager, matchups, def_pos, off_pos)

        self._n_frames += 1
        if self._n_frames % 30 == 0:
            logger.info(
                "  frame %d / %d  |  tracks=%d  |  H=%s  |  teams=%s",
                frame_idx, self._total_frames,
                len(self._last_tracks),
                "✓" if self._homography.is_valid else "–",
                "✓" if self._team_clf.is_fitted else "–",
            )
        return events

    # ── Output assembly ───────────────────────────────────────────────────────

    def _build_output(self, duration_sec: float) -> PipelineOutput:
        all_ids = list(self._all_track_ids) or [
            t.track_id for t in self._last_tracks
        ]
        stats_by_id = self._aggregator.compute(
            self._orchestrator.event_log,
            self._matchup,
            self._speed,
            all_ids,
        )
        total = self._n_frames + self._skipped_frames
        skip_pct = (self._skipped_frames / total * 100) if total else 0
        logger.info(
            "Finished: %d frames processed, %d skipped (%.1f%% non-court)",
            self._n_frames, self._skipped_frames, skip_pct,
        )
        return PipelineOutput(
            event_log=list(self._orchestrator.event_log),
            stats_table=self._aggregator.to_table(stats_by_id),
            event_summary=self._orchestrator.summary(),
            n_frames_processed=self._n_frames,
            duration_sec=duration_sec,
            h_valid=self._homography.is_valid,
            teams_fitted=self._team_clf.is_fitted,
        )

    # ── Identity helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _init_identity(vpath: Path) -> Optional[PlayerIdentityResolver]:
        """Try to find a game_id for this clip and load the roster."""
        try:
            import sqlite3
            db = Path(__file__).resolve().parents[2] / "data" / "pbp.db"
            if not db.exists():
                return None
            conn = sqlite3.connect(str(db))
            # Check if any game has a roster stored (use first match by date)
            rows = conn.execute(
                "SELECT DISTINCT game_id FROM rosters ORDER BY game_id"
            ).fetchall()
            conn.close()
            if not rows:
                return None
            # Use stem name to guess game: future enhancement
            # For now, return None and let the clip batch script pass game_id explicitly
            return None
        except Exception:
            return None

    def set_game_id(self, game_id: str) -> None:
        """Explicitly set the game context for identity resolution."""
        self._identity = PlayerIdentityResolver(game_id)

    # ── Court visibility gate ─────────────────────────────────────────────────

    @staticmethod
    def _is_court_visible(
        bgr: np.ndarray,
        min_floor_fraction: float = 0.12,
        max_floor_fraction: float = 0.55,
    ) -> bool:
        """
        Return True if this frame shows a broadcast shot with multiple players.

        Uses HSV color thresholding on the maple-wood floor color.

        Typical values:
          Full-court wide shot  → 30–50% floor pixels  ← accepted
          Half-court shot       → 15–30% floor pixels  ← accepted
          Floor-level close-up  → 55–80% floor pixels  ← rejected (max)
          Face/crowd close-up   →  0–8%  floor pixels  ← rejected (min)
          Scoreboard / ad       →  0–3%  floor pixels  ← rejected (min)
        """
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo   = np.array([8,  25,  90], dtype=np.uint8)
        hi   = np.array([38, 210, 245], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        frac = float(mask.sum()) / (255.0 * bgr.shape[0] * bgr.shape[1])
        return min_floor_fraction <= frac <= max_floor_fraction

    # ── Visualization ─────────────────────────────────────────────────────────

    def _annotate(
        self,
        bgr: np.ndarray,
        frame_idx: int,
        events: list,
    ) -> np.ndarray:
        """Draw tracks, teams, and event labels on the frame."""
        canvas = bgr.copy()

        for t in self._last_tracks:
            team_id = self._last_assignments.get(t.track_id)
            color   = TEAM_BGR.get(team_id, (128, 128, 128))
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"#{t.track_id}"
            cv2.putText(canvas, label, (x1, y1 - 4), FONT, 0.4, color, 1)

        # Flash event labels at top of frame
        for i, ev in enumerate(events):
            txt = ev.event_type.value.upper().replace("_", " ")
            cv2.putText(canvas, txt, (10, 28 + i * 22), FONT, 0.6, (0, 0, 255), 2)

        cv2.putText(
            canvas,
            f"frame {frame_idx}  H={'ok' if self._homography.is_valid else '--'}  "
            f"teams={'ok' if self._team_clf.is_fitted else '--'}",
            (10, canvas.shape[0] - 8), FONT, 0.38, (200, 200, 200), 1,
        )
        return canvas
