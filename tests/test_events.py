"""
tests/test_events.py

Unit tests for src/events/ and src/stats/.

All tests use synthetic PipelineState sequences — no video or YOLO needed.

Synthetic sequence helpers build sequences of PipelineState frames that
trigger (or don't trigger) specific detectors, letting us verify each
detector's logic precisely.

Run with:  pytest tests/test_events.py -v
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pytest

from src.classification.team_classifier import TEAM_HOME, TEAM_AWAY, TEAM_REFEREE
from src.court.court_model import BASKET_LEFT, BASKET_RIGHT
from src.events.event import Event, EventType, PipelineState
from src.events.event_detector import EventOrchestrator
from src.events.steal_detector import StealDetector
from src.events.block_detector import BlockDetector
from src.events.deflection_detector import DeflectionDetector
from src.events.contest_detector import ContestDetector
from src.events.rebound_detector import ReboundDetector
from src.stats.matchup_tracker import MatchupTracker
from src.stats.speed_calculator import SpeedCalculator
from src.stats.aggregator import StatsAggregator
from src.tracking.multi_tracker import BALL_TRACK_ID, Track
from src.tracking.track_manager import TrackManager


# ── State / track builders ────────────────────────────────────────────────────

def make_track(
    tid=1, cx=0.0, cy=0.0, court_x=0.0, court_y=0.0,
    team_id=TEAM_HOME, frame_idx=0,
    bbox=None,
) -> Track:
    if bbox is None:
        bbox = np.array([cx-25, cy-60, cx+25, cy+60], dtype=np.float32)
    t = Track(
        track_id=tid,
        bbox=bbox,
        center=np.array([cx, cy], dtype=np.float32),
        confidence=0.9,
        class_id=0,
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        team_id=team_id,
    )
    t.court_pos = np.array([court_x, court_y], dtype=np.float32)
    return t


def make_ball(px=320.0, py=240.0, court_x=-20.0, court_y=0.0, frame_idx=0) -> Track:
    t = Track(
        track_id=BALL_TRACK_ID,
        bbox=np.array([px-10, py-10, px+10, py+10], dtype=np.float32),
        center=np.array([px, py], dtype=np.float32),
        confidence=0.8,
        class_id=BALL_TRACK_ID,
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
    )
    t.court_pos = np.array([court_x, court_y], dtype=np.float32)
    return t


def make_state(
    frame_idx=0,
    possessor_id=None,
    team_assignments=None,
    player_tracks=None,
    ball_px=(320, 240),
    ball_court=(-20.0, 0.0),
    ball_is_predicted=False,
    raw_bgr_shape=(480, 640, 3),
) -> PipelineState:
    bgr = np.zeros(raw_bgr_shape, dtype=np.uint8)
    ball = make_ball(
        px=ball_px[0], py=ball_px[1],
        court_x=ball_court[0], court_y=ball_court[1],
        frame_idx=frame_idx,
    )
    return PipelineState(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        raw_bgr=bgr,
        player_tracks=player_tracks or [],
        ball_track=ball,
        ball_is_predicted=ball_is_predicted,
        ball_possessor_id=possessor_id,
        team_assignments=team_assignments or {},
    )


def make_manager() -> TrackManager:
    return TrackManager(history_len=30)


# ── Event dataclass ───────────────────────────────────────────────────────────

class TestEvent:
    def test_to_dict_keys(self):
        e = Event(
            event_type=EventType.STEAL,
            frame_idx=10,
            timestamp_sec=0.333,
            primary_player_id=5,
        )
        d = e.to_dict()
        assert "event_type" in d and "primary_player_id" in d

    def test_to_dict_event_type_is_string(self):
        e = Event(EventType.BLOCK, 0, 0.0, 1)
        assert isinstance(e.to_dict()["event_type"], str)

    def test_court_pos_serialized(self):
        e = Event(EventType.DEFLECTION, 0, 0.0, 1,
                  court_pos=np.array([5.0, -3.0]))
        assert isinstance(e.to_dict()["court_pos"], list)


# ── PipelineState ─────────────────────────────────────────────────────────────

class TestPipelineState:
    def test_offensive_team_from_possessor(self):
        state = make_state(
            possessor_id=1,
            team_assignments={1: TEAM_HOME, 2: TEAM_AWAY},
            player_tracks=[make_track(1, team_id=TEAM_HOME),
                           make_track(2, team_id=TEAM_AWAY)],
        )
        assert state.offensive_team_id == TEAM_HOME
        assert state.defensive_team_id == TEAM_AWAY

    def test_no_possessor_gives_none(self):
        state = make_state(possessor_id=None)
        assert state.offensive_team_id is None
        assert state.defensive_team_id is None

    def test_get_offensive_tracks(self):
        state = make_state(
            possessor_id=1,
            team_assignments={1: TEAM_HOME, 2: TEAM_HOME, 3: TEAM_AWAY},
            player_tracks=[
                make_track(1, team_id=TEAM_HOME),
                make_track(2, team_id=TEAM_HOME),
                make_track(3, team_id=TEAM_AWAY),
            ],
        )
        off = state.get_offensive_tracks()
        assert len(off) == 2
        assert all(t.track_id in (1, 2) for t in off)

    def test_get_defensive_tracks(self):
        state = make_state(
            possessor_id=1,
            team_assignments={1: TEAM_HOME, 3: TEAM_AWAY},
            player_tracks=[
                make_track(1, team_id=TEAM_HOME),
                make_track(3, team_id=TEAM_AWAY),
            ],
        )
        def_ = state.get_defensive_tracks()
        assert len(def_) == 1
        assert def_[0].track_id == 3

    def test_nearest_defender_to(self):
        state = make_state(
            possessor_id=1,
            team_assignments={1: TEAM_HOME, 3: TEAM_AWAY, 4: TEAM_AWAY},
            player_tracks=[
                make_track(1, court_x=-20.0, court_y=0.0, team_id=TEAM_HOME),
                make_track(3, court_x=-22.0, court_y=0.0, team_id=TEAM_AWAY),  # 2ft away
                make_track(4, court_x=-30.0, court_y=0.0, team_id=TEAM_AWAY),  # 10ft
            ],
        )
        pos = np.array([-20.0, 0.0])
        track, dist = state.nearest_defender_to(pos)
        assert track.track_id == 3
        assert dist == pytest.approx(2.0, abs=0.1)

    def test_ball_court_pos_from_track(self):
        state = make_state(ball_court=(-15.0, 5.0))
        pos = state.ball_court_pos
        assert pos[0] == pytest.approx(-15.0)
        assert pos[1] == pytest.approx(5.0)


# ── StealDetector ─────────────────────────────────────────────────────────────

class TestStealDetector:
    def _run_possession_sequence(self, team_seq, confirm=3):
        """Feed a sequence of (possessor_id, team_id) tuples through the detector."""
        det = StealDetector(confirm_frames=confirm, cooldown_frames=5)
        manager = make_manager()
        all_events = []
        for i, (pid, team) in enumerate(team_seq):
            state = make_state(
                frame_idx=i,
                possessor_id=pid,
                team_assignments={pid: team} if pid is not None else {},
            )
            all_events.extend(det.update(state, manager))
        return all_events

    def test_no_steal_same_team(self):
        seq = [(1, TEAM_HOME)] * 10
        assert self._run_possession_sequence(seq) == []

    def test_steal_fires_on_team_change(self):
        # 3 frames team_home, then 3 frames team_away
        seq = ([(1, TEAM_HOME)] * 3) + ([(2, TEAM_AWAY)] * 3)
        events = self._run_possession_sequence(seq)
        steals = [e for e in events if e.event_type == EventType.STEAL]
        assert len(steals) >= 1

    def test_steal_credits_correct_defender(self):
        seq = ([(1, TEAM_HOME)] * 3) + ([(2, TEAM_AWAY)] * 3)
        events = self._run_possession_sequence(seq)
        steals = [e for e in events if e.event_type == EventType.STEAL]
        assert steals[0].primary_player_id == 2
        assert steals[0].secondary_player_id == 1

    def test_no_steal_during_cooldown(self):
        det = StealDetector(confirm_frames=2, cooldown_frames=20)
        manager = make_manager()
        events = []
        # First possession change
        for i, (pid, team) in enumerate([(1, TEAM_HOME)]*2 + [(2, TEAM_AWAY)]*2):
            events.extend(det.update(make_state(frame_idx=i, possessor_id=pid,
                                                team_assignments={pid: team}), manager))
        first_steal = [e for e in events if e.event_type == EventType.STEAL]
        assert len(first_steal) == 1
        # Immediate second change within cooldown
        for i in range(4, 8):
            pid, team = (1, TEAM_HOME) if i < 6 else (2, TEAM_AWAY)
            events.extend(det.update(make_state(frame_idx=i, possessor_id=pid,
                                                team_assignments={pid: team}), manager))
        all_steals = [e for e in events if e.event_type == EventType.STEAL]
        assert len(all_steals) == 1  # cooldown suppressed second

    def test_shot_notification_suppresses_steal(self):
        det = StealDetector(confirm_frames=2, cooldown_frames=5, shot_clearance_frames=10)
        det.notify_shot(frame_idx=3)
        manager = make_manager()
        events = []
        for i, (pid, team) in enumerate([(1, TEAM_HOME)]*2 + [(2, TEAM_AWAY)]*2):
            events.extend(det.update(make_state(frame_idx=i, possessor_id=pid,
                                                team_assignments={pid: team}), manager))
        steals = [e for e in events if e.event_type == EventType.STEAL]
        assert len(steals) == 0  # shot clearance suppressed

    def test_no_steal_without_enough_history(self):
        det = StealDetector(confirm_frames=5)
        manager = make_manager()
        # Only 2 frames of each team — not enough to confirm
        events = []
        for i, (pid, team) in enumerate([(1, TEAM_HOME)]*2 + [(2, TEAM_AWAY)]*2):
            events.extend(det.update(make_state(frame_idx=i, possessor_id=pid,
                                                team_assignments={pid: team}), manager))
        assert events == []

    def test_reset_clears_buffer(self):
        det = StealDetector(confirm_frames=2)
        manager = make_manager()
        for i in range(3):
            det.update(make_state(frame_idx=i, possessor_id=1,
                                  team_assignments={1: TEAM_HOME}), manager)
        det.reset()
        assert len(det._buffer) == 0


# ── BlockDetector ─────────────────────────────────────────────────────────────

class TestBlockDetector:
    def _run_sequence(self, ball_py_seq, defender_overlap=False, confirm=True):
        det = BlockDetector(min_rise_speed_px=3.0, dir_change_thresh_deg=50.0,
                            confirm_window=5, cooldown_frames=5)
        manager = make_manager()
        events = []
        for i, py in enumerate(ball_py_seq):
            # Defender placed to overlap ball when defender_overlap=True
            defender_bbox = np.array([290, py-40, 350, py+60], dtype=np.float32) \
                if defender_overlap else np.array([600, 400, 650, 500], dtype=np.float32)
            defender = make_track(10, cx=320, cy=py, bbox=defender_bbox, team_id=TEAM_AWAY)
            state = make_state(
                frame_idx=i,
                ball_px=(320, py),
                possessor_id=1,
                team_assignments={1: TEAM_HOME, 10: TEAM_AWAY},
                player_tracks=[make_track(1, cx=320, cy=py, team_id=TEAM_HOME), defender],
            )
            events.extend(det.update(state, manager))
        return events

    def test_block_fires_on_rising_ball_with_overlap_and_direction_change(self):
        # Ball rises (py decreasing) then abruptly changes (py increases = direction change)
        seq = [300, 280, 260, 240, 260, 280]  # rise then fall = direction change
        events = self._run_sequence(seq, defender_overlap=True)
        blocks = [e for e in events if e.event_type == EventType.BLOCK]
        assert len(blocks) >= 1

    def test_no_block_without_overlap(self):
        seq = [300, 280, 260, 240, 260, 280]
        events = self._run_sequence(seq, defender_overlap=False)
        blocks = [e for e in events if e.event_type == EventType.BLOCK]
        assert len(blocks) == 0

    def test_no_block_when_ball_not_rising(self):
        seq = [200, 220, 240, 260, 280]  # falling (py increasing) = not rising
        events = self._run_sequence(seq, defender_overlap=True)
        blocks = [e for e in events if e.event_type == EventType.BLOCK]
        assert len(blocks) == 0

    def test_block_credits_defender(self):
        seq = [300, 280, 260, 240, 260, 280]
        events = self._run_sequence(seq, defender_overlap=True)
        blocks = [e for e in events if e.event_type == EventType.BLOCK]
        if blocks:
            assert blocks[0].primary_player_id == 10

    def test_reset_clears_pending(self):
        det = BlockDetector()
        det._pending = (5, 10)
        det.reset()
        assert det._pending is None


# ── DeflectionDetector ────────────────────────────────────────────────────────

class TestDeflectionDetector:
    def _build_state_with_defender(self, ball_px, ball_court, def_court, frame_idx=0):
        defender = make_track(10, cx=400, cy=300, court_x=def_court[0],
                              court_y=def_court[1], team_id=TEAM_AWAY)
        return make_state(
            frame_idx=frame_idx,
            ball_px=ball_px,
            ball_court=ball_court,
            possessor_id=1,
            team_assignments={1: TEAM_HOME, 10: TEAM_AWAY},
            player_tracks=[make_track(1, team_id=TEAM_HOME), defender],
        )

    def test_deflection_fires_on_sharp_direction_change(self):
        det = DeflectionDetector(angle_thresh_deg=40, min_ball_speed_px=5,
                                  max_defender_dist_ft=5.0, cooldown_frames=3)
        manager = make_manager()
        # Ball moving right fast, then sharp turn left
        frames = [
            (100, 240), (120, 240), (140, 240),   # moving right
            (120, 240),                             # sharp reversal
        ]
        events = []
        for i, (px, py) in enumerate(frames):
            state = self._build_state_with_defender(
                (px, py), (-20.0, 0.0), (-21.0, 0.5), frame_idx=i
            )
            events.extend(det.update(state, manager))
        deflections = [e for e in events if e.event_type == EventType.DEFLECTION]
        assert len(deflections) >= 1

    def test_no_deflection_without_nearby_defender(self):
        det = DeflectionDetector(angle_thresh_deg=40, min_ball_speed_px=5,
                                  max_defender_dist_ft=3.0, cooldown_frames=3)
        manager = make_manager()
        frames = [(100,240),(120,240),(140,240),(120,240)]
        events = []
        for i, (px, py) in enumerate(frames):
            state = self._build_state_with_defender(
                (px, py), (-20.0, 0.0), (30.0, 0.0), frame_idx=i  # far defender
            )
            events.extend(det.update(state, manager))
        deflections = [e for e in events if e.event_type == EventType.DEFLECTION]
        assert len(deflections) == 0

    def test_no_deflection_below_min_speed(self):
        det = DeflectionDetector(angle_thresh_deg=40, min_ball_speed_px=50,
                                  cooldown_frames=1)
        manager = make_manager()
        frames = [(100,240),(101,240),(102,240),(101,240)]  # slow ball
        events = []
        for i, (px, py) in enumerate(frames):
            state = self._build_state_with_defender((px,py), (-20.0,0.0), (-21.0,0.5), i)
            events.extend(det.update(state, manager))
        assert all(e.event_type != EventType.DEFLECTION for e in events)

    def test_deflection_metadata_has_angle(self):
        det = DeflectionDetector(angle_thresh_deg=40, min_ball_speed_px=5,
                                  max_defender_dist_ft=5.0, cooldown_frames=3)
        manager = make_manager()
        frames = [(100,240),(120,240),(140,240),(120,240)]
        events = []
        for i, (px, py) in enumerate(frames):
            state = self._build_state_with_defender((px,py), (-20.0,0.0), (-21.0,0.5), i)
            events.extend(det.update(state, manager))
        deflections = [e for e in events if e.event_type == EventType.DEFLECTION]
        if deflections:
            assert "angle_deg" in deflections[0].metadata
            assert "dist_ft" in deflections[0].metadata

    def test_reset_clears_history(self):
        det = DeflectionDetector()
        det._ball_px.extend([np.array([100.0, 200.0])] * 5)
        det.reset()
        assert len(det._ball_px) == 0


# ── ReboundDetector ───────────────────────────────────────────────────────────

class TestReboundDetector:
    def test_defensive_rebound_after_shot_notification(self):
        det = ReboundDetector(basket_radius_ft=15.0, cooldown_frames=3)
        manager = make_manager()

        # Notify shot (shooting team = TEAM_HOME)
        det.notify_shot(frame_idx=0, shooting_team=TEAM_HOME)

        # Ball near basket
        defender = make_track(10, court_x=BASKET_LEFT[0]+3, court_y=0.0,
                              team_id=TEAM_AWAY)
        events = []
        for i in range(1, 6):
            near_basket = (float(BASKET_LEFT[0]) + 2.0, 0.0)
            state = make_state(
                frame_idx=i,
                ball_court=near_basket,
                # possessor appears at frame 3
                possessor_id=(10 if i >= 3 else None),
                team_assignments={10: TEAM_AWAY},
                player_tracks=[defender],
            )
            events.extend(det.update(state, manager))

        rebounds = [e for e in events if e.event_type == EventType.REBOUND_DEF]
        assert len(rebounds) >= 1

    def test_offensive_rebound_when_same_team_recovers(self):
        det = ReboundDetector(basket_radius_ft=15.0, cooldown_frames=3)
        manager = make_manager()
        det.notify_shot(frame_idx=0, shooting_team=TEAM_HOME)

        # Offensive player (TEAM_HOME) gets ball
        off_player = make_track(5, court_x=BASKET_LEFT[0]+2, court_y=0.0,
                                team_id=TEAM_HOME)
        events = []
        for i in range(1, 5):
            state = make_state(
                frame_idx=i,
                ball_court=(float(BASKET_LEFT[0])+2, 0.0),
                possessor_id=(5 if i >= 3 else None),
                team_assignments={5: TEAM_HOME},
                player_tracks=[off_player],
            )
            events.extend(det.update(state, manager))

        off_reb = [e for e in events if e.event_type == EventType.REBOUND_OFF]
        assert len(off_reb) >= 1

    def test_no_rebound_without_shot(self):
        det = ReboundDetector(basket_radius_ft=15.0, cooldown_frames=3)
        manager = make_manager()
        # No shot notification
        defender = make_track(10, court_x=BASKET_LEFT[0]+2, court_y=0.0,
                              team_id=TEAM_AWAY)
        events = []
        for i in range(5):
            state = make_state(
                frame_idx=i,
                ball_court=(float(BASKET_LEFT[0])+2, 0.0),
                possessor_id=(10 if i >= 3 else None),
                team_assignments={10: TEAM_AWAY},
                player_tracks=[defender],
            )
            events.extend(det.update(state, manager))
        assert len(events) == 0

    def test_reset_clears_shot_state(self):
        det = ReboundDetector()
        det.notify_shot(5, TEAM_HOME)
        det.reset()
        assert det._shot_frame is None


# ── EventOrchestrator ─────────────────────────────────────────────────────────

class TestEventOrchestrator:
    def test_collects_events_from_all_detectors(self):
        steal_det = StealDetector(confirm_frames=2, cooldown_frames=5)
        orch = EventOrchestrator([steal_det])
        manager = make_manager()
        # Trigger a steal
        seq = [(1, TEAM_HOME)]*2 + [(2, TEAM_AWAY)]*2
        for i, (pid, team) in enumerate(seq):
            state = make_state(frame_idx=i, possessor_id=pid,
                               team_assignments={pid: team})
            orch.update(state, manager)
        steals = orch.get_events_by_type(EventType.STEAL)
        assert len(steals) >= 1

    def test_get_events_for_player(self):
        steal_det = StealDetector(confirm_frames=2, cooldown_frames=3)
        orch = EventOrchestrator([steal_det])
        manager = make_manager()
        seq = [(1, TEAM_HOME)]*2 + [(2, TEAM_AWAY)]*2
        for i, (pid, team) in enumerate(seq):
            orch.update(make_state(frame_idx=i, possessor_id=pid,
                                   team_assignments={pid: team}), manager)
        player_events = orch.get_events_for_player(2)
        assert all(e.primary_player_id == 2 for e in player_events)

    def test_summary_counts(self):
        steal_det = StealDetector(confirm_frames=2, cooldown_frames=3)
        orch = EventOrchestrator([steal_det])
        manager = make_manager()
        seq = [(1, TEAM_HOME)]*2 + [(2, TEAM_AWAY)]*2
        for i, (pid, team) in enumerate(seq):
            orch.update(make_state(frame_idx=i, possessor_id=pid,
                                   team_assignments={pid: team}), manager)
        s = orch.summary()
        assert s.get("steal", 0) >= 1

    def test_reset_clears_log(self):
        steal_det = StealDetector(confirm_frames=2, cooldown_frames=3)
        orch = EventOrchestrator([steal_det])
        manager = make_manager()
        for i, (pid, team) in enumerate([(1, TEAM_HOME)]*2 + [(2, TEAM_AWAY)]*2):
            orch.update(make_state(frame_idx=i, possessor_id=pid,
                                   team_assignments={pid: team}), manager)
        orch.reset()
        assert orch.event_log == []

    def test_dedup_window_prevents_double_fire(self):
        steal_det = StealDetector(confirm_frames=1, cooldown_frames=1)
        orch = EventOrchestrator([steal_det], dedup_window_frames=10)
        manager = make_manager()
        seq = [(1, TEAM_HOME)] + [(2, TEAM_AWAY)] * 5
        for i, (pid, team) in enumerate(seq):
            orch.update(make_state(frame_idx=i, possessor_id=pid,
                                   team_assignments={pid: team}), manager)
        steals = orch.get_events_by_type(EventType.STEAL)
        assert len(steals) <= 1  # dedup prevents multiple fires


# ── MatchupTracker ────────────────────────────────────────────────────────────

class TestMatchupTracker:
    def test_basic_assignment(self):
        mt = MatchupTracker(max_matchup_dist_ft=10.0)
        off = {1: np.array([-20.0, 0.0]), 2: np.array([-20.0, 10.0])}
        def_ = {3: np.array([-22.0, 0.0]), 4: np.array([-22.0, 10.0])}
        result = mt.update(off, def_)
        assert result[3] == 1   # defender 3 closest to offensive 1
        assert result[4] == 2

    def test_no_matchup_when_too_far(self):
        mt = MatchupTracker(max_matchup_dist_ft=5.0)
        off = {1: np.array([0.0, 0.0])}
        def_ = {2: np.array([20.0, 0.0])}  # 20 ft away > 5 ft threshold
        result = mt.update(off, def_)
        assert result[2] is None

    def test_accumulates_seconds(self):
        mt = MatchupTracker(max_matchup_dist_ft=10.0, stride=3, fps=30.0)
        off = {1: np.array([-20.0, 0.0])}
        def_ = {2: np.array([-22.0, 0.0])}
        for _ in range(10):
            mt.update(off, def_)
        secs = mt.get_total_matchup_seconds(2)
        assert secs == pytest.approx(10 * (3/30.0), rel=0.01)

    def test_primary_matchup(self):
        mt = MatchupTracker(max_matchup_dist_ft=15.0)
        # Defender 3 guards offensive 1 for most frames, 2 briefly
        off1 = {1: np.array([-20.0, 0.0])}
        off2 = {2: np.array([-20.0, 0.0])}
        def_ = {3: np.array([-22.0, 0.0])}
        for _ in range(8):
            mt.update(off1, def_)
        for _ in range(2):
            mt.update(off2, def_)
        assert mt.get_primary_matchup(3) == 1

    def test_empty_input_returns_none_assignments(self):
        mt = MatchupTracker()
        result = mt.update({}, {2: np.array([-22.0, 0.0])})
        # No offensive players → defender 2 has no matchup (None)
        assert result == {2: None}

    def test_reset_clears_data(self):
        mt = MatchupTracker(max_matchup_dist_ft=10.0)
        mt.update({1: np.array([0.0, 0.0])}, {2: np.array([1.0, 0.0])})
        mt.reset()
        assert mt.get_total_matchup_seconds(2) == 0.0


# ── StatsAggregator ───────────────────────────────────────────────────────────

class TestStatsAggregator:
    def _make_event(self, event_type, pid, meta=None):
        return Event(event_type=event_type, frame_idx=0, timestamp_sec=0.0,
                     primary_player_id=pid, metadata=meta or {})

    def test_counts_steals(self):
        agg = StatsAggregator()
        events = [self._make_event(EventType.STEAL, 5)] * 3
        mt = MatchupTracker(); sp = SpeedCalculator()
        stats = agg.compute(events, mt, sp, [5])
        assert stats[5]["steals"] == 3

    def test_counts_blocks(self):
        agg = StatsAggregator()
        events = [self._make_event(EventType.BLOCK, 7)] * 2
        mt = MatchupTracker(); sp = SpeedCalculator()
        stats = agg.compute(events, mt, sp, [7])
        assert stats[7]["blocks"] == 2

    def test_tight_contest_pct(self):
        agg = StatsAggregator()
        events = [
            self._make_event(EventType.CONTESTED_2PT, 3, {"contest_level": "tight"}),
            self._make_event(EventType.CONTESTED_2PT, 3, {"contest_level": "open"}),
        ]
        mt = MatchupTracker(); sp = SpeedCalculator()
        stats = agg.compute(events, mt, sp, [3])
        assert stats[3]["contest_pct_tight"] == pytest.approx(0.5)

    def test_zero_rows_for_tracks_with_no_events(self):
        agg = StatsAggregator()
        mt = MatchupTracker(); sp = SpeedCalculator()
        stats = agg.compute([], mt, sp, [1, 2, 3])
        assert all(stats[tid]["steals"] == 0 for tid in [1, 2, 3])

    def test_to_table_sorted(self):
        agg = StatsAggregator()
        events = [
            self._make_event(EventType.STEAL, 1),
            self._make_event(EventType.STEAL, 1),
            self._make_event(EventType.BLOCK, 2),
        ]
        mt = MatchupTracker(); sp = SpeedCalculator()
        stats = agg.compute(events, mt, sp, [1, 2])
        table = agg.to_table(stats)
        assert table[0]["track_id"] == 1  # 2 steals > 1 block
