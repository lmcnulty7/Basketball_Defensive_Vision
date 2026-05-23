"""
scripts/demo_pipeline.py

End-to-end pipeline demo using a synthetic defensive sequence.

Because no NBA footage is available yet, this demo has two parts:

  Part 1 — Smoke test: Initialize the full PipelineRunner and process
            20 frames of bus.jpg through the tracking + ball detection
            loop.  Confirms all modules connect without errors.

  Part 2 — Stats demo: Inject a realistic 80-frame defensive sequence
            directly into the event detectors and stats aggregators.
            Shows the complete stats table output with:
              - 1 steal (possession changes from home → away at frame 15)
              - 1 contested 2PT shot (tight, frame 40)
              - 1 defensive rebound (frame 55)
              - matchup time and speed for 5 defenders
              - full aggregated stats table

Output:
  data/samples/pipeline_stats.png  — stats table as an annotated image

Run:
    python scripts/demo_pipeline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics.utils import ASSETS

from src.classification.team_classifier import TEAM_AWAY, TEAM_HOME
from src.court.court_model import BASKET_LEFT, BASKET_RIGHT, COURT_KEYPOINTS
from src.events.event import Event, EventType, PipelineState
from src.events.event_detector import EventOrchestrator
from src.events.steal_detector import StealDetector
from src.events.block_detector import BlockDetector
from src.events.deflection_detector import DeflectionDetector
from src.events.contest_detector import ContestDetector
from src.events.rebound_detector import ReboundDetector
from src.pipeline.runner import PipelineRunner
from src.pipeline.state import PipelineConfig
from src.stats.aggregator import StatsAggregator
from src.stats.matchup_tracker import MatchupTracker
from src.stats.speed_calculator import SpeedCalculator
from src.tracking.multi_tracker import BALL_TRACK_ID, Track
from src.tracking.track_manager import TrackManager

OUT_DIR = ROOT / "data" / "samples"
FONT    = cv2.FONT_HERSHEY_SIMPLEX


# ── Synthetic sequence helpers ────────────────────────────────────────────────

def make_track(tid, cx, cy, court_x, court_y, team_id, fi=0):
    t = Track(
        track_id=tid, bbox=np.array([cx-25,cy-60,cx+25,cy+60], dtype=np.float32),
        center=np.array([cx,cy], dtype=np.float32),
        confidence=0.9, class_id=0, frame_idx=fi, timestamp_sec=fi/30.0,
        team_id=team_id,
    )
    t.court_pos = np.array([court_x, court_y], dtype=np.float32)
    return t


def make_ball(px, py, court_x, court_y, fi=0):
    t = Track(
        track_id=BALL_TRACK_ID,
        bbox=np.array([px-8,py-8,px+8,py+8], dtype=np.float32),
        center=np.array([px,py], dtype=np.float32),
        confidence=0.8, class_id=BALL_TRACK_ID, frame_idx=fi, timestamp_sec=fi/30.0,
    )
    t.court_pos = np.array([court_x, court_y], dtype=np.float32)
    return t


def make_state(fi, possessor_id, team_assignments, player_tracks, ball, bgr):
    return PipelineState(
        frame_idx=fi, timestamp_sec=fi/30.0, raw_bgr=bgr,
        player_tracks=player_tracks, ball_track=ball,
        ball_is_predicted=False, ball_possessor_id=possessor_id,
        team_assignments=team_assignments,
    )


# ── Part 1: Smoke test ────────────────────────────────────────────────────────

def part1_smoke_test():
    print("\n[Part 1] Smoke test — initialize PipelineRunner + run 20 frames")
    print("─" * 60)

    config = PipelineConfig(
        min_samples_to_fit=999,   # prevent auto-fit during smoke test
        save_annotated_video=False,
    )
    runner = PipelineRunner(config)

    bgr = cv2.imread(str(ASSETS / "bus.jpg"))
    assert bgr is not None

    for i in range(20):
        events = runner.process_frame(bgr, frame_idx=i * 3, timestamp_sec=i * 0.1)

    out = runner.get_output(duration_sec=2.0)
    print(f"  Frames processed : {out.n_frames_processed}")
    print(f"  Tracks seen      : {runner._manager.n_active()}")
    print(f"  H valid          : {out.h_valid}")
    print(f"  Teams fitted     : {out.teams_fitted}")
    print(f"  Events fired     : {len(out.event_log)}")
    print("  ✓ All modules initialized and chained without error")


# ── Part 2: Stats demo ────────────────────────────────────────────────────────

def part2_stats_demo():
    print("\n[Part 2] Stats demo — 80-frame synthetic defensive sequence")
    print("─" * 60)

    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    manager = TrackManager(history_len=60)

    # 5 home (offense) + 5 away (defense) players
    # Court positions in feet (center-origin)
    HOME_PLAYERS = {
        1: (-20.0,  0.0),   # top of key
        2: (-35.0, -6.0),   # left block
        3: (-18.0, 22.0),   # corner
        4: (-22.0, 12.0),   # wing
        5: (-8.0,   0.0),   # above break
    }
    AWAY_PLAYERS = {
        11: (-22.0,  0.5),   # guarding 1
        12: (-33.0, -6.0),   # guarding 2
        13: (-19.0, 22.5),   # guarding 3 (contested corner shot)
        14: (-24.0, 12.0),   # guarding 4
        15: (-10.0,  0.5),   # guarding 5 (stealer)
    }
    TEAM_ASSIGN = {**{k: TEAM_HOME for k in HOME_PLAYERS},
                   **{k: TEAM_AWAY for k in AWAY_PLAYERS}}

    steal_det   = StealDetector(confirm_frames=3, cooldown_frames=20)
    contest_det = ContestDetector(min_possession_frames=3, cooldown_frames=30)
    rebound_det = ReboundDetector(basket_radius_ft=12.0, cooldown_frames=10)
    contest_det.set_steal_detector(steal_det)

    orch = EventOrchestrator([steal_det, BlockDetector(), DeflectionDetector(),
                              contest_det, rebound_det])
    mt   = MatchupTracker(stride=3, fps=30.0)
    sp   = SpeedCalculator(stride=3, fps=30.0)

    # ── Sequence ──────────────────────────────────────────────────────────────
    # Frames 0-14:  Home player 1 has ball
    # Frames 15-22: Steal — away player 15 takes from home player 1
    # Frames 23-38: Away player 12 driving toward basket (home on defense now)
    # Frames 39-43: Shot attempt by player 12, player 2 (defender) is 2ft away
    # Frames 44-54: Ball near basket
    # Frames 55-65: Home player 2 gains rebound (defensive rebound for them)

    def build_frame(fi, possessor_id, ball_court, ball_px):
        player_tracks = []
        for pid, (cx, cy) in HOME_PLAYERS.items():
            # Slightly move players each frame
            drift_x = cx + fi * 0.01
            player_tracks.append(make_track(
                pid, cx=300+pid*40, cy=240, court_x=drift_x, court_y=cy,
                team_id=TEAM_HOME, fi=fi,
            ))
        for pid, (cx, cy) in AWAY_PLAYERS.items():
            drift_x = cx + fi * 0.008
            player_tracks.append(make_track(
                pid, cx=200+pid*30, cy=200, court_x=drift_x, court_y=cy,
                team_id=TEAM_AWAY, fi=fi,
            ))
        ball = make_ball(ball_px[0], ball_px[1], ball_court[0], ball_court[1], fi)
        state = make_state(fi, possessor_id, TEAM_ASSIGN, player_tracks, ball, bgr)
        # Update manager with court positions
        manager.update(player_tracks, ball, fi, fi/30.0)
        for t in player_tracks:
            if t.team_id is not None:
                manager.update_team_id(t.track_id, t.team_id)
            if t.court_pos is not None:
                manager.update_court_pos(t.track_id, t.court_pos)
        return state

    events_per_frame = []
    for fi in range(80):
        if fi < 15:
            poss = 1; ball_court = (-19.0, 0.0); ball_px = (320, 240)
        elif fi < 23:
            poss = 15; ball_court = (-19.0, 0.5); ball_px = (310, 235)  # steal window
        elif fi < 39:
            poss = None; ball_court = (-32.0, -5.0); ball_px = (290, 300)  # driving
        elif fi < 44:
            # Shot attempt: ball accelerating away from player 12
            poss = None
            ball_court = (-38.0 + (fi-39)*2, -5.0)   # moving toward basket
            ball_px    = (280 - (fi-39)*5, 300 - (fi-39)*15)  # rising in pixel space
        elif fi < 55:
            poss = None; ball_court = (float(BASKET_LEFT[0])+3, 0.0); ball_px = (100, 200)
        else:
            poss = 2; ball_court = (float(BASKET_LEFT[0])+3, 0.0); ball_px = (100, 220)

        state = build_frame(fi, poss, ball_court, ball_px)
        fired = orch.update(state, manager)
        events_per_frame.append(fired)

        # Update matchup + speed stats
        off_pos = {t.track_id: t.court_pos for t in state.get_offensive_tracks()
                   if t.court_pos is not None}
        def_pos = {t.track_id: t.court_pos for t in state.get_defensive_tracks()
                   if t.court_pos is not None}
        matchups = mt.update(off_pos, def_pos)
        sp.update(manager, matchups, def_pos, off_pos)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    all_def_ids = list(AWAY_PLAYERS.keys())
    agg    = StatsAggregator()
    stats  = agg.compute(orch.event_log, mt, sp, track_ids=all_def_ids)
    table  = agg.to_table(stats)

    print(f"\n  Events fired: {orch.summary()}")
    print(f"\n  Per-defender stats:")
    print(f"  {'ID':>4}  {'Steals':>6}  {'Blocks':>6}  {'Defl':>5}  "
          f"{'Cont2':>6}  {'Cont3':>6}  {'DefReb':>7}  "
          f"{'Matchup(s)':>10}  {'DefMPH':>7}")
    print("  " + "─" * 68)
    for row in table:
        print(f"  {row['track_id']:>4}  {row['steals']:>6}  {row['blocks']:>6}  "
              f"{row['deflections']:>5}  {row['contested_2pt']:>6}  "
              f"{row['contested_3pt']:>6}  {row['def_rebounds']:>7}  "
              f"{row['matchup_time_sec']:>10.1f}  {row['def_speed_mph']:>7.2f}")

    # ── Save stats as image ───────────────────────────────────────────────────
    img = np.full((420, 860, 3), 30, dtype=np.uint8)
    cv2.putText(img, "Basketball Defensive Vision — Stats Output",
                (20, 35), FONT, 0.7, (255,255,255), 2)
    cv2.putText(img, f"Events: {orch.summary()}",
                (20, 65), FONT, 0.45, (200,200,200), 1)

    headers = ["ID","Steals","Blocks","Defl","Cont2","Cont3","DefReb","Matchup(s)","DefMPH"]
    col_x   = [20, 80, 160, 240, 300, 370, 440, 520, 640]
    y = 100
    for h, x in zip(headers, col_x):
        cv2.putText(img, h, (x, y), FONT, 0.40, (0, 200, 255), 1)
    cv2.line(img, (15, y+8), (845, y+8), (80,80,80), 1)

    for row in table:
        y += 30
        vals = [str(row['track_id']), str(row['steals']), str(row['blocks']),
                str(row['deflections']), str(row['contested_2pt']),
                str(row['contested_3pt']), str(row['def_rebounds']),
                f"{row['matchup_time_sec']:.1f}", f"{row['def_speed_mph']:.2f}"]
        for val, x in zip(vals, col_x):
            color = (0,220,0) if any(int(v) > 0 for v in [row['steals'],row['blocks'],
                     row['deflections'],row['def_rebounds']] if isinstance(v, int)) \
                     and val != "0" else (180,180,180)
            cv2.putText(img, val, (x, y), FONT, 0.38, color, 1)

    # Event timeline at bottom
    y = 310
    cv2.putText(img, "Event Timeline:", (20, y), FONT, 0.45, (200,200,200), 1)
    y += 25
    for ev in orch.event_log:
        txt = (f"t={ev.timestamp_sec:.2f}s  {ev.event_type.value:<20}  "
               f"defender=#{ev.primary_player_id}  "
               f"conf={ev.confidence:.2f}  {ev.metadata}")
        cv2.putText(img, txt, (20, y), FONT, 0.33, (255,200,0), 1)
        y += 18
        if y > 400:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "pipeline_stats.png"
    cv2.imwrite(str(out_path), img)
    print(f"\n  Stats image → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Basketball Defensive Vision — Full Pipeline Demo")
    print("=" * 60)

    part1_smoke_test()
    part2_stats_demo()

    print("\n" + "=" * 60)
    print("To run on real footage:")
    print("  python scripts/run_pipeline.py --clip data/raw/game.mp4")
    print("=" * 60)


if __name__ == "__main__":
    main()
