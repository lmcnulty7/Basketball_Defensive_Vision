# Basketball Defensive Vision — Full Design & Roadmap

## What This System Does

A computer vision pipeline that reads NBA broadcast video and outputs a full box score —
points, FGM/A, assists, steals, blocks, rebounds, turnovers, contested shots, matchup time,
defensive speed — for every visible player. No proprietary data feeds, no wearables.
Just a `.mp4` and a Mac.

Outputs feed a FastAPI + React dashboard: leaderboard, event feed, shot chart, box score.

---

## Table of Contents

1. [Architecture](#architecture)
2. [What's Working](#whats-working)
3. [What's Broken and Why](#whats-broken-and-why)
4. [Data Flow & Ground Truth](#data-flow--ground-truth)
5. [Validation Targets](#validation-targets)
6. [Phased Roadmap](#phased-roadmap)
7. [Critical Path](#critical-path)

---

## Architecture

Every frame passes through these stages in sequence:

```
Court gate (HSV floor %)
  → Player detection (YOLOv8m)
  → Ball detection (ball_yolo.pt → HSV fallback → Kalman filter)
  → Team classification (SigLIP + UMAP + K-means)
  → Jersey reading (EasyOCR, every 90 frames per track)
  → Player identity (jersey# + team → name via roster DB)
  → Court homography (14 keypoints → H matrix → feet coordinates)
  → Possession state machine (basket assignment, halftime, camera cuts)
  → Event detectors (8 detectors in EventOrchestrator)
  → Matchup + speed tracking
  → Stats aggregation → FastAPI → React
```

`PipelineState` is the central data bus, built fresh each frame and passed to every
event detector. Detectors are stateless with respect to each other — easy to unit test
and swap out independently.

### Event Detectors

| Detector | Pattern | Key Inputs |
|---|---|---|
| StealDetector | Possession flip between teams | possession state, team assignments |
| TurnoverDetector | Non-steal possession change | possession state, ball court position |
| BlockDetector | Rising ball + defender overlap → direction change | ball pixel velocity, player bboxes |
| DeflectionDetector | Ball trajectory change near player | ball history, player bboxes |
| ContestDetector | Defender proximity to shooter | player court positions, shot detector state |
| ReboundDetector | Ball lost near basket → player gains possession | ball track, basket position |
| ShotDetector | Ball velocity toward basket + rising → outcome | ball history, possession, basket direction |
| AssistDetector | Last passer before shot_made | possession history, shot detector state |

### Shot / Block two-phase pattern

Both shot and block detection use a two-phase confirm window to suppress false positives:

- **Phase 1 (candidate):** signal detected (ball rising toward basket / defender overlap)
- **Phase 2 (confirm):** within N frames, confirm the outcome (ball lost near net = made; rebound fires = miss; direction change = block)
- If the confirm window expires without a signal → event discarded

---

## What's Working

| Component | Status | Detail |
|---|---|---|
| Court visibility gate | ✓ | HSV floor color 12–55%, skips ads/replays/close-ups |
| Player detection | ✓ | YOLOv8m, reliable at broadcast distance |
| Multi-object tracking | ✓ | BoT-SORT, camera cut detection (>80% track loss in one frame) |
| Court bbox detector | ✓ | mAP50=0.91, YOLOv8n fine-tuned on court_kp_dataset |
| Court homography | ✓ | H matrix per frame via 14 keypoints, recomputes on camera cut |
| Team classification | ✓ | SigLIP + UMAP + K-means, ~50-frame cold start |
| Jersey reading | ✓ | EasyOCR, every 90 frames per track |
| Player identity | ✓ | Roster DB: 22 teams, 11 games in data/pbp.db |
| Possession tracker | ✓ | Basket assignment + halftime flip detection |
| Steal detector | ✓ | P=0.80, R=0.32, F1=0.46 (ECF 2012 G6) |
| Turnover detector | ✓ | P=0.50, R=0.33, F1=0.40 |
| Stats aggregator | ✓ | Full box score schema (off + def + movement) |
| FastAPI backend | ✓ | 6 endpoints: /stats /events /clips /leaderboard /boxscore /shotchart |
| Validation framework | ✓ | Auto time-alignment vs PBP ground truth, P/R/F1 per event type |
| PBP + roster DB | ✓ | 11 games, 406–455 events each, 22 team rosters |

---

## What's Broken and Why

### 1. Ball Detector — the system-wide blocker

**Impact:** Shot, rebound, and assist detection are all zero. They require a continuous,
reliable ball track. Until the ball detector is fixed, roughly half the box score is dead.

The current `ball_yolo.pt` has **mAP50=0.04** — essentially random. Three compounding problems:

**Problem A — Training data bias:**
`build_ball_dataset.py` auto-labels only frames where circularity ≥ 0.75 (clean,
unoccluded balls). The model never sees balls partially hidden by players, in the
shooting arc, or near the scoreboard — exactly the hard cases. The "review" bucket
(0.55–0.75 circularity) is generated but not incorporated into training.

**Problem B — Dataset too small:**
20 videos yields ~10,000 frames. A basketball spans ~20×20 px at broadcast distance.
Tiny-object detection requires substantially more data and augmentation than this.

**Problem C — Model architecture:**
YOLOv8n is optimized for speed. For a 20×20 px object in 1280×720 frames,
YOLOv8s or YOLOv8m with finer detection head stride would be meaningfully more accurate.

The HSV fallback (`_color_detect`) is the actual working ball tracker right now. It fails on:
orange arena graphics, certain wood floor tones, balls occluded by players, and
unusual lighting. The 15-frame Kalman extrapolation gives ~0.5s bridge, but shot detection
needs `ball_track + possessor + 4 frames of history` simultaneously — any gap drops it.

**Fix plan — see [Phase 1](#phase-1--better-ball-detector) below.**

---

### 2. Block Detector — all false positives (F1=0.00)

**Root cause:** `_find_defender_overlap` in `src/events/block_detector.py:158` fires
whenever the ball center is within 80 px of a defender's upper bbox region. Normal
defensive positioning near the ball handler triggers phase 1 constantly.

**Fix (one line):** Add a ball height requirement — `ball_cy < y1 + 0.30 * (y2 - y1)`.
This requires the ball to be above the defender's shoulder line (top 30% of bbox),
which is only true during an actual shot block, not normal defense.

---

### 3. Homography Breaks on Cropped Frames

**Root cause:** `ClassicalKeyDetector` uses `mid_x = frame_w / 2` to assign left vs.
right keypoint names. When the court bbox detector crops the frame, `frame_w` is the
crop width — not the original frame width — so keypoints are flipped, corrupting H.

**Fix (one line):** Pass original full-frame width to `ClassicalKeyDetector.detect()`.

---

### 4. Deflection `dist_ft` = 0.0

**Root cause:** When possessor is unknown, the fallback computes distance from the
ball to the inferred possessor's bbox center. If the ball overlaps the bbox, distance
is ~0. Fix: compute distance using all players within 150 px, not just the possessor.

---

### 5. Steal Recall Too Low (R=0.32)

**Root cause:** `POSSESSION_DIST_PX = 90` (center-to-center) is too coarse. Defenders
playing tight defense are sometimes closer to the ball than the dribbler, causing
possession to flip to the wrong player before a steal is actually registered.

**Fix:** Replace center-to-center distance with ball-bbox-to-player-lower-body overlap
(bottom 40% of player bbox = waist-to-knee region). Dribbling players have ball near
their knees; defenders nearby do not. This is a targeted change to `BallTracker._infer_possessor`.

---

## Data Flow & Ground Truth

```
NBA.com PBP scraper          Raw broadcast .mp4
(fetch_pbp.py)                    |
      |                           |
      v                           v
  data/pbp.db              PipelineRunner
  406–455 events/game             |
  22 team rosters                 v
      |                    data/processed/
      |                    *_events.json  (detected events + timestamps)
      |                    *_stats.csv    (per-track box score)
      |                           |
      +----------+----------------+
                 |
                 v
        validate_pipeline.py
        auto time-alignment (cross-correlates steal density
        vs PBP steal timestamps to find clock offset)
                 |
                 v
        Precision / Recall / F1 per event type
        (tolerance window = 10 seconds)
```

### Games in pbp.db

**Validation set (tune against these):**
- 2016 Finals G7, 2013 Finals G6, 2017 Finals G5
- 2008 Finals G1, 2008 ECSF G7
- Bucks@Spurs Jan 2025, Suns@Wolves Jan 2025
- ECF 2012 G6 ← clips already on disk, currently the only validated game

**Test set — hold out until all tuning is done:**
- 2019 Finals G6, 2020 Finals G6, Lakers@Warriors Jan 2025

---

## Validation Targets

| Event | Current F1 | Target F1 | Primary Blocker |
|---|---|---|---|
| steal | 0.46 | 0.65 | Possession assignment noise |
| turnover | 0.40 | 0.55 | Possession assignment noise |
| block | 0.00 | 0.40 | False positive logic (1-line fix) |
| shot_made_2pt | 0.00 | 0.50 | Ball detector |
| shot_miss | 0.00 | 0.45 | Ball detector |
| rebound_def | 0.00 | 0.40 | Ball detector |
| assist | 0.00 | 0.40 | Ball detector → shot_made |

---

## Phased Roadmap

### Phase 0 — Quick Fixes (no training required, ~1 hour)

Three bugs, three files, immediate improvement to validation metrics:

**Fix 1 — Block detector false positives** (`src/events/block_detector.py:184`)
```python
# Current:
if x1 <= ball_cx <= x2 and y1 <= ball_cy <= upper_y:

# Fix: add ball height requirement
shoulder_y = y1 + (y2 - y1) * 0.30
if x1 <= ball_cx <= x2 and ball_cy <= shoulder_y:
```

**Fix 2 — Homography crop bug** (`src/court/keypoint_detector.py`)
```python
# Pass full_frame_w to ClassicalKeyDetector so mid_x is computed
# relative to original frame, not the cropped court region.
```

**Fix 3 — Deflection dist_ft** (`src/events/deflection_detector.py`)
```python
# When possessor is None, compute distance to all players within 150px
# and use the minimum, rather than defaulting to a bbox center overlap.
```

After fixes: re-run validation on ECF 2012 clips to get new baseline numbers.

---

### Phase 1 — Better Ball Detector (1–2 weeks)

**Step 1 — Build 500-video dataset (Mac, ~2 days)**
```bash
caffeinate -i python scripts/build_ball_dataset.py \
  --playlist "<nba-highlights-playlist-url>" \
  --max-videos 500 \
  --out-dir data/ball_dataset_500 \
  --segment-sec 300 \
  --start-sec 300
```
This downloads 5-minute segments, samples every 3rd frame, auto-labels with HSV,
and splits 15% into a val set. One video at a time, deletes the raw segment after
processing — disk usage stays manageable.

**Step 2 — Review flagged frames**
The "review" bucket (circularity 0.55–0.75) is the most valuable training signal —
these are partially occluded or arc-phase balls. Even correcting 500 of them manually
makes a large difference.
```bash
pip install label-studio
label-studio start  # then import data/ball_dataset_500/review/images/
```

**Step 3 — Add hard cases from game clips**
Extract 50–100 frames per clip where ball is near a player (possessor proximity zone).
These cases are systematically absent from the auto-labeled dataset.

**Step 4 — Train on RTX 4070**
```bash
# In scripts/train_ball_detector.py: change device="mps" → device="cuda"
python scripts/train_ball_detector.py \
  --data data/ball_dataset_500/dataset.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --batch 128 \
  --imgsz 640
```
Use `yolov8s.pt` (not `n`) — meaningfully better at small object detection.
Expected runtime: 4–6 hours on RTX 4070.

**Step 5 — Evaluate before deploying**
Target mAP50 > 0.40 on val split before replacing `ball_yolo.pt`.
Run visually on a game clip to check false positive rate (orange scoreboards, floor graphics).

---

### Phase 2 — Possession Logic Refinement

With a working ball detector, steal and turnover recall improve naturally — but
possession assignment is still noisy. Replace center-to-center distance with
bbox-overlap logic in `src/tracking/ball_tracker.py:_infer_possessor`:

```python
# Current: closest player center within 90px
# New: player whose lower-body region (bottom 40% of bbox) overlaps ball bbox
# Rationale: dribbling players have ball near their knees, not chest
```

This is a targeted ~20-line change with significant recall impact on steals and turnovers.

---

### Phase 3 — Validation Scale-Up

With block detector fixed and ball detector improved, run the full 7-game set:

1. Download 6 remaining validation game clips (yt-dlp, URLs in CLAUDE.md)
2. Process all:
   ```bash
   caffeinate -i PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/batch_run.py --local
   ```
3. Validate each game:
   ```bash
   python scripts/validate_pipeline.py --game <game_id>
   ```
4. Look for cross-game variance — if steal F1 is 0.65 on one game and 0.20 on another,
   that's an arena-specific or era-specific issue worth diagnosing.
5. Don't touch the 3 test games until all tuning is finalized.

---

### Phase 4 — Dashboard Polish

The API is largely complete (6 endpoints). Frontend state needs review. Key gaps:

**Player names:** `track_id` appears everywhere until jersey reading + identity resolution
settles. Need a name-resolution pass at the API layer or in the aggregator output.
`/api/leaderboard` and `/api/boxscore` should prefer `player_name` when available,
fall back to `#jersey_number`, fall back to `track_id`.

**Shot chart:** Verify court geometry renders correctly with the coordinate system
(center-origin feet, x: ±47, y: ±25). Left basket: (-41.75, 0), right: (+41.75, 0).

**Box score:** Handle clips processed before identity resolution — those have `track_id`
where `player_name` should be. The UI should make clear which players are identified.

---

### Phase 5 — Scale to Live Data (future)

Once validation metrics are solid on 7 held-out games:

**Real-time processing**
The pipeline already supports `process_frame()` for streaming. Main latency bottleneck:
SigLIP (~150ms/frame on M-series). Options: (a) run team classification on a background
thread with a frame buffer; (b) fall back to HSV classifier after the first 50 frames
(once team colors are fitted, SigLIP isn't needed every frame).

**NBA.com tracking stats comparison**
Compare detected stats vs. official tracking stats, not just PBP events. This validates
not just event detection but the counts — are we getting roughly the right number of
contested shots, deflections, etc.?

**Defensive metrics (the core value proposition)**
Once possession tracking is reliable:
- Contested FG% allowed by defender
- Help defense rate (how often does a player leave their matchup to help)
- Closeout distance and speed
- Defensive rebound positioning (were they in position before the shot?)

These don't exist in standard box scores. This is what makes the system interesting.

**Play type classification**
Pick-and-roll, isolation, post-up, transition — each requires 3–5 seconds of
ball + player court trajectories. The homography coordinates give the 2D floor map.
A sequence classifier (LSTM or lightweight Transformer over the last 90 frames of
court positions) is the natural architecture. Training data: Synergy Sports labels,
or manual annotation of 500 possessions.

**GAN wargaming (long-term)**
Learn a generative model of NBA possessions conditioned on defensive scheme.
Sample counterfactual possessions to evaluate defensive adjustments.
Requires large-scale labeled possessions first — this is a Phase 6+ goal.

---

## Critical Path

Everything gates on the ball detector. The unblocking sequence:

```
TODAY
  Phase 0: Fix block detector + homography + deflection (1 hour, no training)
  Re-run validation → get new block/steal/turnover baselines
       |
       v
WEEK 1-2
  Build 500-video ball dataset (build_ball_dataset.py, runs overnight)
  Review flagged frames in Label Studio
  Train ball_yolo.pt on RTX 4070 (4-6 hrs)
  Integrate + test on game clips
       |
       v
WEEK 2-3
  Re-run full 7-game validation set
  Tighten possession logic (steal/turnover recall)
  Download + process 6 remaining validation game clips
       |
       v
WEEK 3-4
  Dashboard: player names, shot chart, box score
  Hit validation targets across all 7 games
  Freeze tuning, run test set for final numbers
       |
       v
BEYOND
  Real-time processing → live game support
  Defensive metrics (contested FG%, help defense rate)
  Play type classification
  Scale to full seasons
```

The three Phase 0 fixes cost ~1 hour and give immediately useful information:
block detector turns on, homography stabilizes, and the next validation run gives
a cleaner read on where steal/turnover performance actually stands.

---

*Last updated: May 2026*
*Current model versions: court_detector_yolov8n.pt (mAP50=0.91), ball_yolo.pt (mAP50=0.04, pre-production)*
*Validated on: ECF 2012 G6, tolerance=10s*
