# Basketball Defensive Vision — Project Context for Claude Code

## What This Project Does
Computer vision pipeline that extracts **full NBA box score stats** (offensive + defensive)
from broadcast video. Runs on Apple Silicon Mac (MPS).
Outputs a FastAPI + React dashboard with leaderboard, event feed, shot chart, and box score.

Input: YouTube broadcast clips → Output: points, FGM/A, assists, steals, blocks, etc. per player

---

## Architecture (current)

```
VideoReader → MultiTracker (BoT-SORT) → BallTracker (HSV orange)
    → SigLIPTeamClassifier (SigLIP + UMAP + K-means)
    → JerseyReader (EasyOCR) → PlayerIdentityResolver
    → EnhancedClassicalDetector → CourtHomography (H matrix)
    → PossessionTracker (basket assignment + halftime detection)
    → EventOrchestrator:
        StealDetector, TurnoverDetector, BlockDetector, DeflectionDetector,
        ContestDetector, ReboundDetector, ShotDetector, AssistDetector
    → StatsAggregator (full box score) → FastAPI → React dashboard
```

### Key source files
| File | Purpose |
|---|---|
| `src/pipeline/runner.py` | Top-level frame loop, wires all stages |
| `src/classification/siglip_team_classifier.py` | SigLIP + UMAP + K-means team classifier |
| `src/classification/team_classifier.py` | HSV K-means fallback (used when SigLIP unavailable) |
| `src/detection/jersey_reader.py` | EasyOCR jersey number reading |
| `src/tracking/player_identity.py` | Maps jersey# → player name via roster DB |
| `src/tracking/possession_tracker.py` | Basket assignment + halftime detection |
| `src/court/court_detector.py` | YOLOv8n bbox detector for court region |
| `src/court/keypoint_detector.py` | EnhancedClassicalDetector + NeuralKeyDetector |
| `src/court/homography.py` | CourtHomography — pixel ↔ court-feet mapping |
| `src/court/court_model.py` | NBA court geometry, COURT_KEYPOINTS dict |
| `src/events/event.py` | PipelineState + Event + PossessionState dataclasses |
| `src/events/shot_detector.py` | Two-phase shot detection (attempt → made/miss) |
| `src/events/assist_detector.py` | Last-passer before made basket |
| `src/events/turnover_detector.py` | Non-steal possession changes |
| `src/events/block_detector.py` | Two-phase block detection |
| `src/events/steal_detector.py` | Possession-change steal detection |
| `src/events/deflection_detector.py` | Ball trajectory change detection |
| `src/stats/aggregator.py` | Full box score (points, FGM/A, FG%, assists, TOV, etc.) |
| `api/main.py` | FastAPI: /api/events /api/leaderboard /api/clips /api/boxscore /api/shotchart |
| `scripts/batch_run.py` | Process all .mp4s in data/raw/ (--local flag) |
| `scripts/run_pipeline.py` | Process single clip (--clip flag) |
| `scripts/validate_pipeline.py` | P/R/F1 vs PBP ground truth with auto time-alignment |
| `scripts/fetch_pbp.py` | Scrape + store PBP in data/pbp.db |
| `scripts/fetch_rosters.py` | Scrape + store team rosters in data/pbp.db |
| `scripts/train_court_detector.py` | Train court bbox detector |
| `scripts/build_ball_dataset.py` | Build YOLO ball detector dataset from YouTube playlist (HSV auto-label) |
| `scripts/train_ball_detector.py` | Fine-tune YOLOv8n ball detector on ball dataset |

---

## Current State (as of May 2026)

### What works
- Full pipeline end-to-end on MPS (~1× real-time, slower with SigLIP)
- Court bounding-box detector: mAP50=0.91 (models/checkpoints/court_detector_yolov8n.pt)
- Ball detector: trained on 20-video dataset, mAP50=0.04 (models/checkpoints/ball_yolo.pt) — exists but not production-ready, BallDetector loads it automatically
- Homography: H=✓ per frame using EnhancedClassicalDetector + court detector crop
- SigLIP team classifier: loaded and working (falls back to HSV if unavailable)
- EasyOCR jersey reader: working (reads numbers every 90 frames per track)
- Player identity: roster DB populated for 11 games, call runner.set_game_id(game_id)
- PBP database: 11 games stored in data/pbp.db (406–455 events each)
- Rosters: 22 teams stored in data/pbp.db rosters table
- Validation framework: validate_pipeline.py with auto time-alignment working

### Event detection (last validated — ECF 2012 G6, tolerance=10s)
| Event | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| steal | 0.800 | 0.320 | 0.457 | Good precision, low recall |
| turnover | 0.500 | 0.333 | 0.400 | New detector, working |
| block | 0.000 | 0.000 | 0.000 | All false positives — needs fix |
| shot_made/miss | 0 | 0 | 0 | Needs reliable ball tracker |
| rebounds | 0 | 0 | 0 | Needs reliable ball tracker |

### PBP + Rosters stored (data/pbp.db)
**Validation games:** 2016 Finals G7, 2013 Finals G6, 2017 Finals G5,
2008 Finals G1, 2008 ECSF G7, Bucks@Spurs Jan 2025, Suns@Wolves Jan 2025, ECF 2012 G6
**Test games (hold out):** 2019 Finals G6, 2020 Finals G6, Lakers@Warriors Jan 2025

---

## Known Issues

### 1. Block detector: all false positives
**Root cause:** `_find_defender_overlap` fires on normal defensive positioning near ball,
not just actual shot blocks. No ball-height requirement.
**Fix needed:** Require ball above player's shoulder height (top 30% of bbox).

### 2. Shot/rebound detectors not firing
**Root cause:** Ball tracker (HSV-based) loses ball too frequently. ShotDetector needs
`ball_track is not None AND ball_possessor_id is not None AND ≥4 frames history` simultaneously.
**Partial fix:** ball_yolo.pt trained (mAP50=0.04, 20-video dataset). Not production-ready.
**Fix needed:** Retrain on 500-video dataset (use RTX 4070, ~4–6 hrs) or find pre-trained model.
Roboflow Universe models explored — no free weight download available.

### 3. dist_ft=0.0 on most deflections
**Root cause:** Pixel-space fallback when possessor unknown → ball overlaps player bbox center.
**Fix needed:** Use all nearby players when possessor is unknown.

### 4. Homography keypoint name assignment breaks on cropped frames
**Root cause:** ClassicalKeyDetector uses `mid_x = frame_w/2` but crop changes frame width.
**Fix needed:** Pass full-frame width to ClassicalKeyDetector.

### 5. SmolVLM2 not loading (jersey reader uses EasyOCR instead)
**Root cause:** SmolVLM2 requires transformers ≥ 4.52 which requires PyTorch ≥ 2.4.
Current PyTorch: 2.2.2. EasyOCR is used as fallback — works well for jersey reading.
**Fix:** Upgrade PyTorch to ≥ 2.4 when ready.

---

## Immediately Next

**Priority 1 — Better ball detector (blocks shot/rebound detection)**
Options (pick one):
- RTX 4070: build 500-video dataset on Mac, train on 4070 with `--batch 128 --epochs 100` (~4–6 hrs)
- Build script: `caffeinate -i python scripts/build_ball_dataset.py --playlist "<url>" --out-dir data/ball_dataset_500`
- Train: change `device="mps"` → `device="cuda"` in train_ball_detector.py, then run with `--data data/ball_dataset_500/dataset.yaml`

**Priority 2 — Re-run pipeline + Validation**
```bash
caffeinate -i PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/batch_run.py --local
python scripts/validate_pipeline.py --game 201206070BOS
```

After re-run, download the 10 validation game clips and process them.

### 10 validation game YouTube URLs
**Validation set (7 games):**
- 2016 Finals G7: https://www.youtube.com/watch?v=EoVTttvKfRs
- 2013 Finals G6: https://www.youtube.com/watch?v=JOLD8XXh78A
- 2017 Finals G5: https://www.youtube.com/watch?v=OPPhDYSxozU
- 2008 Finals G1: https://www.youtube.com/watch?v=BdcP-XwUCk8
- 2008 ECSF G7:   https://www.youtube.com/watch?v=b5c8HHgPqEE
- Bucks@Spurs Jan 2025: https://www.youtube.com/watch?v=L7o4UCIIqS4
- Suns@Wolves Jan 2025: https://www.youtube.com/watch?v=RTpRlPeuDoo

**Test set — DO NOT evaluate until tuning is done:**
- 2019 Finals G6: https://www.youtube.com/watch?v=Z4ji-KQVyrw
- 2020 Finals G6: https://www.youtube.com/watch?v=-dwMxiCJZfE
- Lakers@Warriors Jan 2025: https://www.youtube.com/watch?v=jwR1ajwx-qM

### Game ID → clip mapping (for validate_pipeline.py)
Add to CLIP_GAME_MAP in scripts/validate_pipeline.py as new clips are processed:
```python
"clip_10m00_18m00": "201206070BOS",  # ECF 2012 G6
"clip_26m00_34m00": "201206070BOS",
"clip_40m00_48m00": "201206070BOS",
"clip_55m00_63m00": "201206070BOS",
"clip_70m00_78m00": "201206070BOS",
```

---

## MPS / Apple Silicon Setup

Always run pipeline with:
```bash
caffeinate -i PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/batch_run.py --local
```

Use `caffeinate -i` to prevent Mac sleep during long runs.

The lap stub (scipy-backed linear assignment) is injected in run_pipeline.py and
batch_run.py to avoid x86_64/arm64 conflict with the official lap wheel.

---

## Files to Transfer (gitignored — copy manually)
| File | Size | Notes |
|---|---|---|
| `models/checkpoints/court_detector_yolov8n.pt` | 6MB | Court bbox detector |
| `models/checkpoints/ball_yolo.pt` | 6MB | Ball detector (mAP50=0.04, first training run) |
| `models/checkpoints/yolov8m.pt` | 50MB | Player detector (auto-downloads if missing) |
| `yolov8n.pt` | 6MB | YOLOv8n base weights |
| `data/raw/*.mp4` | Large | Raw video clips |
| `data/pbp.db` | ~8MB | PBP + rosters for 11 games |
| `data/processed/*.json` + `*.csv` | Small | Processed events + stats |

### USB transfer complete (May 2026)
All files copied to USB root. On the original Mac:
```
cp /Volumes/USB\ STICK/ball_yolo.pt models/checkpoints/
cp /Volumes/USB\ STICK/court_detector_yolov8n.pt models/checkpoints/
cp /Volumes/USB\ STICK/yolov8m.pt models/checkpoints/
cp /Volumes/USB\ STICK/yolov8n.pt .
cp /Volumes/USB\ STICK/pbp.db data/
git pull
```

---

## Running the Dashboard
```bash
# Terminal 1 — backend
cd api && uvicorn main:app --reload

# Terminal 2 — frontend
cd frontend && npm run dev
# Opens at http://localhost:5173
```

---

## Coordinate System
Court coordinates: center-origin, feet.
- x: -47 (left baseline) → 0 (center) → +47 (right baseline)
- y: -25 (bottom sideline) → 0 (center) → +25 (top sideline)
- Left basket: (-41.75, 0), Right basket: (41.75, 0)

Keypoint names (14 total, order matters for YOLO-pose):
home_paint_bl, home_paint_tl, home_paint_br, home_paint_tr,
away_paint_bl, away_paint_tl, away_paint_br, away_paint_tr,
half_bot, half_top,
home_three_bl, home_three_tl, away_three_bl, away_three_tl

---

## Long-term Vision
Full basketball intelligence platform:
1. Live stat tracking (offensive + defensive box score) ← current focus
2. Pattern recognition within + across games
3. Tactical schema recognition (pick-and-roll, zone defense, etc.)
4. GAN for strategy wargaming
5. Player profiles over time
6. Roster building + rotation recommendations

Build order: PBP stats → NBA.com tracking stats → Synergy play types → tactical annotation → GAN
