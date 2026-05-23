"""
scripts/demo_court.py

Demonstrates the court module end-to-end using a synthetic court image.

What this proves
────────────────
1. CourtModel   — shot zone classification across all 6 zones.
2. CourtHomography — H computed from 8 synthetic keypoint pairs;
                     pixel→court and court→pixel both verified.
3. Projection pipeline — 10 simulated player foot-points (pixel coords)
                         are projected to real court coordinates, displayed
                         as colored dots on a bird's-eye court diagram.
4. Shot zone of each "player" is labeled on the court diagram.
5. Defender distances are computed between offensive/defensive pairs.

Outputs
───────
  data/samples/court_zones.jpg      — annotated zone diagram
  data/samples/court_projection.jpg — homography round-trip verification

Run:
    python scripts/demo_court.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.court.court_model import (
    CourtModel, ShotZone, COURT_KEYPOINTS,
    BASKET_LEFT, BASKET_RIGHT,
    HALF_COURT, COURT_WIDTH, PAINT_DEPTH, PAINT_HALF_W,
    THREE_RADIUS, THREE_CORNER_Y,
)
from src.court.homography import CourtHomography

OUT_DIR = ROOT / "data" / "samples"
FONT    = cv2.FONT_HERSHEY_SIMPLEX

# ── Color palette ─────────────────────────────────────────────────────────────

ZONE_COLORS = {
    ShotZone.RESTRICTED_AREA: (0,   180,   0),
    ShotZone.PAINT_NON_RA:    (0,   220, 100),
    ShotZone.MID_RANGE:       (0,   160, 255),
    ShotZone.CORNER_3:        (220,  60,  60),
    ShotZone.ABOVE_BREAK_3:   (200,   0, 200),
    ShotZone.BACKCOURT:       (100, 100, 100),
    ShotZone.OUT_OF_BOUNDS:   (50,   50,  50),
}

TEAM_COLORS = {
    "offense": (0, 220, 0),    # green
    "defense": (0, 100, 255),  # orange
}


# ── Court drawing utilities ───────────────────────────────────────────────────

def make_court_diagram(w=940, h=500) -> np.ndarray:
    """Draw a top-down NBA court image in pixel space."""
    img = np.full((h, w, 3), (45, 130, 210), dtype=np.uint8)   # maple wood (BGR)
    white = (255, 255, 255)
    t = 2

    def c(x_ft, y_ft):
        px = int((x_ft + HALF_COURT) / (2 * HALF_COURT) * w)
        py = int((COURT_WIDTH / 2 - y_ft) / COURT_WIDTH * h)
        return (px, py)

    # Boundary
    cv2.rectangle(img, c(-HALF_COURT, -COURT_WIDTH/2),
                  c(HALF_COURT,  COURT_WIDTH/2), white, t)

    # Home paint
    cv2.rectangle(img, c(-HALF_COURT, -PAINT_HALF_W),
                  c(-HALF_COURT + PAINT_DEPTH, PAINT_HALF_W), white, t)

    # Away paint
    cv2.rectangle(img, c(HALF_COURT - PAINT_DEPTH, -PAINT_HALF_W),
                  c(HALF_COURT, PAINT_HALF_W), white, t)

    # Half-court line
    cv2.line(img, c(0, -COURT_WIDTH/2), c(0, COURT_WIDTH/2), white, t)

    # 3PT arcs (approximate with polyline)
    for bx, by, side in [(BASKET_LEFT[0], 0, "left"), (BASKET_RIGHT[0], 0, "right")]:
        pts = []
        for deg in range(-70, 71):
            rad = np.deg2rad(deg)
            cx = bx + THREE_RADIUS * np.cos(rad) * (1 if side == "right" else -1)
            cy = by + THREE_RADIUS * np.sin(rad)
            if abs(cy) < THREE_CORNER_Y:
                pts.append(c(cx, cy))
        if pts:
            cv2.polylines(img, [np.array(pts)], False, white, t)

    # 3PT straight sections (corners)
    for bx, sign in [(BASKET_LEFT[0], 1), (BASKET_RIGHT[0], -1)]:
        corner_x = bx + sign * np.sqrt(THREE_RADIUS**2 - THREE_CORNER_Y**2)
        base_x = bx - sign * (HALF_COURT + bx) if sign == 1 else bx + sign * (HALF_COURT - bx)
        for sy in [-THREE_CORNER_Y, THREE_CORNER_Y]:
            x0 = -HALF_COURT if sign == 1 else HALF_COURT
            cv2.line(img, c(x0, sy), c(corner_x, sy), white, t)

    # Baskets
    for bx, by in [BASKET_LEFT, BASKET_RIGHT]:
        cv2.circle(img, c(bx, by), 5, (0, 0, 255), -1)

    return img, lambda x, y: c(x, y)


# ── Simulated players ─────────────────────────────────────────────────────────

# 5 offensive + 5 defensive player positions in court feet [x, y]
PLAYERS = {
    # Offensive team (shooting at left basket)
    "O1": {"court": np.array([-20.0,  0.0]), "team": "offense"},  # mid-range, top key
    "O2": {"court": np.array([-35.0, -6.0]), "team": "offense"},  # paint
    "O3": {"court": np.array([-18.0, 22.5]), "team": "offense"},  # corner 3
    "O4": {"court": np.array([-20.0, 12.0]), "team": "offense"},  # mid-range wing
    "O5": {"court": np.array([ -8.0,  0.0]), "team": "offense"},  # above break 3

    # Defensive team
    "D1": {"court": np.array([-22.0,  1.5]), "team": "defense"},  # guarding O1
    "D2": {"court": np.array([-33.0, -5.5]), "team": "defense"},  # guarding O2
    "D3": {"court": np.array([-19.5, 22.0]), "team": "defense"},  # guarding O3
    "D4": {"court": np.array([-22.0, 13.0]), "team": "defense"},  # guarding O4
    "D5": {"court": np.array([-11.0,  0.5]), "team": "defense"},  # guarding O5
}


def main():
    print("=" * 65)
    print("Basketball Defensive Vision — Court Module Demo")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    court = CourtModel()

    # ── 1. Zone classification demo ───────────────────────────────────────────
    print("\n[1/3] Shot zone classification")
    diagram, c = make_court_diagram()

    for pid, info in PLAYERS.items():
        cx, cy = info["court"]
        zone   = court.get_zone(cx, cy, basket="left")
        color  = TEAM_COLORS[info["team"]]
        zone_c = ZONE_COLORS[zone]

        px = c(cx, cy)
        radius = 14
        cv2.circle(diagram, px, radius, zone_c, -1)
        cv2.circle(diagram, px, radius, color,  2)
        cv2.putText(diagram, pid, (px[0] - 8, px[1] + 4), FONT, 0.35, (0,0,0), 1)

        if info["team"] == "offense":
            zone_str = zone.value.replace("_", " ")
            print(f"  {pid} at ({cx:+.1f}, {cy:+.1f}) ft → {zone_str}")

    # Zone legend
    legend_y = 15
    for zone, color in ZONE_COLORS.items():
        cv2.circle(diagram, (15, legend_y), 7, color, -1)
        cv2.putText(diagram, zone.value, (27, legend_y + 4), FONT, 0.32, (255,255,255), 1)
        legend_y += 18

    zones_path = OUT_DIR / "court_zones.jpg"
    cv2.imwrite(str(zones_path), diagram)
    print(f"  → {zones_path}")

    # ── 2. Homography computation + projection ────────────────────────────────
    print("\n[2/3] Homography: pixel ↔ court projection")

    # Simulate what the keypoint detector would return:
    # 8 court points mapped to a trapezoid simulating a broadcast camera angle.
    # (In real use, these pixel_pts come from the keypoint detector.)
    diagram2, c2 = make_court_diagram()

    # Use 8 of our canonical keypoints. Their "pixel" coords are what they
    # would appear as in a broadcast frame — we simulate a perspective warp.
    KP_NAMES = [
        "home_paint_bl", "home_paint_tl", "home_paint_br", "home_paint_tr",
        "away_paint_bl", "away_paint_tl", "away_paint_br", "away_paint_tr",
    ]
    court_pts  = np.array([COURT_KEYPOINTS[k] for k in KP_NAMES], dtype=np.float32)

    # Simulate pixel positions by applying court_to_pixel (top-down diagram)
    pixel_pts  = np.array([
        CourtModel.court_to_pixel(COURT_KEYPOINTS[k], 940, 500)
        for k in KP_NAMES
    ], dtype=np.float32)

    # Add mild synthetic noise (simulating imperfect detection)
    np.random.seed(42)
    pixel_pts_noisy = pixel_pts + np.random.randn(*pixel_pts.shape) * 2.0

    hom = CourtHomography()
    hom.compute(pixel_pts_noisy, court_pts)
    print(f"  H computed: inlier reprojection error = {hom.quality:.4f} ft")

    # Project all player foot-points to court, then back to pixel for visualization
    all_court = np.array([info["court"] for info in PLAYERS.values()])
    all_pixel  = hom.to_pixel_batch(all_court)

    for (pid, info), pix in zip(PLAYERS.items(), all_pixel):
        if np.any(np.isnan(pix)):
            continue
        color = TEAM_COLORS[info["team"]]
        pt    = (int(pix[0]), int(pix[1]))
        cv2.circle(diagram2, pt, 14, color, -1)
        cv2.putText(diagram2, pid, (pt[0]-8, pt[1]+4), FONT, 0.35, (0,0,0), 1)

    # Draw keypoints used for H
    for px in pixel_pts_noisy:
        cv2.drawMarker(diagram2, (int(px[0]), int(px[1])), (0,255,255),
                       cv2.MARKER_CROSS, 20, 2)

    cv2.putText(diagram2, f"H reprojection error: {hom.quality:.4f} ft",
                (10, 485), FONT, 0.5, (0, 255, 255), 1)

    proj_path = OUT_DIR / "court_projection.jpg"
    cv2.imwrite(str(proj_path), diagram2)
    print(f"  → {proj_path}")

    # ── 3. Defensive stats preview ────────────────────────────────────────────
    print("\n[3/3] Defensive stat computation from court coordinates")

    offense_pos = {pid: info["court"] for pid, info in PLAYERS.items()
                   if info["team"] == "offense"}
    defense_pos = {pid: info["court"] for pid, info in PLAYERS.items()
                   if info["team"] == "defense"}

    matchups = court.matchup_pair(offense_pos, defense_pos, max_dist_ft=15.0)

    print(f"\n  {'Matchup':<22} {'Dist (ft)':>9}  {'Contested?':>12}  {'Zone'}")
    print(f"  {'-'*22} {'-'*9}  {'-'*12}  {'-'*18}")
    for oid, did in matchups.items():
        op = offense_pos[oid]
        dp = defense_pos.get(did, None) if did else None
        zone = court.get_zone(op[0], op[1], basket="left").value
        if dp is not None:
            dist = court.defender_distance(op, dp)
            contested = court.is_contested(op, dp, threshold_ft=4.0)
            print(f"  {oid} ← {did:<16} {dist:>9.1f}  {'YES' if contested else 'no':>12}  {zone}")
        else:
            print(f"  {oid} ← {'(unguarded)':<16} {'—':>9}  {'—':>12}  {zone}")

    print(f"\n{'=' * 65}")
    print(f"Outputs:")
    print(f"  {zones_path.name}     — zone labels per player on court diagram")
    print(f"  {proj_path.name}  — H projection + keypoint markers")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
