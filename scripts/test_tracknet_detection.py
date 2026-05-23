"""
Quick visual validation of TrackNetDetector on real game frames.
Saves 50 annotated frames to /tmp/tracknet_test/ for inspection.
Usage: PYTHONPATH=. python scripts/test_tracknet_detection.py --clip data/raw/clip_10m00_18m00.mp4
"""
import argparse
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np

TRACKNET_H, TRACKNET_W = 288, 512
SEQ_LEN = 8
CONF_THRESH = 0.50
MAX_FRAMES = 200
SAVE_EVERY = 1  # save every frame (up to MAX_FRAMES)
OUT_DIR = Path("/tmp/tracknet_test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="data/raw/clip_10m00_18m00.mp4")
    parser.add_argument("--weights", default="models/checkpoints/tracknet_best.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    from src.detection.tracknet import TrackNetV3, load_pretrained

    print(f"Loading TrackNetV3 from {args.weights} on {args.device} ...")
    model = TrackNetV3()
    load_pretrained(model, args.weights, device=args.device)
    model.to(args.device).eval()
    print("Model ready.")

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.clip}")

    buf: deque = deque(maxlen=SEQ_LEN)
    n_saved = 0
    n_detections = 0
    frame_idx = 0
    stride = 3

    while cap.isOpened() and n_saved < MAX_FRAMES:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        buf.append(bgr.copy())

        if len(buf) < SEQ_LEN:
            continue

        # Preprocess: [bg, f1..f8] where bg = oldest frame
        frames = list(buf)

        def _t(f):
            sm = cv2.resize(f, (TRACKNET_W, TRACKNET_H))
            rgb = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(rgb).permute(2, 0, 1)

        bg = _t(frames[0])
        seq = [_t(f) for f in frames]
        x = torch.cat([bg] + seq, dim=0).unsqueeze(0).to(args.device)  # (1, 27, H, W)

        with torch.no_grad():
            out = model(x)  # (1, 8, 288, 512)

        hm = out[0, -1].cpu().numpy()  # last channel = current frame
        peak_conf = float(hm.max())

        # Current frame (last in buffer)
        vis = cv2.resize(bgr, (TRACKNET_W, TRACKNET_H))
        orig_h, orig_w = bgr.shape[:2]

        # Overlay heatmap
        hm_uint8 = (hm * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(vis, 0.6, hm_color, 0.4, 0)

        ball_detected = False
        if peak_conf >= args.conf:
            n_detections += 1
            ball_detected = True
            iy, ix = np.unravel_index(hm.argmax(), hm.shape)
            # Scale back to orig size for annotation
            cx_orig = int(ix * orig_w / TRACKNET_W)
            cy_orig = int(iy * orig_h / TRACKNET_H)
            # Also draw on overlay (288×512 space)
            cv2.circle(overlay, (ix, iy), 10, (0, 255, 0), 2)
            cv2.putText(overlay, f"conf={peak_conf:.2f}", (ix + 12, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        label = f"frame={frame_idx}  conf={peak_conf:.3f}  {'BALL' if ball_detected else 'no-ball'}"
        cv2.putText(overlay, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        out_path = OUT_DIR / f"{n_saved:04d}_f{frame_idx}.jpg"
        cv2.imwrite(str(out_path), overlay)
        n_saved += 1

    cap.release()
    det_rate = n_detections / max(n_saved, 1) * 100
    print(f"\nResults: {n_saved} frames processed, {n_detections} ball detections ({det_rate:.1f}%)")
    print(f"Annotated frames saved to {OUT_DIR}/")
    print(f"Conf threshold: {args.conf}  |  Peak confs seen: check frames above")


if __name__ == "__main__":
    main()
