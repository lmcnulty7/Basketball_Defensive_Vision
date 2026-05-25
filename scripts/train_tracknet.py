"""
scripts/train_tracknet.py

Fine-tune TrackNetV3 on basketball tracking data.

Dataset (from build_tracknet_dataset.py):
  data/tracknet_dataset/
    train/<clip_name>/frames/000001.jpg ...
    train/<clip_name>/labels.csv   ← frame_id, visibility, cx, cy, orig_frame_idx
    val/  ...

Input tensor  : (B, 27, 288, 512)  — [bg_frame, f1..f8], each RGB [0,1]
Target tensor : (B, 8, 288, 512)   — Gaussian heatmap per frame (σ=5px)
Loss          : focal BCE (handles ~0.05% ball-pixel class imbalance)
Metric        : within-distance accuracy (WDA) at 5px threshold

Usage (A100):
    python scripts/train_tracknet.py \\
        --data-dir  data/tracknet_dataset \\
        --weights   models/checkpoints/tracknet_best.pt \\
        --out-dir   models/checkpoints \\
        --batch-size 64 \\
        --epochs     50 \\
        --lr         1e-4 \\
        --device     cuda

Output:
    models/checkpoints/tracknet_basketball.pt       ← best val WDA
    models/checkpoints/tracknet_basketball_last.pt  ← latest epoch
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TRACKNET_H          = 288
TRACKNET_W          = 512
SEQ_LEN             = 8      # must match pretrained weights
GAUSS_SIGMA         = 5.0    # heatmap Gaussian std dev (pixels at 288×512)
WDA_THRESH          = 5.0    # within-distance accuracy threshold (pixels)
FOCAL_ALPHA         = 0.25   # focal loss alpha
FOCAL_GAMMA         = 2.0    # focal loss gamma
FRAME_STRIDE        = 3      # video frame stride used by the dataset builder
MAX_ORIG_FRAME_SPAN = 90     # max raw-frame span of an 8-frame window (~3s at 30fps)
                              # windows exceeding this span a camera cut — skip them


# ── Gaussian heatmap ───────────────────────────────────────────────────────────

def _make_heatmap(
    cx: float,
    cy: float,
    h: int = TRACKNET_H,
    w: int = TRACKNET_W,
    sigma: float = GAUSS_SIGMA,
) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    return np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────

class TrackNetDataset(Dataset):
    """
    Sliding-window dataset over saved frame sequences.

    Each sample is a window of SEQ_LEN consecutive saved frames from one clip.
    Windows are accepted only when:
      - ≥ min_visible frames have a ball label (visibility=1)
      - The span of orig_frame_idx values ≤ MAX_ORIG_FRAME_SPAN raw frames,
        which skips windows that cross camera cuts or non-court gaps.

    orig_frame_idx is read from labels.csv if present (new builder format).
    Old-format datasets without the column fall back to estimated values.
    """

    def __init__(
        self,
        split_dir: Path,
        seq_len: int = SEQ_LEN,
        sigma: float = GAUSS_SIGMA,
        min_visible: int = 1,
        max_orig_span: int = MAX_ORIG_FRAME_SPAN,
    ) -> None:
        self.seq_len = seq_len
        self.sigma   = sigma
        self._samples: List[Tuple[Path, List[int], Dict[int, Tuple]]] = []

        for clip_dir in sorted(split_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            frames_dir = clip_dir / "frames"
            label_path = clip_dir / "labels.csv"
            if not frames_dir.exists() or not label_path.exists():
                continue

            # labels[frame_id] = (visibility, cx, cy, orig_frame_idx)
            labels: Dict[int, Tuple[int, float, float, int]] = {}
            with open(label_path) as f:
                reader = csv.DictReader(f)
                has_orig = "orig_frame_idx" in (reader.fieldnames or [])
                for row in reader:
                    fid  = int(row["frame_id"])
                    vis  = int(row["visibility"])
                    cx   = float(row["cx"])
                    cy   = float(row["cy"])
                    orig = int(row["orig_frame_idx"]) if has_orig else fid * FRAME_STRIDE
                    labels[fid] = (vis, cx, cy, orig)

            fids = sorted(labels.keys())
            n_accepted = 0
            for i in range(len(fids) - seq_len + 1):
                window = fids[i : i + seq_len]

                # Skip windows that cross camera cuts / large gaps
                orig_span = labels[window[-1]][3] - labels[window[0]][3]
                if orig_span > max_orig_span:
                    continue

                n_vis = sum(labels[fid][0] for fid in window)
                if n_vis < min_visible:
                    continue

                self._samples.append((frames_dir, window, labels))
                n_accepted += 1

        logger.info("TrackNetDataset: %d windows from %s", len(self._samples), split_dir)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frames_dir, window, labels = self._samples[idx]

        def _load(fid: int) -> torch.Tensor:
            path = frames_dir / f"{fid:06d}.jpg"
            bgr  = cv2.imread(str(path))
            if bgr is None:
                bgr = np.zeros((TRACKNET_H, TRACKNET_W, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W)

        frames = [_load(fid) for fid in window]

        # Input: [bg, f1..f8] where bg = oldest frame
        bg = frames[0]
        x  = torch.cat([bg] + frames, dim=0)  # (27, H, W)

        # Target: Gaussian heatmap per frame
        targets = []
        for fid in window:
            vis, cx, cy, _ = labels[fid]   # discard orig_frame_idx
            if vis == 1:
                hm = _make_heatmap(cx, cy, sigma=self.sigma)
            else:
                hm = np.zeros((TRACKNET_H, TRACKNET_W), dtype=np.float32)
            targets.append(torch.from_numpy(hm))
        y = torch.stack(targets, dim=0)  # (SEQ_LEN, H, W)

        return x, y


# ── Loss ───────────────────────────────────────────────────────────────────────

def focal_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    """
    Focal BCE loss for heatmap regression.
    Handles the ~0.05% ball-pixel class imbalance without manual pos_weight tuning.

    pred / target : (B, SEQ_LEN, H, W), values in [0, 1]
    """
    eps  = 1e-7
    pred = pred.clamp(eps, 1.0 - eps)

    pos = -alpha       * (1.0 - pred) ** gamma * target        * pred.log()
    neg = -(1.0-alpha) * pred          ** gamma * (1.0-target) * (1.0 - pred).log()
    return (pos + neg).mean()


# ── Metric ─────────────────────────────────────────────────────────────────────

def within_dist_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = WDA_THRESH,
) -> Tuple[int, int]:
    """
    Returns (n_correct, n_total) for visible frames in this batch.
    Correct = predicted argmax within `threshold` pixels of target peak.
    """
    B, S, H, W = pred.shape
    n_correct = n_total = 0
    pred_np   = pred.cpu().numpy()
    tgt_np    = target.cpu().numpy()

    for b in range(B):
        for s in range(S):
            t_hm = tgt_np[b, s]
            if t_hm.max() < 0.1:   # invisible frame
                continue
            n_total += 1
            ty, tx = np.unravel_index(t_hm.argmax(), t_hm.shape)
            py, px = np.unravel_index(pred_np[b, s].argmax(), pred_np[b, s].shape)
            if np.hypot(px - tx, py - ty) <= threshold:
                n_correct += 1

    return n_correct, n_total


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.cuda.amp.GradScaler",
    device: str,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.split(":")[0], enabled=(device != "cpu")):
            pred = model(x)
            loss = focal_bce_loss(pred, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = total_visible = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.split(":")[0], enabled=(device != "cpu")):
            pred = model(x)
            loss = focal_bce_loss(pred, y)

        total_loss += loss.item()
        nc, nt = within_dist_accuracy(pred, y)
        total_correct  += nc
        total_visible  += nt

    avg_loss = total_loss / max(len(loader), 1)
    wda      = total_correct / max(total_visible, 1)
    return avg_loss, wda


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrackNetV3 on basketball data")
    parser.add_argument("--data-dir",   default="data/tracknet_dataset")
    parser.add_argument("--weights",    default="models/checkpoints/tracknet_best.pt",
                        help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--out-dir",    default="models/checkpoints")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=8)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--resume",     default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Model (init CUDA before DataLoader pin_memory) ────────────────────────
    from src.detection.tracknet import TrackNetV3, load_pretrained

    model = TrackNetV3().to(device)
    load_pretrained(model, args.weights, device=device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = TrackNetDataset(data_dir / "train")
    val_ds   = TrackNetDataset(data_dir / "val",   min_visible=1)

    if len(train_ds) == 0:
        logger.error("No training samples found in %s/train", data_dir)
        raise SystemExit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device != "cpu"),
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(args.workers > 0),
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device != "cpu" and "cuda" in device))

    start_epoch = 0
    best_wda    = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_wda    = ckpt.get("best_wda", 0.0)
        logger.info("Resumed from epoch %d  (best WDA=%.3f)", start_epoch, best_wda)

    logger.info(
        "Training: %d samples | Val: %d samples | batch=%d | lr=%g | device=%s",
        len(train_ds), len(val_ds), args.batch_size, args.lr, device,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_wda = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_WDA=%.3f  lr=%.2e  t=%.0fs",
            epoch + 1, args.epochs,
            train_loss, val_loss, val_wda,
            optimizer.param_groups[0]["lr"],
            elapsed,
        )

        # Save last checkpoint
        ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_wda":  best_wda,
            "val_wda":   val_wda,
        }
        torch.save(ckpt, out_dir / "tracknet_basketball_last.pt")

        # Save best checkpoint
        if val_wda > best_wda:
            best_wda = val_wda
            torch.save(ckpt, out_dir / "tracknet_basketball.pt")
            logger.info("  ✓ New best  WDA=%.3f — saved to tracknet_basketball.pt", best_wda)

    logger.info("Training complete.  Best val WDA: %.3f", best_wda)
    logger.info("Best checkpoint: %s", out_dir / "tracknet_basketball.pt")


if __name__ == "__main__":
    main()
