"""
src/detection/tracknet.py

TrackNetV3 architecture that exactly matches the pretrained checkpoint from
qaz812345/TrackNetV3 (seq_len=8, bg_mode='concat').

Architecture
────────────
Input : (B, 27, 288, 512)   — 9 frames × 3 RGB channels
         = [bg_frame, f1, f2, f3, f4, f5, f6, f7, f8] each normalized to [0,1]

Encoder:
  down_block_1 : CBR(27→64) × 2  → skip1, MaxPool
  down_block_2 : CBR(64→128) × 2 → skip2, MaxPool
  down_block_3 : CBR(128→256) × 3 → skip3, MaxPool
  bottleneck   : CBR(256→512) × 3

Decoder:
  up_block_1 : bilinear(×2) + cat(skip3) → CBR(768→256) × 3
  up_block_2 : bilinear(×2) + cat(skip2) → CBR(384→128) × 2
  up_block_3 : bilinear(×2) + cat(skip1) → CBR(192→64) × 2

Output:
  predictor  : Conv2d(64, 8, 1×1) → (B, 8, 288, 512) sigmoid heatmaps
  Channel index 7 (last) = heatmap for the most-recent (8th) input frame.

Weight loading
──────────────
load_pretrained(model, path) — strict load from ckpt['model'].
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Building blocks ────────────────────────────────────────────────────────────

class _CBR(nn.Module):
    """Conv(3×3) + BN + ReLU.  Attribute names .conv / .bn match checkpoint keys."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _DownBlock(nn.Module):
    """n_convs × CBR → MaxPool2d(2).  Returns (pooled, skip_before_pool)."""

    def __init__(self, in_ch: int, out_ch: int, n_convs: int) -> None:
        super().__init__()
        chs = [in_ch] + [out_ch] * n_convs
        for i in range(n_convs):
            setattr(self, f"conv_{i+1}", _CBR(chs[i], chs[i+1]))
        self._n = n_convs
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        for i in range(self._n):
            x = getattr(self, f"conv_{i+1}")(x)
        return self.pool(x), x   # (pooled, skip)


class _Bottleneck(nn.Module):
    """3 × CBR at the bottom of the U-Net, no pooling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv_1 = _CBR(in_ch,  out_ch)
        self.conv_2 = _CBR(out_ch, out_ch)
        self.conv_3 = _CBR(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_3(self.conv_2(self.conv_1(x)))


class _UpBlock(nn.Module):
    """Bilinear ×2 → concat skip → n_convs × CBR."""

    def __init__(
        self, in_ch_up: int, in_ch_skip: int, out_ch: int, n_convs: int
    ) -> None:
        super().__init__()
        chs = [in_ch_up + in_ch_skip] + [out_ch] * n_convs
        for i in range(n_convs):
            setattr(self, f"conv_{i+1}", _CBR(chs[i], chs[i+1]))
        self._n = n_convs

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        for i in range(self._n):
            x = getattr(self, f"conv_{i+1}")(x)
        return x


# ── Full model ─────────────────────────────────────────────────────────────────

class TrackNetV3(nn.Module):
    """
    TrackNetV3 matching the official pretrained checkpoint (seq_len=8).

    Input  : (B, 27, 288, 512) — [bg, f1..f8] each frame normalized to [0,1]
    Output : (B, 8, 288, 512) sigmoid heatmaps, one per input frame.

    For inference: use heatmap[:, -1, :, :] (last channel = current frame).
    """

    INPUT_H = 288
    INPUT_W = 512
    SEQ_LEN = 8  # matches pretrained weights

    def __init__(self) -> None:
        super().__init__()
        in_ch = (self.SEQ_LEN + 1) * 3   # = 27

        self.down_block_1 = _DownBlock(in_ch, 64,  n_convs=2)
        self.down_block_2 = _DownBlock(64,  128,    n_convs=2)
        self.down_block_3 = _DownBlock(128, 256,    n_convs=3)
        self.bottleneck   = _Bottleneck(256, 512)

        self.up_block_1 = _UpBlock(512, 256, 256, n_convs=3)
        self.up_block_2 = _UpBlock(256, 128, 128, n_convs=2)
        self.up_block_3 = _UpBlock(128,  64,  64, n_convs=2)

        self.predictor = nn.Conv2d(64, self.SEQ_LEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, s1 = self.down_block_1(x)
        x2, s2 = self.down_block_2(x1)
        x3, s3 = self.down_block_3(x2)
        xb     = self.bottleneck(x3)

        d = self.up_block_1(xb, s3)
        d = self.up_block_2(d,  s2)
        d = self.up_block_3(d,  s1)
        return torch.sigmoid(self.predictor(d))


# ── Weight loading ─────────────────────────────────────────────────────────────

def load_pretrained(model: TrackNetV3, weights_path, device: str = "cpu") -> None:
    """Load official TrackNetV3 checkpoint with strict key matching."""
    path = Path(weights_path)
    if not path.exists():
        logger.warning("TrackNet weights not found at %s — running random init", path)
        return

    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = len(state) - len(unexpected)
    if missing:
        logger.warning("Missing keys (%d): %s …", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:5])
    logger.info(
        "TrackNetV3 weights loaded: %d/%d keys  (missing=%d, unexpected=%d)",
        n_loaded, len(state), len(missing), len(unexpected),
    )
