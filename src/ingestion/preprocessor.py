"""
src/ingestion/preprocessor.py

Prepares raw BGR frames for model inference.

Two things happen here that are critical to get right once and never touch again:

  1. Letterbox resize  — scales the frame to a square (e.g. 640×640) WITHOUT
     distorting aspect ratio.  Empty space is filled with gray (114).  YOLOv8
     expects this exact format.

  2. Coord back-projection — because we resized + padded, any bounding box the
     model returns is in "preprocessed space."  unproject_coords() maps those
     boxes back to the original pixel space so they line up with the raw frame
     for visualization and downstream math.

Typical usage
─────────────
    from src.ingestion.video_reader import VideoReader
    from src.ingestion.preprocessor import Preprocessor

    prep = Preprocessor(input_size=640, device="cpu")

    with VideoReader("game.mp4", stride=3) as reader:
        for frame in reader:
            proc = prep.process(frame)
            # proc.tensor  → (3, 640, 640) float32 tensor ready for YOLO
            # proc.original_bgr → raw frame for visualization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.ingestion.video_reader import Frame

logger = logging.getLogger(__name__)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class ProcessedFrame:
    """
    A preprocessed frame ready for model inference, bundled with everything
    needed to map model outputs back to the original image.

    Attributes
    ──────────
    tensor        : Float32 torch tensor of shape (3, H, W), values in [0, 1],
                    in RGB channel order.  Plug directly into YOLO / torchvision.
    original_bgr  : Untouched raw frame from the video, shape (H, W, 3) uint8.
                    Used for annotation and visualization — never modified.
    frame_idx     : Absolute frame index from the source video.
    timestamp_sec : Frame timestamp in seconds.
    scale         : Uniform scale factor applied when resizing the original frame.
                    original_dim * scale = resized_dim (before padding).
    pad           : (pad_left, pad_top) pixels of letterbox padding added.
                    Needed to invert the coordinate transform.
    orig_shape    : (H, W) of the original frame before any processing.
    """
    tensor: torch.Tensor
    original_bgr: np.ndarray
    frame_idx: int
    timestamp_sec: float
    scale: float
    pad: Tuple[int, int]          # (pad_left_px, pad_top_px)
    orig_shape: Tuple[int, int]   # (orig_H, orig_W)


# ── Preprocessor ─────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Frame preprocessor for YOLO / torchvision models.

    Parameters
    ──────────
    input_size : Target square resolution for model input.
                 640 is the YOLOv8 default and a good starting point.
    device     : Torch device string.  Use "mps" on Apple Silicon,
                 "cuda" on GPU servers, "cpu" otherwise.

    Why letterbox instead of plain resize?
    ───────────────────────────────────────
    A plain resize (e.g. 1920×1080 → 640×640) squashes the aspect ratio.
    Players become short and fat; the model's geometry assumptions break down.
    Letterboxing resizes so the longer side fits 640, then pads the shorter
    side with neutral gray — aspect ratio preserved, model input size met.
    """

    LETTERBOX_FILL = 114   # gray value for padding (YOLOv8 convention)

    def __init__(self, input_size: int = 640, device: str = "cpu") -> None:
        self.input_size = input_size
        self.device = torch.device(device)
        logger.info(
            "Preprocessor ready — input_size=%d, device=%s",
            self.input_size, self.device
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame: Frame) -> ProcessedFrame:
        """
        Preprocess a single Frame.

        Steps
        ─────
        1. Letterbox resize to (input_size × input_size)
        2. BGR → RGB
        3. HWC → CHW  (OpenCV is H,W,C; PyTorch/YOLO expects C,H,W)
        4. uint8 [0,255] → float32 [0,1]
        5. Move to target device
        """
        bgr = frame.data
        orig_h, orig_w = bgr.shape[:2]

        letterboxed, scale, pad = self._letterbox(bgr)

        # Color conversion + axis reorder
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)        # HWC → CHW
            .float()
            .div(255.0)              # [0,255] → [0.0, 1.0]
            .to(self.device)
        )

        return ProcessedFrame(
            tensor=tensor,
            original_bgr=bgr,
            frame_idx=frame.frame_idx,
            timestamp_sec=frame.timestamp_sec,
            scale=scale,
            pad=pad,
            orig_shape=(orig_h, orig_w),
        )

    def process_batch(
        self,
        frames: List[Frame],
    ) -> Tuple[torch.Tensor, List[ProcessedFrame]]:
        """
        Preprocess a list of frames into a single batched tensor.

        Returns
        ───────
        batch_tensor  : shape (B, 3, H, W) — ready for model.forward().
        processed     : list of ProcessedFrame objects in the same order.
                        Keep these; you need them to call unproject_coords()
                        on the model outputs.
        """
        processed = [self.process(f) for f in frames]
        batch_tensor = torch.stack([p.tensor for p in processed], dim=0)
        logger.debug("Batched %d frames → shape %s", len(frames), tuple(batch_tensor.shape))
        return batch_tensor, processed

    def unproject_coords(
        self,
        coords: np.ndarray,
        processed: ProcessedFrame,
    ) -> np.ndarray:
        """
        Map bounding-box coordinates from preprocessed (letterboxed) space
        back to original frame pixel coordinates.

        Parameters
        ──────────
        coords    : numpy array of shape (N, 2) for points [x, y]
                    or (N, 4) for boxes [x1, y1, x2, y2].
                    Coordinates are in the letterboxed input_size space.
        processed : The ProcessedFrame the coords came from (carries scale + pad).

        Returns
        ───────
        Coordinates in original image pixel space, same shape as input.

        Why this matters
        ────────────────
        YOLO outputs boxes in 640×640 space.  To draw them on the raw frame
        or convert them to court coordinates, they must be in original pixel
        space first.  This is the inverse of the letterbox transform.
        """
        pad_left, pad_top = processed.pad
        scale = processed.scale
        out = coords.astype(float).copy()

        if out.ndim == 1:
            out = out[np.newaxis, :]

        if out.shape[1] == 2:
            # (x, y) points
            out[:, 0] = (out[:, 0] - pad_left) / scale
            out[:, 1] = (out[:, 1] - pad_top)  / scale
        elif out.shape[1] == 4:
            # (x1, y1, x2, y2) boxes
            out[:, 0] = (out[:, 0] - pad_left) / scale  # x1
            out[:, 1] = (out[:, 1] - pad_top)  / scale  # y1
            out[:, 2] = (out[:, 2] - pad_left) / scale  # x2
            out[:, 3] = (out[:, 3] - pad_top)  / scale  # y2
        else:
            raise ValueError(
                f"coords must have 2 or 4 columns, got {out.shape[1]}"
            )

        # Clip to valid pixel range
        orig_h, orig_w = processed.orig_shape
        if out.shape[1] == 2:
            out[:, 0] = np.clip(out[:, 0], 0, orig_w - 1)
            out[:, 1] = np.clip(out[:, 1], 0, orig_h - 1)
        else:
            out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w - 1)
            out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h - 1)

        return out

    # ── Private helpers ───────────────────────────────────────────────────────

    def _letterbox(
        self,
        bgr: np.ndarray,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize bgr to (input_size × input_size) with gray padding.

        Returns
        ───────
        canvas  : Letterboxed image of shape (input_size, input_size, 3).
        scale   : Uniform scale factor (same for x and y — no distortion).
        pad     : (pad_left, pad_top) in pixels — needed for unproject_coords().
        """
        h, w = bgr.shape[:2]
        scale = self.input_size / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full(
            (self.input_size, self.input_size, 3),
            self.LETTERBOX_FILL,
            dtype=np.uint8,
        )

        pad_left = (self.input_size - new_w) // 2
        pad_top  = (self.input_size - new_h) // 2

        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        return canvas, scale, (pad_left, pad_top)
