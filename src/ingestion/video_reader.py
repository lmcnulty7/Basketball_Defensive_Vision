"""
src/ingestion/video_reader.py

Handles all video file I/O and frame extraction.

Provides a generator-based, context-manager interface so frames stream
one at a time and never load the entire video into memory.

Typical usage
─────────────
    from src.ingestion.video_reader import VideoReader

    with VideoReader("data/raw/game1.mp4", stride=3) as reader:
        print(reader.metadata)           # fps, resolution, duration, etc.
        for frame in reader:
            process(frame.data)          # frame.data is a BGR numpy array
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VideoMetadata:
    """Static properties of an opened video file."""
    path: Path
    fps: float
    total_frames: int
    width: int
    height: int
    duration_sec: float

    def __str__(self) -> str:
        return (
            f"{self.path.name} | "
            f"{self.width}×{self.height} | "
            f"{self.fps:.2f} fps | "
            f"{self.total_frames} frames | "
            f"{self.duration_sec:.1f}s"
        )


@dataclass
class Frame:
    """
    A single decoded video frame with its temporal metadata.

    Attributes
    ──────────
    data          : Raw BGR image as a numpy array, shape (H, W, 3), dtype uint8.
                    BGR is OpenCV's native format; preprocessor.py converts to RGB.
    frame_idx     : Absolute 0-based frame index within the source video.
    timestamp_sec : Timestamp in seconds (frame_idx / fps).
    """
    data: np.ndarray
    frame_idx: int
    timestamp_sec: float


# ── VideoReader ───────────────────────────────────────────────────────────────

class VideoReader:
    """
    Context-manager video reader with configurable frame stride.

    Parameters
    ──────────
    video_path  : Path to the .mp4 (or any OpenCV-supported format).
    stride      : Yield every Nth frame.  stride=3 on a 30 fps broadcast
                  gives ~10 fps throughput — fast enough for event detection
                  while cutting inference cost by 3×.
    start_frame : First frame index to read (0-based).  Useful for skipping
                  pre-game content or processing a specific possession.
    end_frame   : Exclusive upper bound.  None = read to end of file.

    Example
    ───────
        # Process the first 5 minutes of a game at ~10 fps
        with VideoReader("game.mp4", stride=3, end_frame=9000) as reader:
            for frame in reader:
                ...
    """

    def __init__(
        self,
        video_path: str | Path,
        stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.stride = max(1, stride)
        self.start_frame = start_frame
        self.end_frame = end_frame

        self._cap: Optional[cv2.VideoCapture] = None
        self.metadata: Optional[VideoMetadata] = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the video file and populate metadata."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Force FFmpeg backend to avoid VideoToolbox ↔ PyTorch framework conflict
        # on macOS (VideoToolbox holds Metal resources that clash with first
        # YOLO inference, causing a segfault if the capture stays open during track()).
        self._cap = cv2.VideoCapture(str(self.video_path), cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            # Fall back to default backend if FFmpeg is unavailable
            self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"OpenCV could not open video: {self.video_path}")

        fps          = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.metadata = VideoMetadata(
            path=self.video_path,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration_sec=total_frames / fps,
        )

        logger.info("Opened video: %s", self.metadata)

        # Seek to requested start position
        if self.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            logger.debug("Seeked to frame %d", self.start_frame)

    def close(self) -> None:
        """Release the VideoCapture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("VideoCapture released: %s", self.video_path.name)

    # ── Iteration ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Generator[Frame, None, None]:
        """
        Yield Frame objects one at a time, respecting stride and bounds.

        Why a generator?
        ────────────────
        A 30-min NBA broadcast at 1080p is ~50 GB uncompressed.  Generators
        keep memory usage flat at ~1 frame regardless of clip length.
        """
        if self._cap is None:
            raise RuntimeError(
                "VideoReader is not open. Use it as a context manager:\n"
                "  with VideoReader(...) as reader: ..."
            )

        end = self.end_frame if self.end_frame is not None else self.metadata.total_frames
        current_idx = self.start_frame

        while current_idx < end:
            ret, bgr = self._cap.read()
            if not ret:
                logger.debug("Stream ended at frame %d (expected %d)", current_idx, end)
                break

            yield Frame(
                data=bgr,
                frame_idx=current_idx,
                timestamp_sec=current_idx / self.metadata.fps,
            )

            current_idx += self.stride

            # Jump ahead if stride > 1 (avoid decoding skipped frames)
            if self.stride > 1 and current_idx < end:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)

    def __len__(self) -> int:
        """Number of frames this reader will yield given current stride and bounds."""
        if self.metadata is None:
            return 0
        end = self.end_frame if self.end_frame is not None else self.metadata.total_frames
        span = max(0, end - self.start_frame)
        return (span + self.stride - 1) // self.stride  # ceiling division

    # ── Random access ─────────────────────────────────────────────────────────

    def read_frame(self, frame_idx: int) -> Optional[Frame]:
        """
        Read a single specific frame by its absolute index.

        Useful for loading a frame when an event is detected and you want
        the exact frame for annotation or debugging — without re-iterating.

        Returns None if the seek or read fails.
        """
        if self._cap is None:
            raise RuntimeError("VideoReader is not open.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = self._cap.read()
        if not ret:
            logger.warning("Could not read frame %d from %s", frame_idx, self.video_path.name)
            return None

        return Frame(
            data=bgr,
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / self.metadata.fps,
        )
