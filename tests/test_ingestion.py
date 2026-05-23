"""
tests/test_ingestion.py

Unit tests for src/ingestion/video_reader.py and src/ingestion/preprocessor.py.
Run with:  pytest tests/test_ingestion.py -v
"""

import numpy as np
import pytest
import torch

from src.ingestion.video_reader import Frame, VideoMetadata, VideoReader
from src.ingestion.preprocessor import Preprocessor, ProcessedFrame


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fake_frame(h: int = 1080, w: int = 1920, idx: int = 0) -> Frame:
    """Create a synthetic Frame with random pixel data."""
    return Frame(
        data=np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
        frame_idx=idx,
        timestamp_sec=idx / 30.0,
    )


# ── VideoReader tests ─────────────────────────────────────────────────────────

class TestVideoReader:
    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            with VideoReader("nonexistent.mp4") as r:
                pass

    def test_len_stride_1(self, tmp_path):
        """__len__ should equal total_frames when stride=1."""
        # We test __len__ logic directly without needing a real video
        reader = VideoReader.__new__(VideoReader)
        from src.ingestion.video_reader import VideoMetadata
        from pathlib import Path
        reader.video_path = Path("fake.mp4")
        reader.stride = 1
        reader.start_frame = 0
        reader.end_frame = None
        reader._cap = None
        reader.metadata = VideoMetadata(
            path=Path("fake.mp4"), fps=30.0, total_frames=300,
            width=1920, height=1080, duration_sec=10.0
        )
        assert len(reader) == 300

    def test_len_stride_3(self):
        """With stride=3 and 300 frames, __len__ should be 100."""
        reader = VideoReader.__new__(VideoReader)
        from src.ingestion.video_reader import VideoMetadata
        from pathlib import Path
        reader.stride = 3
        reader.start_frame = 0
        reader.end_frame = None
        reader._cap = None
        reader.metadata = VideoMetadata(
            path=Path("fake.mp4"), fps=30.0, total_frames=300,
            width=1920, height=1080, duration_sec=10.0
        )
        assert len(reader) == 100

    def test_len_with_start_end(self):
        """start_frame=60, end_frame=90, stride=3 → 10 frames."""
        reader = VideoReader.__new__(VideoReader)
        from src.ingestion.video_reader import VideoMetadata
        from pathlib import Path
        reader.stride = 3
        reader.start_frame = 60
        reader.end_frame = 90
        reader._cap = None
        reader.metadata = VideoMetadata(
            path=Path("fake.mp4"), fps=30.0, total_frames=300,
            width=1920, height=1080, duration_sec=10.0
        )
        assert len(reader) == 10


# ── Preprocessor tests ────────────────────────────────────────────────────────

class TestPreprocessor:

    def test_output_tensor_shape(self):
        """process() should return a (3, 640, 640) tensor for any input size."""
        prep = Preprocessor(input_size=640, device="cpu")
        frame = make_fake_frame(h=1080, w=1920)
        proc = prep.process(frame)
        assert proc.tensor.shape == (3, 640, 640)

    def test_output_dtype_and_range(self):
        """Tensor should be float32 with values in [0, 1]."""
        prep = Preprocessor(input_size=640, device="cpu")
        proc = prep.process(make_fake_frame())
        assert proc.tensor.dtype == torch.float32
        assert proc.tensor.min().item() >= 0.0
        assert proc.tensor.max().item() <= 1.0

    def test_original_bgr_unchanged(self):
        """original_bgr must be the unmodified raw frame data."""
        prep = Preprocessor(input_size=640, device="cpu")
        frame = make_fake_frame()
        original_copy = frame.data.copy()
        proc = prep.process(frame)
        np.testing.assert_array_equal(proc.original_bgr, original_copy)

    def test_metadata_preserved(self):
        """frame_idx and timestamp_sec must pass through unchanged."""
        prep = Preprocessor(input_size=640, device="cpu")
        frame = make_fake_frame(idx=42)
        proc = prep.process(frame)
        assert proc.frame_idx == 42
        assert abs(proc.timestamp_sec - 42 / 30.0) < 1e-6

    def test_letterbox_square_input(self):
        """A square input needs no padding — pad should be (0, 0)."""
        prep = Preprocessor(input_size=640, device="cpu")
        proc = prep.process(make_fake_frame(h=640, w=640))
        assert proc.pad == (0, 0)

    def test_letterbox_wide_input(self):
        """Widescreen (1920×1080): horizontal padding should be 0, vertical > 0."""
        prep = Preprocessor(input_size=640, device="cpu")
        proc = prep.process(make_fake_frame(h=1080, w=1920))
        pad_left, pad_top = proc.pad
        # Width is the longer dimension → no horizontal padding
        assert pad_left == 0
        # Height is shorter → has top padding
        assert pad_top > 0

    def test_batch_shape(self):
        """process_batch on 4 frames should produce a (4, 3, 640, 640) tensor."""
        prep = Preprocessor(input_size=640, device="cpu")
        frames = [make_fake_frame(idx=i) for i in range(4)]
        batch, processed = prep.process_batch(frames)
        assert batch.shape == (4, 3, 640, 640)
        assert len(processed) == 4

    def test_unproject_points_roundtrip(self):
        """
        project a corner point into letterbox space, then unproject it.
        Should recover (approximately) the original pixel coordinate.
        """
        prep = Preprocessor(input_size=640, device="cpu")
        frame = make_fake_frame(h=1080, w=1920)
        proc = prep.process(frame)

        # Top-left corner of original image = (0, 0)
        # In letterbox space, that maps to (pad_left, pad_top)
        pad_left, pad_top = proc.pad
        letterbox_point = np.array([[pad_left, pad_top]], dtype=float)

        original_point = prep.unproject_coords(letterbox_point, proc)
        np.testing.assert_allclose(original_point, [[0.0, 0.0]], atol=1.0)

    def test_unproject_boxes(self):
        """unproject_coords works for (N, 4) box format as well."""
        prep = Preprocessor(input_size=640, device="cpu")
        frame = make_fake_frame(h=1080, w=1920)
        proc = prep.process(frame)

        pad_left, pad_top = proc.pad
        # A box in letterbox space that should map back near (0,0,0,0)
        box = np.array([[pad_left, pad_top, pad_left, pad_top]], dtype=float)
        result = prep.unproject_coords(box, proc)
        np.testing.assert_allclose(result, [[0.0, 0.0, 0.0, 0.0]], atol=1.0)

    def test_unproject_wrong_cols_raises(self):
        """unproject_coords should raise ValueError for unexpected column count."""
        prep = Preprocessor(input_size=640, device="cpu")
        frame = make_fake_frame()
        proc = prep.process(frame)
        with pytest.raises(ValueError):
            prep.unproject_coords(np.array([[1, 2, 3]]), proc)
