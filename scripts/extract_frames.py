"""
scripts/extract_frames.py

Standalone utility: extract and save individual frames from a video clip.
Useful for building annotation datasets or sanity-checking the ingestion module.

Usage
─────
    python scripts/extract_frames.py \\
        --video  data/raw/game1.mp4 \\
        --output data/processed/frames/game1 \\
        --stride 3 \\
        --start  0 \\
        --end    900
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from tqdm import tqdm

from src.ingestion.video_reader import VideoReader
from src.ingestion.preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frames from a video file.")
    p.add_argument("--video",  required=True,  help="Path to input video (.mp4)")
    p.add_argument("--output", required=True,  help="Directory to save extracted frames")
    p.add_argument("--stride", type=int, default=3, help="Save every Nth frame (default: 3)")
    p.add_argument("--start",  type=int, default=0, help="First frame index (default: 0)")
    p.add_argument("--end",    type=int, default=None, help="Last frame index exclusive (default: end of file)")
    p.add_argument("--size",   type=int, default=None, help="If set, also save preprocessed (letterboxed) frames at this size")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    prep = Preprocessor(input_size=args.size, device="cpu") if args.size else None

    with VideoReader(args.video, stride=args.stride, start_frame=args.start, end_frame=args.end) as reader:
        logger.info("Video: %s", reader.metadata)
        logger.info("Will extract %d frames → %s", len(reader), output_dir)

        for frame in tqdm(reader, total=len(reader), unit="frame"):
            # Save raw BGR frame
            fname = output_dir / f"frame_{frame.frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame.data)

            # Optionally save preprocessed version side-by-side
            if prep is not None:
                proc = prep.process(frame)
                # Convert tensor back to uint8 BGR for saving
                rgb_np = (proc.tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
                bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                prep_fname = output_dir / f"frame_{frame.frame_idx:06d}_prep.jpg"
                cv2.imwrite(str(prep_fname), bgr_np)

    logger.info("Done. Frames saved to %s", output_dir)


if __name__ == "__main__":
    main()
