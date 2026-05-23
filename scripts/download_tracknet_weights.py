"""
scripts/download_tracknet_weights.py

Download pretrained TrackNetV3 weights into models/checkpoints/.

The official TrackNet_best.pt is published by qaz812345 (NYCU) alongside the
TrackNetV3 paper.  Get the Google Drive link from:
    https://github.com/qaz812345/TrackNetV3  →  README  →  "Download checkpoints"

Usage
─────
    # Auto-download using gdown (needs the file ID from the Drive link):
    python scripts/download_tracknet_weights.py --file-id <GDRIVE_FILE_ID>

    # If you already downloaded TrackNet_best.pt manually, just copy it:
    python scripts/download_tracknet_weights.py --src /path/to/TrackNet_best.pt

    # Verify an existing checkpoint:
    python scripts/download_tracknet_weights.py --verify
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEST = Path("models/checkpoints/tracknet_best.pt")


def download_gdrive(file_id: str) -> None:
    try:
        import gdown
    except ImportError:
        logger.error("gdown not installed.  Run: pip install gdown")
        sys.exit(1)

    DEST.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info("Downloading from Google Drive → %s", DEST)
    gdown.download(url, str(DEST), quiet=False)


def copy_local(src: str) -> None:
    src_path = Path(src)
    if not src_path.exists():
        logger.error("Source file not found: %s", src_path)
        sys.exit(1)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, DEST)
    logger.info("Copied %s → %s", src_path, DEST)


def verify() -> None:
    if not DEST.exists():
        logger.error("Weights not found at %s", DEST)
        logger.error("Run with --file-id or --src to download/copy them.")
        sys.exit(1)

    import torch
    from src.detection.tracknet import TrackNetV3, load_pretrained

    logger.info("Loading TrackNetV3 with weights from %s ...", DEST)
    model = TrackNetV3()
    load_pretrained(model, DEST, device="cpu")
    model.eval()

    dummy = torch.zeros(1, 27, 288, 512)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 8, 288, 512), f"Unexpected output shape: {out.shape}"
    logger.info("Verification passed — output shape: %s", tuple(out.shape))
    logger.info("TrackNet weights are ready at %s", DEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / verify TrackNetV3 weights")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file-id", help="Google Drive file ID from the TrackNetV3 repo README")
    group.add_argument("--src",     help="Local path to TrackNet_best.pt to copy in")
    parser.add_argument("--verify", action="store_true",
                        help="Load and forward-pass the model to confirm weights work")
    args = parser.parse_args()

    if args.file_id:
        download_gdrive(args.file_id)
    elif args.src:
        copy_local(args.src)

    if args.verify or (not args.file_id and not args.src):
        verify()


if __name__ == "__main__":
    main()
