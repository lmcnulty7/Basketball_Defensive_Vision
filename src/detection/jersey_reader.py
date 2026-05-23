"""
src/detection/jersey_reader.py

Reads jersey numbers from player crops using EasyOCR.

EasyOCR is purpose-built for text extraction, fast on MPS/CPU, and has no
PyTorch version constraints.  It reliably reads large printed numbers on
jerseys even at broadcast resolution.

When PyTorch is upgraded to ≥ 2.4, SmolVLM2 can replace EasyOCR here for
better handling of partial occlusions and unusual jersey fonts.

Algorithm
─────────
Every READ_INTERVAL frames, for each tracked player:
  1. Crop the jersey region (upper torso where number is printed).
  2. Run EasyOCR on the crop.
  3. Filter results to 1-2 digit numbers in [0, 99].
  4. Accumulate readings; lock the most common number after LOCK_THRESHOLD
     consistent reads.  Once locked, stop reading that track.

Usage
─────
  reader = JerseyReader()
  for frame in video:
      reader.update(player_tracks, bgr, frame_idx)
      jersey = reader.get_number(track_id)  # "23" or None
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.tracking.multi_tracker import Track

logger = logging.getLogger(__name__)

READ_INTERVAL  = 90     # attempt OCR every N frames per track
LOCK_THRESHOLD = 3      # consistent reads before locking


class JerseyReader:
    """
    Reads jersey numbers using EasyOCR.  Falls back gracefully if
    EasyOCR is unavailable, so the pipeline runs unchanged.
    """

    def __init__(self, device: str = "mps") -> None:
        self._reader  = None
        self._ready   = False
        self._device  = device

        # Per track state
        self._readings:  Dict[int, List[str]] = defaultdict(list)
        self._locked:    Dict[int, str]       = {}
        self._last_read: Dict[int, int]       = {}

        self._load()

    def _load(self) -> None:
        try:
            import easyocr
            # Use GPU if available (MPS not directly supported by EasyOCR, uses CPU)
            gpu = self._device in ("cuda",)
            self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
            self._ready  = True
            logger.info("JerseyReader: EasyOCR ready (gpu=%s)", gpu)
        except Exception as e:
            logger.warning("JerseyReader: EasyOCR unavailable (%s) — jersey reading disabled", e)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_number(self, track_id: int) -> Optional[str]:
        """Return the locked jersey number for a track, or None."""
        return self._locked.get(track_id)

    def update(
        self,
        player_tracks: List[Track],
        bgr: np.ndarray,
        frame_idx: int,
    ) -> None:
        if not self._ready:
            return

        for track in player_tracks:
            tid = track.track_id
            if tid in self._locked:
                continue
            if frame_idx - self._last_read.get(tid, -9999) < READ_INTERVAL:
                continue

            self._last_read[tid] = frame_idx
            crop = self._jersey_crop(bgr, track.bbox)
            if crop is None:
                continue

            number = self._ocr_number(crop)
            if number is not None:
                self._readings[tid].append(number)
                if len(self._readings[tid]) >= LOCK_THRESHOLD:
                    best, count = Counter(self._readings[tid]).most_common(1)[0]
                    if count >= LOCK_THRESHOLD - 1:
                        self._locked[tid] = best
                        logger.debug("Jersey locked: track #%d → #%s", tid, best)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _jersey_crop(bgr: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Crop the jersey number region (upper-mid torso)."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w_frame = bgr.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_frame, x2); y2 = min(h, y2)

        bh = y2 - y1
        bw = x2 - x1
        if bh < 20 or bw < 10:
            return None

        # Jersey number sits in top 25–65% of the bounding box, center columns
        cy1 = y1 + int(bh * 0.20)
        cy2 = y1 + int(bh * 0.65)
        cx1 = x1 + int(bw * 0.15)
        cx2 = x1 + int(bw * 0.85)

        crop = bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        # Upscale for better OCR accuracy
        th = 80
        if (cy2 - cy1) < th:
            scale = th / max(cy2 - cy1, 1)
            crop = cv2.resize(
                crop,
                (int((cx2 - cx1) * scale), th),
                interpolation=cv2.INTER_CUBIC,
            )
        return crop

    def _ocr_number(self, bgr_crop: np.ndarray) -> Optional[str]:
        """Run EasyOCR and extract a valid jersey number."""
        try:
            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            results = self._reader.readtext(
                rgb,
                allowlist="0123456789",
                min_size=8,
                text_threshold=0.6,
                low_text=0.3,
            )
            candidates = []
            for _, text, conf in results:
                text = text.strip()
                if re.fullmatch(r"\d{1,2}", text):
                    num = int(text)
                    if 0 <= num <= 99:
                        candidates.append((conf, text))

            if candidates:
                # Return highest-confidence reading
                return sorted(candidates, reverse=True)[0][1]
        except Exception as e:
            logger.debug("JerseyReader OCR failed: %s", e)
        return None
