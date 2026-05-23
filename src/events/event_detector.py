"""
src/events/event_detector.py

BaseEventDetector — abstract interface every detector implements.
EventOrchestrator — calls all detectors each frame, deduplicates, logs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional

from src.events.event import Event, EventType, PipelineState
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)


# ── Base class ─────────────────────────────────────────────────────────────────

class BaseEventDetector(ABC):
    """
    Abstract base for all defensive event detectors.

    Each detector is stateful (tracks history across frames) and exposes
    a single update() method called by the orchestrator every frame.
    """

    @abstractmethod
    def update(
        self,
        state: PipelineState,
        manager: TrackManager,
    ) -> List[Event]:
        """
        Process one frame and return any events that fired.

        Parameters
        ──────────
        state   : Current frame's pipeline state (all detections, teams, etc.)
        manager : TrackManager for accessing multi-frame position history.

        Returns
        ───────
        List of Event objects (empty list if nothing fired this frame).
        """

    def reset(self) -> None:
        """Reset internal state (call at game breaks, camera cuts, etc.)."""


# ── Orchestrator ───────────────────────────────────────────────────────────────

class EventOrchestrator:
    """
    Runs all event detectors each frame and maintains the event log.

    Parameters
    ──────────
    detectors          : List of BaseEventDetector instances to run.
    dedup_window_frames: Within this window, duplicate event types for the
                         same primary player are suppressed.  Prevents the
                         same steal/block from firing on every adjacent frame.

    Example
    ───────
        orchestrator = EventOrchestrator([
            StealDetector(),
            BlockDetector(),
            DeflectionDetector(),
            ContestDetector(),
            ReboundDetector(),
        ])
        for frame in video:
            state = build_pipeline_state(frame, ...)
            new_events = orchestrator.update(state, manager)
    """

    def __init__(
        self,
        detectors: List[BaseEventDetector],
        dedup_window_frames: int = 10,
    ) -> None:
        self.detectors           = detectors
        self.dedup_window        = dedup_window_frames
        self.event_log: List[Event] = []

        # (event_type, primary_player_id) → last frame_idx it fired
        self._last_fired: Dict[tuple, int] = defaultdict(lambda: -9999)

    def update(
        self,
        state: PipelineState,
        manager: TrackManager,
    ) -> List[Event]:
        """
        Run all detectors and return deduplicated new events.

        Deduplication rule: the same (event_type, primary_player_id) pair
        can fire at most once per dedup_window frames.  Each individual
        detector also has its own cooldown; deduplication here is a second
        safety net for edge-case double-fires.
        """
        new_events: List[Event] = []

        for detector in self.detectors:
            try:
                fired = detector.update(state, manager)
            except Exception as e:
                logger.error(
                    "%s.update() raised: %s",
                    type(detector).__name__, e, exc_info=True,
                )
                continue

            for event in fired:
                key = (event.event_type, event.primary_player_id)
                if state.frame_idx - self._last_fired[key] >= self.dedup_window:
                    new_events.append(event)
                    self._last_fired[key] = state.frame_idx

        self.event_log.extend(new_events)

        if new_events:
            for e in new_events:
                logger.info(
                    "EVENT  %-18s  defender=#%-3d  t=%.2fs  conf=%.2f  %s",
                    e.event_type.value,
                    e.primary_player_id,
                    e.timestamp_sec,
                    e.confidence,
                    e.metadata,
                )

        return new_events

    def reset(self) -> None:
        """Clear log and reset all detectors (call at halftime/possession reset)."""
        self.event_log.clear()
        self._last_fired.clear()
        for d in self.detectors:
            d.reset()

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        return [e for e in self.event_log if e.event_type == event_type]

    def get_events_for_player(self, track_id: int) -> List[Event]:
        return [e for e in self.event_log if e.primary_player_id == track_id]

    def get_stat_counts(self, track_id: int) -> Dict[str, int]:
        """Return a {event_type_name: count} dict for a player's defensive stats."""
        counts: Dict[str, int] = defaultdict(int)
        for e in self.get_events_for_player(track_id):
            counts[e.event_type.value] += 1
        return dict(counts)

    def summary(self) -> Dict[str, int]:
        """Total counts per event type across all players."""
        counts: Dict[str, int] = defaultdict(int)
        for e in self.event_log:
            counts[e.event_type.value] += 1
        return dict(counts)
