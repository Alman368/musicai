"""
Event Encoding for PerformanceRNN - Magenta's 388-class scheme.

Event types:
- NOTE_ON: 128 events (MIDI pitches 0-127)
- NOTE_OFF: 128 events (MIDI pitches 0-127)
- TIME_SHIFT: 100 events (10ms steps, max 1 second)
- VELOCITY: 32 events (quantized velocity bins)

Total: 388 discrete event classes
Encoding: event_type_offset + event_value = event_id (0-387)
"""

from enum import IntEnum
from typing import List, NamedTuple

import numpy as np


class EventType(IntEnum):
    """Event type identifiers."""

    NOTE_ON = 0
    NOTE_OFF = 1
    TIME_SHIFT = 2
    VELOCITY = 3


class PerformanceEvent(NamedTuple):
    """A single performance event."""

    event_type: EventType
    event_value: int


class EventEncoder:
    """
    Encodes/decodes performance events to/from integer IDs.

    Magenta's 388-class encoding:
        [0-127]:   NOTE_ON (128 MIDI pitches)
        [128-255]: NOTE_OFF (128 MIDI pitches)
        [256-355]: TIME_SHIFT (100 time steps)
        [356-387]: VELOCITY (32 velocity bins)
    """

    # Constants matching Magenta
    NUM_PITCHES = 128
    MAX_SHIFT_STEPS = 100
    NUM_VELOCITY_BINS = 32
    STEPS_PER_SECOND = 100  # 10ms per step

    # Event offsets
    NOTE_ON_OFFSET = 0
    NOTE_OFF_OFFSET = NUM_PITCHES
    TIME_SHIFT_OFFSET = NOTE_OFF_OFFSET + NUM_PITCHES
    VELOCITY_OFFSET = TIME_SHIFT_OFFSET + MAX_SHIFT_STEPS

    VOCAB_SIZE = VELOCITY_OFFSET + NUM_VELOCITY_BINS  # 388

    def __init__(self):
        """Initialize encoder."""
        assert self.VOCAB_SIZE == 388, f"Vocab size must be 388, got {self.VOCAB_SIZE}"

    def encode_event(self, event: PerformanceEvent) -> int:
        """
        Encode a performance event to an integer ID.

        Args:
            event: PerformanceEvent

        Returns:
            event_id: Integer in range [0, 387]
        """
        if event.event_type == EventType.NOTE_ON:
            assert 0 <= event.event_value < self.NUM_PITCHES
            return self.NOTE_ON_OFFSET + event.event_value

        elif event.event_type == EventType.NOTE_OFF:
            assert 0 <= event.event_value < self.NUM_PITCHES
            return self.NOTE_OFF_OFFSET + event.event_value

        elif event.event_type == EventType.TIME_SHIFT:
            assert 0 <= event.event_value < self.MAX_SHIFT_STEPS
            return self.TIME_SHIFT_OFFSET + event.event_value

        elif event.event_type == EventType.VELOCITY:
            assert 0 <= event.event_value < self.NUM_VELOCITY_BINS
            return self.VELOCITY_OFFSET + event.event_value

        else:
            raise ValueError(f"Unknown event type: {event.event_type}")

    def decode_event(self, event_id: int) -> PerformanceEvent:
        """
        Decode an integer ID to a performance event.

        Args:
            event_id: Integer in range [0, 387]

        Returns:
            PerformanceEvent
        """
        assert 0 <= event_id < self.VOCAB_SIZE, f"Invalid event_id: {event_id}"

        if event_id < self.NOTE_OFF_OFFSET:
            # NOTE_ON
            pitch = event_id - self.NOTE_ON_OFFSET
            return PerformanceEvent(EventType.NOTE_ON, pitch)

        elif event_id < self.TIME_SHIFT_OFFSET:
            # NOTE_OFF
            pitch = event_id - self.NOTE_OFF_OFFSET
            return PerformanceEvent(EventType.NOTE_OFF, pitch)

        elif event_id < self.VELOCITY_OFFSET:
            # TIME_SHIFT
            steps = event_id - self.TIME_SHIFT_OFFSET
            return PerformanceEvent(EventType.TIME_SHIFT, steps)

        else:
            # VELOCITY
            velocity_bin = event_id - self.VELOCITY_OFFSET
            return PerformanceEvent(EventType.VELOCITY, velocity_bin)

    def quantize_velocity(self, midi_velocity: int) -> int:
        """
        Quantize MIDI velocity (0-127) to bin (0-31).

        Args:
            midi_velocity: MIDI velocity value

        Returns:
            Velocity bin index
        """
        midi_velocity = np.clip(midi_velocity, 1, 127)  # 0 velocity = note off
        bin_size = 128 / self.NUM_VELOCITY_BINS
        return min(int(midi_velocity / bin_size), self.NUM_VELOCITY_BINS - 1)

    def dequantize_velocity(self, velocity_bin: int) -> int:
        """
        Convert velocity bin back to MIDI velocity.

        Args:
            velocity_bin: Bin index (0-31)

        Returns:
            MIDI velocity (approximate)
        """
        bin_size = 128 / self.NUM_VELOCITY_BINS
        return int((velocity_bin + 0.5) * bin_size)

    def quantize_time(self, time_seconds: float) -> int:
        """
        Quantize time in seconds to time shift steps.

        Args:
            time_seconds: Time in seconds

        Returns:
            Time shift steps (0-99)
        """
        steps = int(time_seconds * self.STEPS_PER_SECOND)
        return min(max(0, steps), self.MAX_SHIFT_STEPS - 1)

    def dequantize_time(self, time_steps: int) -> float:
        """
        Convert time steps back to seconds.

        Args:
            time_steps: Time shift steps

        Returns:
            Time in seconds
        """
        return time_steps / self.STEPS_PER_SECOND

    def encode_sequence(self, events: List[PerformanceEvent]) -> List[int]:
        """
        Encode a sequence of events to event IDs.

        Args:
            events: List of PerformanceEvent

        Returns:
            List of event IDs
        """
        return [self.encode_event(event) for event in events]

    def decode_sequence(self, event_ids: List[int]) -> List[PerformanceEvent]:
        """
        Decode a sequence of event IDs to events.

        Args:
            event_ids: List of event IDs

        Returns:
            List of PerformanceEvent
        """
        return [self.decode_event(eid) for eid in event_ids]

    def __repr__(self) -> str:
        return f"EventEncoder(vocab_size={self.VOCAB_SIZE})"


# Global encoder instance
encoder = EventEncoder()
