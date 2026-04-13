"""
MIDI Data Module for PerformanceRNN - Event ID based.

Converts MIDI files to sequences of event IDs (0-387) for training.
"""

import math
import os
from pathlib import Path
from typing import List, Optional

import lightning as L
import mido
import numpy as np
import torch
import torch.utils.data

from performance_net.event_encoder import EventEncoder, EventType, PerformanceEvent


class MIDIDataModule(L.LightningDataModule):
    """
    Lightning DataModule for MIDI data using event ID encoding.
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 512,
        batch_size: int = 64,
        stride: int = 256,
        file_ext: str = ".mid,.midi",
        num_workers: int = 0,
        val_split: float = 0.1,
    ) -> None:
        """
        Args:
            data_dir: Directory containing MIDI files
            sequence_length: Length of input sequences (default: 512)
            batch_size: Training batch size (default: 64)
            stride: Stride for sliding window (default: 256, use 512 for no overlap)
            file_ext: Comma-separated file extensions
            num_workers: DataLoader workers
            val_split: Validation split fraction
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.stride = stride
        self.num_workers = num_workers
        self.val_split = val_split

        self.encoder = EventEncoder()

        # Discover MIDI files
        self.midi_files: List[Path] = []
        if os.path.isdir(data_dir):
            allowed_ext = [ext.strip().lower() for ext in file_ext.split(",")]
            for dir_path, _, files in os.walk(data_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in allowed_ext):
                        self.midi_files.append(Path(dir_path) / file)
            print(f"Found {len(self.midi_files)} MIDI files in {data_dir}")
        else:
            print(f"Warning: '{data_dir}' is not a directory")

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train and validation datasets."""
        if not self.midi_files:
            raise ValueError("No MIDI files found")

        # Split files
        n_files = len(self.midi_files)
        n_val = max(1, int(n_files * self.val_split))
        n_train = n_files - n_val

        train_files = self.midi_files[:n_train]
        val_files = self.midi_files[n_train:]

        print(f"Train: {len(train_files)} files, Val: {len(val_files)} files")
        print(f"Vocabulary size: {self.encoder.VOCAB_SIZE}")
        print(f"Sequence length: {self.sequence_length}")

        self.train_dataset = EventIDDataset(
            midi_files=train_files,
            encoder=self.encoder,
            sequence_length=self.sequence_length,
            stride=self.stride,
        )

        self.val_dataset = EventIDDataset(
            midi_files=val_files,
            encoder=self.encoder,
            sequence_length=self.sequence_length,
            stride=self.stride,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class EventIDDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset that yields (input, target) event ID sequences.

    Format:
        input: [seq_len] event IDs
        target: [seq_len] event IDs (shifted by 1 for teacher forcing)
    """

    def __init__(
        self,
        midi_files: List[Path],
        encoder: EventEncoder,
        sequence_length: int,
        stride: int = 256,
    ) -> None:
        super().__init__()
        self.midi_files = midi_files
        self.encoder = encoder
        self.sequence_length = sequence_length
        self.stride = stride

    def __iter__(self):
        # Handle multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            start_idx = 0
            end_idx = len(self.midi_files)
        else:
            per_worker = math.ceil(len(self.midi_files) / worker_info.num_workers)
            start_idx = per_worker * worker_info.id
            end_idx = min(start_idx + per_worker, len(self.midi_files))

        for midi_path in self.midi_files[start_idx:end_idx]:
            yield from self._process_file(midi_path)

    def _process_file(self, midi_path: Path):
        """Convert MIDI file to event ID sequences."""
        try:
            # Convert MIDI to performance events
            events = self._midi_to_events(midi_path)

            if len(events) < self.sequence_length + 1:
                return  # Skip short files

            # Encode to event IDs
            event_ids = self.encoder.encode_sequence(events)
            event_ids = np.array(event_ids, dtype=np.int64)

            # Sliding window with teacher forcing (with stride to reduce data size)
            for i in range(0, len(event_ids) - self.sequence_length, self.stride):
                x = event_ids[i : i + self.sequence_length]
                y = event_ids[i + 1 : i + self.sequence_length + 1]

                yield torch.from_numpy(x), torch.from_numpy(y)

        except Exception as e:
            print(f"Error processing {midi_path}: {e}")

    def _midi_to_events(self, midi_path: Path) -> List[PerformanceEvent]:
        """
        Convert MIDI file to performance events.

        Returns:
            List of PerformanceEvent objects
        """
        midi = mido.MidiFile(midi_path)

        # Collect all note events with absolute time
        note_events = []
        current_time = 0.0

        for msg in midi:
            current_time += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                note_events.append(
                    {
                        "type": "note_on",
                        "pitch": msg.note,
                        "velocity": msg.velocity,
                        "time": current_time,
                    }
                )
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                note_events.append(
                    {
                        "type": "note_off",
                        "pitch": msg.note,
                        "time": current_time,
                    }
                )

        # Sort by time
        note_events.sort(key=lambda x: x["time"])

        # Convert to performance events
        events = []
        prev_time = 0.0
        current_velocity = 64  # Default velocity

        for event in note_events:
            # Time shift
            time_delta = event["time"] - prev_time
            if time_delta > 0.01:  # Minimum 10ms
                time_steps = self.encoder.quantize_time(time_delta)
                # Split large time shifts into multiple events
                while time_steps > 0:
                    steps_to_emit = min(time_steps, self.encoder.MAX_SHIFT_STEPS - 1)
                    events.append(PerformanceEvent(EventType.TIME_SHIFT, steps_to_emit))
                    time_steps -= steps_to_emit

            if event["type"] == "note_on":
                # Velocity change (if different from current)
                velocity_bin = self.encoder.quantize_velocity(event["velocity"])
                if velocity_bin != current_velocity:
                    events.append(PerformanceEvent(EventType.VELOCITY, velocity_bin))
                    current_velocity = velocity_bin

                # Note on
                events.append(PerformanceEvent(EventType.NOTE_ON, event["pitch"]))

            else:  # note_off
                events.append(PerformanceEvent(EventType.NOTE_OFF, event["pitch"]))

            prev_time = event["time"]

        return events
