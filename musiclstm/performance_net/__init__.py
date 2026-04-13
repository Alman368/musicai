"""
PerformanceRNN - PyTorch Lightning Implementation

Exact replication of Magenta's PerformanceRNN architecture.
Uses event ID encoding (388 classes) with embedding layer.

Reference: https://magenta.tensorflow.org/performance-rnn
"""

from .data import MIDIDataModule
from .event_encoder import EventEncoder, EventType, PerformanceEvent
from .model import PerformanceNet
from .trainer import PerformanceNetTrainer

__all__ = [
    "MIDIDataModule",
    "PerformanceNet",
    "PerformanceNetTrainer",
    "EventEncoder",
    "EventType",
    "PerformanceEvent",
]

__version__ = "2.0.0"
