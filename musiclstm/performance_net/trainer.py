"""
PerformanceRNN Trainer configuration.

Wraps PyTorch Lightning Trainer with sensible defaults
matching Magenta's training setup.
"""

import os
from datetime import datetime, timedelta

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger


class PerformanceNetTrainer(L.Trainer):
    """
    Configured trainer for PerformanceRNN.

    Defaults:
        - Gradient clipping at norm 3.0 (matching Magenta)
        - TensorBoard logging
        - Periodic checkpointing
        - Optional early stopping
    """

    def __init__(
        self,
        run_dir: str = "runs",
        gradient_clip_val: float = 3.0,
        save_every_n_minutes: int = 30,
        max_epochs: int = 150,
        enable_early_stopping: bool = False,
        early_stopping_patience: int = 10,
        *args,
        **kwargs,
    ):
        """
        Args:
            run_dir: Base directory for runs
            gradient_clip_val: Gradient clipping threshold (Magenta uses 3.0)
            save_every_n_minutes: Checkpoint interval in minutes
            max_epochs: Maximum training epochs
            enable_early_stopping: Enable early stopping on val_loss
            early_stopping_patience: Epochs to wait before stopping
        """
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = os.path.join(run_dir, timestamp)
        os.makedirs(run_path, exist_ok=True)

        # Callbacks
        callbacks = [
            # Periodic checkpointing
            ModelCheckpoint(
                dirpath=run_path,
                filename="checkpoint-{epoch:03d}-{val_loss:.4f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
                train_time_interval=timedelta(minutes=save_every_n_minutes),
            ),
            # Save best model
            ModelCheckpoint(
                dirpath=run_path,
                filename="best",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            ),
            # Learning rate logging
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # Optional early stopping
        if enable_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    mode="min",
                    verbose=True,
                )
            )

        # Logger
        logger = TensorBoardLogger(
            save_dir=run_path,
            name="tensorboard",
        )

        # Apply configuration
        kwargs["callbacks"] = callbacks
        kwargs["logger"] = logger
        kwargs["gradient_clip_val"] = gradient_clip_val
        kwargs["max_epochs"] = max_epochs
        kwargs["default_root_dir"] = run_path

        super().__init__(*args, **kwargs)

        # Store run path for metrics saving
        self._run_path = run_path

    @property
    def run_path(self) -> str:
        """Get the run directory path."""
        return self._run_path
