"""
PerformanceRNN Model - EXACT Magenta Replication

Architecture (from Magenta's performance_model.py):
1. Input: event_ids [batch, seq_len] integers 0-387 (388 classes)
2. Embedding(388, 256) → dense representation
3. 3×LSTM(512 hidden), dropout=0.3 recurrent
4. Dropout(0.3) on ALL sequence outputs
5. Dense(512→388) on ALL timesteps (teacher forcing)
6. CrossEntropyLoss over full sequence
7. Adam(lr=0.002)

Reference: https://magenta.tensorflow.org/performance-rnn
"""

import os

import lightning as L
import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PerformanceNet(L.LightningModule):
    """
    PerformanceRNN - Exact Magenta architecture replication.

    Key differences from typical implementations:
    - Embedding layer (388 vocab → 256 dim)
    - Predicts ALL timesteps (not just last)
    - Dropout on recurrent connections AND outputs
    - Teacher forcing during training
    """

    def __init__(
        self,
        vocab_size: int = 388,
        embedding_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.002,
        clip_norm: float = 3.0,
    ):
        """
        Args:
            vocab_size: Number of discrete events (default: 388)
            embedding_size: Embedding dimension (default: 256)
            hidden_size: LSTM hidden units (default: 512)
            num_layers: Number of LSTM layers (default: 3)
            dropout: Dropout probability (default: 0.3)
            learning_rate: Adam learning rate (default: 0.002)
            clip_norm: Gradient clipping norm (default: 3.0)
        """
        super().__init__()
        self.save_hyperparameters()

        # Magenta architecture
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # Recurrent dropout
        )

        self.dropout = nn.Dropout(dropout)  # Output dropout

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding

        # Metrics storage
        self._init_metrics()

    def _init_metrics(self):
        """Initialize metric tracking."""
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.train_accuracies = []
        self.val_accuracies = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Event IDs [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Embed event IDs
        embedded = self.embedding(x)  # [batch, seq_len, embedding_size]

        # LSTM (with recurrent dropout)
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_size]

        # Dropout on outputs
        lstm_out = self.dropout(lstm_out)

        # Project to vocabulary (ALL timesteps, not just last)
        logits = self.fc(lstm_out)  # [batch, seq_len, vocab_size]

        return logits

    def _compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Compute loss, accuracy, and perplexity.

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
        """
        # Flatten for loss computation
        logits_flat = logits.view(-1, logits.size(-1))  # [batch*seq_len, vocab_size]
        targets_flat = targets.view(-1)  # [batch*seq_len]

        # Loss
        loss = self.criterion(logits_flat, targets_flat)

        # Accuracy (exclude padding)
        predictions = torch.argmax(logits_flat, dim=-1)
        mask = targets_flat != -1
        correct = (predictions == targets_flat) & mask
        accuracy = correct.sum().float() / mask.sum().float()

        # Perplexity
        perplexity = torch.exp(loss.clamp(max=10))

        return loss, accuracy, perplexity

    def training_step(self, batch, batch_idx):
        """
        Training step with teacher forcing.

        Batch format:
            x: [batch, seq_len] event IDs
            y: [batch, seq_len] target event IDs (shifted by 1)
        """
        x, y = batch

        logits = self(x)  # [batch, seq_len, vocab_size]

        loss, accuracy, perplexity = self._compute_metrics(logits, y)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_ppl", perplexity, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch

        logits = self(x)

        loss, accuracy, perplexity = self._compute_metrics(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_ppl", perplexity, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        """Collect training metrics."""
        metrics = self.trainer.callback_metrics

        if "train_loss" in metrics:
            self.train_losses.append(metrics["train_loss"].item())
        if "train_acc" in metrics:
            self.train_accuracies.append(metrics["train_acc"].item())
        if "train_ppl" in metrics:
            self.train_perplexities.append(metrics["train_ppl"].item())

    def on_validation_epoch_end(self):
        """Collect validation metrics and update plots."""
        metrics = self.trainer.callback_metrics

        if "val_loss" in metrics:
            self.val_losses.append(metrics["val_loss"].item())
        if "val_acc" in metrics:
            self.val_accuracies.append(metrics["val_acc"].item())
        if "val_ppl" in metrics:
            self.val_perplexities.append(metrics["val_ppl"].item())

        self._save_plots()

    def _save_plots(self):
        """Generate and save training metrics plots."""
        if not self.train_losses:
            return

        epochs = range(1, len(self.train_losses) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("PerformanceRNN Training Metrics", fontsize=14)

        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.train_losses, "b-", label="Train", linewidth=1.5)
        if self.val_losses:
            val_epochs = range(1, len(self.val_losses) + 1)
            ax.plot(val_epochs, self.val_losses, "r-", label="Val", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Cross-Entropy Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Perplexity
        ax = axes[0, 1]
        ax.plot(epochs, self.train_perplexities, "b-", label="Train", linewidth=1.5)
        if self.val_perplexities:
            val_epochs = range(1, len(self.val_perplexities) + 1)
            ax.plot(val_epochs, self.val_perplexities, "r-", label="Val", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity (exp(loss))")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[1, 0]
        ax.plot(epochs, self.train_accuracies, "b-", label="Train", linewidth=1.5)
        if self.val_accuracies:
            val_epochs = range(1, len(self.val_accuracies) + 1)
            ax.plot(val_epochs, self.val_accuracies, "r-", label="Val", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Prediction Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Learning rate
        ax = axes[1, 1]
        if (
            hasattr(self.trainer, "lr_scheduler_configs")
            and self.trainer.lr_scheduler_configs
        ):
            ax.text(
                0.5,
                0.5,
                f"LR: {self.hparams.learning_rate}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
        else:
            ax.text(
                0.5, 0.5, "Reserved", ha="center", va="center", transform=ax.transAxes
            )
        ax.set_title("Configuration")

        plt.tight_layout()

        # Save
        save_dir = self.trainer.log_dir if self.trainer.log_dir else "runs"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, "training_metrics.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    def configure_optimizers(self):
        """Configure Adam optimizer (matching Magenta)."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def on_before_optimizer_step(self, optimizer):
        """Apply gradient clipping."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.clip_norm)
