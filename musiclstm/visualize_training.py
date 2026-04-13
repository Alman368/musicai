#!/usr/bin/env python3
"""
Visualize training metrics from TensorBoard logs.

Usage:
    python visualize_training.py --logdir runs/20260105_010848/tensorboard
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(logdir: str):
    """Load data from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    tags = ea.Tags()
    print("Available metrics:")
    for tag in tags["scalars"]:
        print(f"  - {tag}")

    metrics = {}
    for tag in tags["scalars"]:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        metrics[tag] = {"steps": steps, "values": values}

    return metrics


def create_final_metrics_chart(
    logdir: str, output_path: str = "final_training_metrics.png"
):
    """Create comprehensive training metrics visualization."""

    logdir_path = Path(logdir)
    event_files = list(logdir_path.rglob("events.out.tfevents.*"))

    if not event_files:
        print(f"No TensorBoard events found in {logdir}")
        return

    print(f"Loading from: {event_files[0].parent}")
    metrics = load_tensorboard_data(str(event_files[0].parent))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "PerformanceRNN - Final Training Results", fontsize=16, fontweight="bold"
    )

    # Calculate steps per epoch
    if "train_loss" in metrics:
        steps_per_epoch = max(metrics["train_loss"]["steps"]) / 150
    else:
        steps_per_epoch = 1585  # fallback

    # Store final values for summary
    final_train_loss = None
    final_val_loss = None
    final_train_acc = None
    final_val_acc = None
    final_train_ppl = None
    final_val_ppl = None

    # Plot 1: Loss
    ax = axes[0, 0]
    if "train_loss" in metrics:
        train_steps = metrics["train_loss"]["steps"]
        train_values = metrics["train_loss"]["values"]
        train_epochs = [s / steps_per_epoch for s in train_steps]
        ax.plot(
            train_epochs, train_values, "b-", label="Train", alpha=0.7, linewidth=1.5
        )
        final_train_loss = train_values[-1]

    if "val_loss" in metrics:
        val_steps = metrics["val_loss"]["steps"]
        val_values = metrics["val_loss"]["values"]
        val_epochs = [s / steps_per_epoch for s in val_steps]
        ax.plot(
            val_epochs, val_values, "r-", label="Validation", alpha=0.7, linewidth=2
        )
        final_val_loss = val_values[-1]

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if final_val_loss:
        ax.text(
            0.98,
            0.98,
            f"Final Val Loss: {final_val_loss:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 2: Perplexity
    ax = axes[0, 1]
    if "train_ppl" in metrics:
        train_steps = metrics["train_ppl"]["steps"]
        train_values = metrics["train_ppl"]["values"]
        train_epochs = [s / steps_per_epoch for s in train_steps]
        ax.plot(
            train_epochs, train_values, "b-", label="Train", alpha=0.7, linewidth=1.5
        )
        final_train_ppl = train_values[-1]

    if "val_ppl" in metrics:
        val_steps = metrics["val_ppl"]["steps"]
        val_values = metrics["val_ppl"]["values"]
        val_epochs = [s / steps_per_epoch for s in val_steps]
        ax.plot(
            val_epochs, val_values, "r-", label="Validation", alpha=0.7, linewidth=2
        )
        final_val_ppl = val_values[-1]

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Perplexity", fontsize=11)
    ax.set_title("Perplexity (exp(loss))", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if final_val_ppl:
        ax.text(
            0.98,
            0.98,
            f"Final Val Perplexity: {final_val_ppl:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 3: Accuracy
    ax = axes[1, 0]
    if "train_acc" in metrics:
        train_steps = metrics["train_acc"]["steps"]
        train_values = metrics["train_acc"]["values"]
        train_epochs = [s / steps_per_epoch for s in train_steps]
        ax.plot(
            train_epochs, train_values, "b-", label="Train", alpha=0.7, linewidth=1.5
        )
        final_train_acc = train_values[-1]

    if "val_acc" in metrics:
        val_steps = metrics["val_acc"]["steps"]
        val_values = metrics["val_acc"]["values"]
        val_epochs = [s / steps_per_epoch for s in val_steps]
        ax.plot(
            val_epochs, val_values, "r-", label="Validation", alpha=0.7, linewidth=2
        )
        final_val_acc = val_values[-1]

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Prediction Accuracy", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    if final_val_acc:
        ax.text(
            0.98,
            0.02,
            f"Final Val Accuracy: {final_val_acc:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis("off")

    # Build summary text with CORRECT values
    summary_text = "FINAL METRICS\n" + "=" * 30 + "\n\n"

    if final_train_loss and final_val_loss:
        summary_text += f"Train Loss:  {final_train_loss:.4f}\n"
        summary_text += f"Val Loss:    {final_val_loss:.4f}\n\n"

    if final_train_acc and final_val_acc:
        summary_text += f"Train Acc:   {final_train_acc:.1%}\n"
        summary_text += f"Val Acc:     {final_val_acc:.1%}\n\n"

    if final_train_ppl and final_val_ppl:
        summary_text += f"Train PPL:   {final_train_ppl:.2f}\n"
        summary_text += f"Val PPL:     {final_val_ppl:.2f}\n\n"

    summary_text += "=" * 30 + "\n\n"
    summary_text += "ARCHITECTURE\n"
    summary_text += "-" * 30 + "\n"
    summary_text += "Embedding:   388 → 256\n"
    summary_text += "LSTM:        3 × 512 units\n"
    summary_text += "Dropout:     0.3\n"
    summary_text += "Params:      6.08M\n\n"
    summary_text += "TRAINING\n"
    summary_text += "-" * 30 + "\n"
    summary_text += "Epochs:      150\n"
    summary_text += "Steps:       237,750\n"
    summary_text += "LR:          0.002\n"
    summary_text += "Batch:       64\n"

    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFinal metrics chart saved to: {output_path}")

    return {
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "train_acc": final_train_acc,
        "val_acc": final_val_acc,
        "train_ppl": final_train_ppl,
        "val_ppl": final_val_ppl,
    }


def print_final_metrics(final_metrics):
    """Print formatted final metrics."""
    print("\n" + "=" * 60)
    print("FINAL TRAINING METRICS")
    print("=" * 60)

    if final_metrics["val_loss"]:
        print(f"Validation Loss:       {final_metrics['val_loss']:.4f}")
    if final_metrics["train_loss"]:
        print(f"Training Loss:         {final_metrics['train_loss']:.4f}")
    if final_metrics["val_acc"]:
        print(f"Validation Accuracy:   {final_metrics['val_acc']:.2%}")
    if final_metrics["train_acc"]:
        print(f"Training Accuracy:     {final_metrics['train_acc']:.2%}")
    if final_metrics["val_ppl"]:
        print(f"Validation Perplexity: {final_metrics['val_ppl']:.2f}")
    if final_metrics["train_ppl"]:
        print(f"Training Perplexity:   {final_metrics['train_ppl']:.2f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs/20260105_010848/tensorboard",
        help="Path to TensorBoard log directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_training_metrics.png",
        help="Output plot filename",
    )
    args = parser.parse_args()

    final_metrics = create_final_metrics_chart(args.logdir, args.output)

    if final_metrics:
        print_final_metrics(final_metrics)


if __name__ == "__main__":
    main()
