#!/usr/bin/env python3
"""
Visualize training metrics from TensorBoard logs or CSV results.

Usage:
    python visualize_training.py --logdir saved_models/tensorboard
    python visualize_training.py --csv saved_models/results/results.csv
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt

# Try to import tensorboard, but CSV fallback is available
try:
    from tensorboard.backend.event_processing import event_accumulator

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def load_tensorboard_data(logdir: str):
    """Load data from TensorBoard logs."""
    if not HAS_TENSORBOARD:
        print("TensorBoard not available, use --csv instead")
        return None

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


def load_csv_data(csv_path: str):
    """Load data from CSV results file."""
    metrics = {
        "train_loss": {"steps": [], "values": []},
        "val_loss": {"steps": [], "values": []},
        "train_acc": {"steps": [], "values": []},
        "val_acc": {"steps": [], "values": []},
        "train_ppl": {"steps": [], "values": []},
        "val_ppl": {"steps": [], "values": []},
    }

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) >= 8:
                # New format with perplexity
                epoch = int(row[0])
                train_loss = float(row[2])
                train_acc = float(row[3])
                train_ppl = float(row[4])
                val_loss = float(row[5])
                val_acc = float(row[6])
                val_ppl = float(row[7])
            elif len(row) >= 6:
                # Old format without perplexity
                epoch = int(row[0])
                train_loss = float(row[2])
                train_acc = float(row[3])
                val_loss = float(row[4])
                val_acc = float(row[5])
                # Calculate perplexity from loss
                import math

                train_ppl = math.exp(train_loss) if train_loss < 100 else float("inf")
                val_ppl = math.exp(val_loss) if val_loss < 100 else float("inf")
            else:
                continue

            metrics["train_loss"]["steps"].append(epoch)
            metrics["train_loss"]["values"].append(train_loss)
            metrics["val_loss"]["steps"].append(epoch)
            metrics["val_loss"]["values"].append(val_loss)
            metrics["train_acc"]["steps"].append(epoch)
            metrics["train_acc"]["values"].append(train_acc)
            metrics["val_acc"]["steps"].append(epoch)
            metrics["val_acc"]["values"].append(val_acc)
            metrics["train_ppl"]["steps"].append(epoch)
            metrics["train_ppl"]["values"].append(train_ppl)
            metrics["val_ppl"]["steps"].append(epoch)
            metrics["val_ppl"]["values"].append(val_ppl)

    return metrics


def create_final_metrics_chart(
    metrics: dict,
    output_path: str = "final_training_metrics.png",
    model_params: dict = None,
):
    """Create comprehensive training metrics visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "MusicTransformer - Final Training Results", fontsize=16, fontweight="bold"
    )

    # Store final values for summary
    final_train_loss = None
    final_val_loss = None
    final_train_acc = None
    final_val_acc = None
    final_train_ppl = None
    final_val_ppl = None

    # Plot 1: Loss
    ax = axes[0, 0]
    if "train_loss" in metrics and metrics["train_loss"]["values"]:
        epochs = metrics["train_loss"]["steps"]
        values = metrics["train_loss"]["values"]
        ax.plot(epochs, values, "b-", label="Train", alpha=0.7, linewidth=1.5)
        final_train_loss = values[-1]
    elif "Avg_CE_loss/train" in metrics:
        epochs = metrics["Avg_CE_loss/train"]["steps"]
        values = metrics["Avg_CE_loss/train"]["values"]
        ax.plot(epochs, values, "b-", label="Train", alpha=0.7, linewidth=1.5)
        final_train_loss = values[-1]

    if "val_loss" in metrics and metrics["val_loss"]["values"]:
        epochs = metrics["val_loss"]["steps"]
        values = metrics["val_loss"]["values"]
        ax.plot(epochs, values, "r-", label="Validation", alpha=0.7, linewidth=2)
        final_val_loss = values[-1]
    elif "Avg_CE_loss/eval" in metrics:
        epochs = metrics["Avg_CE_loss/eval"]["steps"]
        values = metrics["Avg_CE_loss/eval"]["values"]
        ax.plot(epochs, values, "r-", label="Validation", alpha=0.7, linewidth=2)
        final_val_loss = values[-1]

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
    if "train_ppl" in metrics and metrics["train_ppl"]["values"]:
        epochs = metrics["train_ppl"]["steps"]
        values = metrics["train_ppl"]["values"]
        ax.plot(epochs, values, "b-", label="Train", alpha=0.7, linewidth=1.5)
        final_train_ppl = values[-1]
    elif "Perplexity/train" in metrics:
        epochs = metrics["Perplexity/train"]["steps"]
        values = metrics["Perplexity/train"]["values"]
        ax.plot(epochs, values, "b-", label="Train", alpha=0.7, linewidth=1.5)
        final_train_ppl = values[-1]

    if "val_ppl" in metrics and metrics["val_ppl"]["values"]:
        epochs = metrics["val_ppl"]["steps"]
        values = metrics["val_ppl"]["values"]
        ax.plot(epochs, values, "r-", label="Validation", alpha=0.7, linewidth=2)
        final_val_ppl = values[-1]
    elif "Perplexity/eval" in metrics:
        epochs = metrics["Perplexity/eval"]["steps"]
        values = metrics["Perplexity/eval"]["values"]
        ax.plot(epochs, values, "r-", label="Validation", alpha=0.7, linewidth=2)
        final_val_ppl = values[-1]

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
    if "train_acc" in metrics and metrics["train_acc"]["values"]:
        epochs = metrics["train_acc"]["steps"]
        values = metrics["train_acc"]["values"]
        ax.plot(epochs, values, "b-", label="Train", alpha=0.7, linewidth=1.5)
        final_train_acc = values[-1]
    elif "Accuracy/train" in metrics:
        epochs = metrics["Accuracy/train"]["steps"]
        values = metrics["Accuracy/train"]["values"]
        ax.plot(epochs, values, "b-", label="Train", alpha=0.7, linewidth=1.5)
        final_train_acc = values[-1]

    if "val_acc" in metrics and metrics["val_acc"]["values"]:
        epochs = metrics["val_acc"]["steps"]
        values = metrics["val_acc"]["values"]
        ax.plot(epochs, values, "r-", label="Validation", alpha=0.7, linewidth=2)
        final_val_acc = values[-1]
    elif "Accuracy/eval" in metrics:
        epochs = metrics["Accuracy/eval"]["steps"]
        values = metrics["Accuracy/eval"]["values"]
        ax.plot(epochs, values, "r-", label="Validation", alpha=0.7, linewidth=2)
        final_val_acc = values[-1]

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

    if model_params:
        summary_text += f"d_model:     {model_params.get('d_model', 256)}\n"
        summary_text += f"n_layers:    {model_params.get('n_layers', 4)}\n"
        summary_text += f"num_heads:   {model_params.get('num_heads', 8)}\n"
        summary_text += f"dim_ff:      {model_params.get('dim_feedforward', 1024)}\n"
        summary_text += f"max_seq:     {model_params.get('max_sequence', 512)}\n"
        summary_text += f"Dropout:     {model_params.get('dropout', 0.1)}\n"
    else:
        summary_text += "Transformer (decoder-only)\n"
        summary_text += "with RPR attention\n"

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


def load_model_params(output_dir: str):
    """Load model parameters from model_params.txt."""
    params_file = os.path.join(output_dir, "model_params.txt")
    params = {}

    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    # Try to convert to appropriate type
                    if value.lower() == "true":
                        params[key] = True
                    elif value.lower() == "false":
                        params[key] = False
                    elif value.lower() == "none":
                        params[key] = None
                    else:
                        try:
                            params[key] = int(value)
                        except ValueError:
                            try:
                                params[key] = float(value)
                            except ValueError:
                                params[key] = value

    return params


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Path to TensorBoard log directory",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to results.csv file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_models",
        help="Output directory (for finding model_params.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_training_metrics.png",
        help="Output plot filename",
    )
    args = parser.parse_args()

    metrics = None

    # Try to auto-detect data source
    if args.csv:
        print(f"Loading from CSV: {args.csv}")
        metrics = load_csv_data(args.csv)
    elif args.logdir:
        print(f"Loading from TensorBoard: {args.logdir}")
        logdir_path = Path(args.logdir)
        event_files = list(logdir_path.rglob("events.out.tfevents.*"))
        if event_files:
            metrics = load_tensorboard_data(str(event_files[0].parent))
    else:
        # Auto-detect: try CSV first, then tensorboard
        csv_path = os.path.join(args.output_dir, "results", "results.csv")
        tb_path = os.path.join(args.output_dir, "tensorboard")

        if os.path.exists(csv_path):
            print(f"Auto-detected CSV: {csv_path}")
            metrics = load_csv_data(csv_path)
        elif os.path.exists(tb_path):
            print(f"Auto-detected TensorBoard: {tb_path}")
            event_files = list(Path(tb_path).rglob("events.out.tfevents.*"))
            if event_files:
                metrics = load_tensorboard_data(str(event_files[0].parent))

    if metrics is None:
        print("Error: Could not load metrics. Specify --csv or --logdir")
        return

    # Load model parameters
    model_params = load_model_params(args.output_dir)

    final_metrics = create_final_metrics_chart(metrics, args.output, model_params)

    if final_metrics:
        print_final_metrics(final_metrics)


if __name__ == "__main__":
    main()
