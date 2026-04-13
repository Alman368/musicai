#!/usr/bin/env python3
"""
Extract and visualize training metrics from checkpoint.

Usage:
    python extract_metrics.py --checkpoint runs/20260105_010848/best.ckpt
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def extract_metrics_from_checkpoint(checkpoint_path: str):
    """Extract metrics from Lightning checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract hyperparameters
    hparams = checkpoint.get("hyper_parameters", {})

    # Try to extract metrics from the model state
    state_dict = checkpoint.get("state_dict", {})

    # Get callback metrics if available
    callbacks = checkpoint.get("callbacks", {})

    print("\n" + "=" * 60)
    print("MODEL HYPERPARAMETERS")
    print("=" * 60)
    for key, value in hparams.items():
        print(f"{key:20s}: {value}")

    print("\n" + "=" * 60)
    print("TRAINING INFO")
    print("=" * 60)
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Global step: {checkpoint.get('global_step', 'N/A')}")

    # Extract final metrics
    if "callbacks" in checkpoint:
        print("\n" + "=" * 60)
        print("CALLBACKS INFO")
        print("=" * 60)
        for cb_name, cb_data in callbacks.items():
            print(f"\n{cb_name}:")
            if isinstance(cb_data, dict):
                for k, v in cb_data.items():
                    print(f"  {k}: {v}")

    return checkpoint


def create_metrics_summary(run_dir: str):
    """Create comprehensive metrics summary from run directory."""
    run_path = Path(run_dir)

    # Check for tensorboard logs
    tb_dir = run_path / "tensorboard"
    if tb_dir.exists():
        print(f"\nTensorBoard logs found at: {tb_dir}")
        print("To view: tensorboard --logdir " + str(tb_dir))

    # Check for training_metrics.png
    metrics_plot = run_path / "training_metrics.png"
    if metrics_plot.exists():
        print(f"\nMetrics plot found: {metrics_plot}")

    # Load best checkpoint
    best_ckpt = run_path / "best.ckpt"
    if best_ckpt.exists():
        checkpoint = extract_metrics_from_checkpoint(str(best_ckpt))
        return checkpoint

    return None


def create_comparison_table(checkpoint):
    """Create a formatted comparison table."""
    hparams = checkpoint.get("hyper_parameters", {})

    table = []
    table.append("\n" + "=" * 80)
    table.append("PERFNET - FINAL TRAINING RESULTS")
    table.append("=" * 80)
    table.append("")
    table.append("ARCHITECTURE:")
    table.append(f"  Model Type:           PerformanceRNN (Magenta replication)")
    table.append(f"  Vocabulary Size:      {hparams.get('vocab_size', 388)}")
    table.append(f"  Embedding Dim:        {hparams.get('embedding_size', 256)}")
    table.append(f"  LSTM Hidden Units:    {hparams.get('hidden_size', 512)}")
    table.append(f"  LSTM Layers:          {hparams.get('num_layers', 3)}")
    table.append(f"  Dropout:              {hparams.get('dropout', 0.3)}")
    table.append("")
    table.append("TRAINING:")
    table.append(f"  Learning Rate:        {hparams.get('learning_rate', 0.002)}")
    table.append(f"  Gradient Clip:        {hparams.get('clip_norm', 3.0)}")
    table.append(f"  Final Epoch:          {checkpoint.get('epoch', 'N/A')}")
    table.append(f"  Total Steps:          {checkpoint.get('global_step', 'N/A')}")
    table.append("")
    table.append("MODEL SIZE:")

    # Calculate model size
    state_dict = checkpoint.get("state_dict", {})
    total_params = 0
    for name, param in state_dict.items():
        if "num_batches_tracked" not in name:
            total_params += param.numel()

    table.append(f"  Total Parameters:     {total_params:,}")
    table.append(f"  Model Size (MB):      {total_params * 4 / (1024**2):.2f}")
    table.append("")
    table.append("=" * 80)

    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description="Extract metrics from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/20260105_010848/best.ckpt",
        help="Path to checkpoint or run directory",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.is_dir():
        # It's a run directory
        print(f"Analyzing run directory: {checkpoint_path}")
        checkpoint = create_metrics_summary(str(checkpoint_path))
    else:
        # It's a checkpoint file
        checkpoint = extract_metrics_from_checkpoint(str(checkpoint_path))

    if checkpoint:
        # Create and print comparison table
        table = create_comparison_table(checkpoint)
        print(table)

        # Save to file
        output_file = "training_results.txt"
        with open(output_file, "w") as f:
            f.write(table)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
