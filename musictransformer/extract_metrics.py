#!/usr/bin/env python3
"""
Extract and display training metrics from saved model directory.

Usage:
    python extract_metrics.py --output_dir saved_models
    python extract_metrics.py --weights saved_models/results/best_loss_weights.pickle
"""

import argparse
import csv
import math
import os
from pathlib import Path

import torch


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


def load_best_epochs(output_dir: str):
    """Load best epoch information from best_epochs.txt."""
    best_file = os.path.join(output_dir, "results", "best_epochs.txt")
    best_info = {}

    if os.path.exists(best_file):
        with open(best_file, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    key = key.strip().replace(" ", "_")
                    value = value.strip()
                    try:
                        best_info[key] = float(value)
                    except ValueError:
                        best_info[key] = value

    return best_info


def load_csv_results(output_dir: str):
    """Load results from CSV file."""
    csv_file = os.path.join(output_dir, "results", "results.csv")
    results = []

    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                results.append(row)

    return results


def count_parameters(state_dict):
    """Count total parameters in model state dict."""
    total = 0
    for name, param in state_dict.items():
        total += param.numel()
    return total


def extract_metrics_from_weights(weights_path: str):
    """Extract information from model weights file."""
    print(f"Loading weights: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")

    total_params = count_parameters(state_dict)

    print("\n" + "=" * 60)
    print("MODEL WEIGHTS INFO")
    print("=" * 60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size (MB):  {total_params * 4 / (1024**2):.2f}")
    print("=" * 60)

    return {"total_params": total_params}


def create_metrics_summary(output_dir: str):
    """Create comprehensive metrics summary from output directory."""
    print(f"\nAnalyzing: {output_dir}")
    print("=" * 60)

    # Load model parameters
    params = load_model_params(output_dir)
    if params:
        print("\nMODEL PARAMETERS:")
        print("-" * 40)
        for key, value in params.items():
            print(f"  {key}: {value}")

    # Load best epochs info
    best_info = load_best_epochs(output_dir)
    if best_info:
        print("\nBEST RESULTS:")
        print("-" * 40)
        for key, value in best_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    # Load CSV results and get final metrics
    results = load_csv_results(output_dir)
    if results:
        last_row = results[-1]
        print("\nFINAL EPOCH METRICS:")
        print("-" * 40)

        if len(last_row) >= 8:
            # New format with perplexity
            epoch = int(last_row[0])
            lr = float(last_row[1])
            train_loss = float(last_row[2])
            train_acc = float(last_row[3])
            train_ppl = float(last_row[4])
            val_loss = float(last_row[5])
            val_acc = float(last_row[6])
            val_ppl = float(last_row[7])
        elif len(last_row) >= 6:
            # Old format
            epoch = int(last_row[0])
            lr = float(last_row[1])
            train_loss = float(last_row[2])
            train_acc = float(last_row[3])
            val_loss = float(last_row[4])
            val_acc = float(last_row[5])
            train_ppl = math.exp(train_loss) if train_loss < 100 else float("inf")
            val_ppl = math.exp(val_loss) if val_loss < 100 else float("inf")
        else:
            print("  Unexpected CSV format")
            return None

        print(f"  Epoch:           {epoch}")
        print(f"  Learning Rate:   {lr:.6f}")
        print(f"  Train Loss:      {train_loss:.4f}")
        print(f"  Train Accuracy:  {train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"  Train Perplexity:{train_ppl:.2f}")
        print(f"  Val Loss:        {val_loss:.4f}")
        print(f"  Val Accuracy:    {val_acc:.4f} ({val_acc * 100:.2f}%)")
        print(f"  Val Perplexity:  {val_ppl:.2f}")

    # Check for weights files
    results_dir = os.path.join(output_dir, "results")
    weights_dir = os.path.join(output_dir, "weights")

    print("\nAVAILABLE CHECKPOINTS:")
    print("-" * 40)

    if os.path.exists(os.path.join(results_dir, "best_loss_weights.pickle")):
        print("  - best_loss_weights.pickle")
    if os.path.exists(os.path.join(results_dir, "best_acc_weights.pickle")):
        print("  - best_acc_weights.pickle")

    if os.path.exists(weights_dir):
        weight_files = sorted(
            [f for f in os.listdir(weights_dir) if f.endswith(".pickle")]
        )
        if weight_files:
            print(f"  - {len(weight_files)} epoch checkpoints in weights/")
            print(f"    Latest: {weight_files[-1]}")

    # Check for tensorboard
    tb_dir = os.path.join(output_dir, "tensorboard")
    if os.path.exists(tb_dir):
        print(f"\nTensorBoard logs: {tb_dir}")
        print(f"  View with: tensorboard --logdir {tb_dir}")

    print("\n" + "=" * 60)

    return {
        "params": params,
        "best_info": best_info,
        "final_results": results[-1] if results else None,
    }


def create_comparison_table(output_dir: str):
    """Create a formatted comparison table for the model."""
    params = load_model_params(output_dir)
    best_info = load_best_epochs(output_dir)
    results = load_csv_results(output_dir)

    # Count parameters from best weights if available
    best_weights = os.path.join(output_dir, "results", "best_loss_weights.pickle")
    total_params = 0
    if os.path.exists(best_weights):
        state_dict = torch.load(best_weights, map_location="cpu")
        total_params = count_parameters(state_dict)

    table = []
    table.append("\n" + "=" * 80)
    table.append("MUSIC TRANSFORMER - FINAL TRAINING RESULTS")
    table.append("=" * 80)
    table.append("")
    table.append("ARCHITECTURE:")
    table.append(f"  Model Type:           Music Transformer (decoder-only)")
    table.append(f"  Vocab Size:           390 (388 events + END + PAD)")
    table.append(f"  d_model:              {params.get('d_model', 'N/A')}")
    table.append(f"  n_layers:             {params.get('n_layers', 'N/A')}")
    table.append(f"  num_heads:            {params.get('num_heads', 'N/A')}")
    table.append(f"  dim_feedforward:      {params.get('dim_feedforward', 'N/A')}")
    table.append(f"  max_sequence:         {params.get('max_sequence', 'N/A')}")
    table.append(f"  Dropout:              {params.get('dropout', 'N/A')}")
    table.append(f"  RPR (Relative Pos):   {params.get('rpr', 'N/A')}")
    table.append("")
    table.append("TRAINING:")
    table.append(f"  Batch Size:           {params.get('batch_size', 'N/A')}")
    table.append(f"  Learning Rate:        {params.get('lr', 'Scheduled')}")
    table.append(f"  Label Smoothing:      {params.get('ce_smoothing', 'None')}")

    if results:
        table.append(f"  Total Epochs:         {len(results)}")

    table.append("")
    table.append("BEST RESULTS:")
    if best_info:
        if "Best_eval_loss" in best_info:
            table.append(f"  Best Eval Loss:       {best_info['Best_eval_loss']:.4f}")
        if "Best_eval_loss_epoch" in best_info:
            table.append(
                f"  Best Loss Epoch:      {int(best_info['Best_eval_loss_epoch'])}"
            )
        if "Best_eval_acc" in best_info:
            table.append(f"  Best Eval Accuracy:   {best_info['Best_eval_acc']:.4f}")
        if "Best_eval_acc_epoch" in best_info:
            table.append(
                f"  Best Acc Epoch:       {int(best_info['Best_eval_acc_epoch'])}"
            )

    table.append("")
    table.append("MODEL SIZE:")
    if total_params > 0:
        table.append(f"  Total Parameters:     {total_params:,}")
        table.append(f"  Model Size (MB):      {total_params * 4 / (1024**2):.2f}")

    table.append("")
    table.append("=" * 80)

    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description="Extract metrics from training output")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_models",
        help="Path to training output directory",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to specific weights file to analyze",
    )
    args = parser.parse_args()

    if args.weights:
        extract_metrics_from_weights(args.weights)
    else:
        if os.path.isdir(args.output_dir):
            create_metrics_summary(args.output_dir)

            # Create and print comparison table
            table = create_comparison_table(args.output_dir)
            print(table)

            # Save to file
            output_file = os.path.join(args.output_dir, "training_results.txt")
            with open(output_file, "w") as f:
                f.write(table)
            print(f"\nResults saved to: {output_file}")
        else:
            print(f"Error: Directory not found: {args.output_dir}")


if __name__ == "__main__":
    main()
