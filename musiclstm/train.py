#!/usr/bin/env python3
"""
Training script for PerformanceRNN.

EXACT Magenta PerformanceRNN replication with embedding layer:
- Input: event_ids [batch, seq_len] integers 0-387
- Embedding(388, 256)
- 3-layer LSTM with 512 hidden units, dropout=0.3
- Adam optimizer with lr=0.002
- Gradient clipping at norm 3.0

Usage:
    python train.py
    python train.py --config custom_config.yaml

Recommended: Use Lightning CLI instead
    python cli.py fit --config config.yaml
"""

import argparse

import yaml
from performance_net import (
    MIDIDataModule,
    PerformanceNet,
    PerformanceNetTrainer,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train PerformanceRNN")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("=" * 60)
    print("PerformanceRNN Training - Magenta Replication")
    print("=" * 60)
    print(f"Config: {args.config}")
    print()

    # Create data module
    data_module = MIDIDataModule(
        data_dir=config["data"]["data_dir"],
        sequence_length=config["data"]["sequence_length"],
        batch_size=config["data"]["batch_size"],
        stride=config["data"].get("stride", 256),
        file_ext=config["data"]["file_ext"],
        num_workers=config["data"]["num_workers"],
        val_split=config["data"]["val_split"],
    )

    print(f"Vocabulary size: {config['model']['vocab_size']}")
    print(
        f"Architecture: {config['model']['num_layers']}x LSTM-{config['model']['hidden_size']} "
        f"with Embedding({config['model']['vocab_size']}→{config['model']['embedding_size']})"
    )
    print(f"Dropout: {config['model']['dropout']}")
    print(f"Learning rate: {config['model']['learning_rate']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Sequence length: {config['data']['sequence_length']}")
    print()

    # Create model
    model = PerformanceNet(
        vocab_size=config["model"]["vocab_size"],
        embedding_size=config["model"]["embedding_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        learning_rate=config["model"]["learning_rate"],
        clip_norm=config["model"]["clip_norm"],
    )

    # Create trainer
    trainer = PerformanceNetTrainer(
        max_epochs=config["trainer"]["max_epochs"],
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
        save_every_n_minutes=config["trainer"]["save_every_n_minutes"],
        run_dir=config["trainer"]["run_dir"],
        enable_early_stopping=config["trainer"].get("enable_early_stopping", False),
        early_stopping_patience=config["trainer"].get("early_stopping_patience", 10),
    )

    print(f"Run directory: {trainer.run_path}")
    print("=" * 60)
    print()

    # Train
    trainer.fit(model, data_module)

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {trainer.run_path}")
    print(f"Metrics plot: {trainer.run_path}/training_metrics.png")
    print(f"TensorBoard: tensorboard --logdir {trainer.run_path}/tensorboard")
    print("=" * 60)


if __name__ == "__main__":
    main()
