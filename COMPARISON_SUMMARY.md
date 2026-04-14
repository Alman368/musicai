# Deep Learning Final Project: LSTM vs Transformer for Music Generation

## Model Comparison

| Metric              | LSTM (PerformanceRNN) | Transformer (RPR) | Winner      |
| ------------------- | --------------------- | ----------------- | ----------- |
| **Val Loss**        | 2.2376                | 2.0426            | Transformer |
| **Val Accuracy**    | 33.43%                | 40.67%            | Transformer |
| **Val Perplexity**  | 9.53                  | 7.71              | Transformer |
| **Parameters**      | 6.08M                 | ~25M              | LSTM        |
| **Training Time**   | 10 hours              | 5 hours           | Transformer |
| **Epochs**          | 150                   | 100               | -           |

## Key Findings

### Transformer Advantages
- **+7.24% higher accuracy** (40.67% vs 33.43%)
- **-1.82 lower perplexity** (7.71 vs 9.53) - better prediction confidence
- **2x faster training** (5h vs 10h) despite more parameters
- Better at capturing long-range musical dependencies via self-attention

### LSTM Advantages
- **4x fewer parameters** (6M vs 25M) - more efficient model
- Simpler architecture, easier to understand and debug
- Lower memory requirements during inference

## Architecture Summary

### LSTM (PerformanceRNN)
```
Embedding(388 -> 256) -> 3x LSTM(512) -> Linear(512 -> 388)
```
- Sequential processing with hidden state
- Limited context window (512 tokens)
- Gradient issues with long sequences

### Transformer (RPR)
```
Embedding(388 -> 512) + PositionalEncoding -> 6x TransformerEncoder(8 heads) -> Linear(512 -> 388)
```
- Parallel processing with self-attention
- Relative Position Representations (RPR) for musical structure
- Full context access (2048 tokens)

## Training Details

| Configuration    | LSTM          | Transformer      |
| ---------------- | ------------- | ---------------- |
| Dataset          | MAESTRO v3.0.0 | MAESTRO v3.0.0  |
| Sequence Length  | 512           | 2048             |
| Batch Size       | 64            | 2                |
| Learning Rate    | 0.002         | LR Scheduler     |
| Optimizer        | Adam          | Adam             |
| Dropout          | 0.3           | 0.1              |
| GPU              | RTX 3070      | RTX 3070         |

## Metrics Explanation

- **Loss (Cross-Entropy)**: Lower is better. Measures prediction error.
- **Accuracy**: Higher is better. Percentage of correct next-token predictions out of 388 classes.
- **Perplexity**: Lower is better. exp(loss) - represents the effective number of choices the model is uncertain between. A perplexity of 10 means the model narrows 388 options to ~10.

## Conclusion

The **Transformer with Relative Position Representations** outperforms the **LSTM-based PerformanceRNN** on all quality metrics while training faster. The self-attention mechanism's ability to capture long-range dependencies in music proves more effective than the LSTM's sequential hidden state approach.

However, the LSTM remains a viable choice for resource-constrained environments due to its 4x smaller parameter count and simpler architecture.

## Project Structure

```
deep-learning/
├── perfnet/                      # LSTM Implementation
│   ├── README.md
│   ├── final_training_metrics.png
│   └── runs/                     # Trained model
│
├── MusicTransformer-Pytorch/     # Transformer Implementation
│   ├── README.md
│   ├── final_training_metrics.png
│   └── saved_models/             # Trained model
│
└── COMPARISON_SUMMARY.md         # This file
```
