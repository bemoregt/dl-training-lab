# FrFT Loss for ResNet-18 Image Classification

An experiment applying a **Fractional Fourier Transform (FrFT)-based loss function** to fine-tune ResNet-18, comparing it against standard Cross-Entropy loss on the Hymenoptera (ants vs. bees) binary classification task.

![이미지 예시](https://github.com/bemoregt/Training_Resnet_Using_FrFTLossFunction/blob/main/frft_vs_standard_accuracy.png)

## Overview

This project introduces `FrFTLoss`, a novel composite loss function that augments standard Cross-Entropy loss with additional supervision signal derived from the Fractional Fourier Transform domain. The hypothesis is that penalizing classification errors across multiple fractional frequency domains can improve model generalization.

### Loss Formula

```
total_loss = CE_loss(logits, targets) + λ · mean(CE_loss(|FrFT_α(logits)|, targets))
```

where the FrFT is applied at multiple rotation orders `α ∈ {0.5, 1.0, 1.5}` and `λ = 0.5`.

## Dataset

**Hymenoptera Dataset** — a small binary classification dataset from the PyTorch transfer learning tutorial.

| Split | Samples |
|-------|---------|
| Train | ~244    |
| Val   | ~153    |
| Classes | `ants`, `bees` |

## Files

| File | Description |
|------|-------------|
| `frft_resnet18.py` | Version 1: auto-downloads dataset, 10-epoch FrFT training only |
| `frft_resnet2.py` | Version 2: 10-epoch head-to-head comparison (FrFT vs. Standard), full loss + accuracy plots |
| `frft_resnet3.py` | Version 3: 30-epoch head-to-head comparison, accuracy-only plots |
| `frft_loss_resnet18_results.png` | Training curves for FrFT Loss (v1) |
| `frft_vs_standard_comparison.png` | 4-panel comparison chart (loss + accuracy, v2) |
| `frft_vs_standard_accuracy.png` | Accuracy-only comparison chart (v3) |

## Model & Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | ResNet-18 (ImageNet pretrained) |
| Output head | Linear(512 → 2) |
| Optimizer | SGD, lr=0.001, momentum=0.9 |
| LR scheduler | StepLR (step=7, γ=0.1) |
| Batch size | 16 |
| FrFT α orders | 0.5, 1.0, 1.5 |
| λ (FrFT weight) | 0.5 |
| Device | Apple MPS (falls back to CPU) |

## How FrFT Loss Works

1. The model produces logits of shape `(batch_size, num_classes)`.
2. For each fractional order `α`, a 1-D discrete FrFT kernel is applied to each sample's logit vector.
3. The magnitude of the complex FrFT output is taken as transformed logits.
4. Cross-Entropy is computed in the FrFT domain and averaged across all `α` orders.
5. The final loss is the sum of the original CE loss and the scaled average FrFT-domain loss.

## Requirements

```
torch
torchvision
numpy
scipy
matplotlib
```

Install with:

```bash
pip install torch torchvision numpy scipy matplotlib
```

## Usage

```bash
# Version 1 — downloads dataset automatically
python frft_resnet18.py

# Version 2 — 10-epoch comparison (requires dataset at configured path)
python frft_resnet2.py

# Version 3 — 30-epoch accuracy comparison
python frft_resnet3.py
```

> **Note:** `frft_resnet2.py` and `frft_resnet3.py` expect the dataset at
> `/Users/m1_4k/그림/hymenoptera_data`. Update `data_dir` if running on a different machine.

## Results

Each script saves comparison plots to the project directory. The charts visualize:

- Training and validation loss curves over epochs
- Training and validation accuracy curves over epochs
- Final performance bar chart (FrFT Loss vs. Standard Cross-Entropy)
