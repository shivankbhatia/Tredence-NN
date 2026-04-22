# Self-Pruning Neural Network

A neural network implementation that learns to prune itself during training using learnable gate parameters.

## Overview

This project implements a feed-forward neural network with a built-in self-pruning mechanism. Each weight in the network has an associated "gate" parameter that learns whether the weight should be active or pruned during training.

## Key Features

- **PrunableLinear Layer**: Custom linear layer with learnable gates for each weight
- **Sparsity Regularization**: L1 penalty on gate values encourages automatic pruning
- **Dynamic Architecture**: Network adapts its structure during training
- **Trade-off Analysis**: Experiments with different sparsity parameters (λ)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the training script:

```bash
python self_pruning_network.py
```

This will:
1. Train five models with different λ values (0.0002, 0.002, 0.02)
2. Generate gate distribution plots for each model
3. Generate detailed training history plots (Loss, Accuracy, Sparsity) for each model
4. Create a RESULTS.md file with detailed analysis
5. Print a summary table comparing accuracy vs. sparsity

## How It Works

### 1. Prunable Linear Layer

Each weight `w` has a gate `g` computed as:
```
g = sigmoid(gate_score)
pruned_weight = w * g
```

### 2. Loss Function

```
Total Loss = Classification Loss + λ * Sparsity Loss
Sparsity Loss = Σ(all gate values)
```

### 3. Training Process

- The optimizer updates both weights and gate_scores
- L1 penalty on gates encourages values toward 0 or 1
- Network learns which connections are essential

## Results

After training, you'll get:
- Test accuracy for each λ value
- Sparsity percentage (% of pruned weights)
- Gate distribution plots showing bimodal distribution
- Training history plots tracking Loss, Accuracy, and Sparsity iteratively over epochs
- Detailed analysis in RESULTS.md

## Expected Behavior

A successful implementation shows:
- Higher λ → More pruning, potentially lower accuracy
- Lower λ → Less pruning, higher accuracy
- Gate distributions with spike at 0 (pruned) and cluster at higher values (active)

## File Structure

```
.
├── self_pruning_network.py    # Main implementation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── RESULTS.md                  # Generated after training
├── gate_distribution_*.png     # Generated gate plots
└── training_history_*.png      # Generated history plots
```

## Implementation Details

### PrunableLinear Class
- Inherits from `nn.Module`
- Has `weight`, `bias`, and `gate_scores` parameters
- Forward pass applies sigmoid gates to weights

### SelfPruningNetwork Class
- Stack of PrunableLinear layers with default hidden dimensions [1024, 512, 256] and ReLU activations.
- Computes sparsity loss across all prunable layers
- Tracks sparsity statistics during training

### Training Loop
- Standard SGD with Adam optimizer
- Custom loss combining cross-entropy and L1 sparsity
- Learning rate scheduling using StepLR for better convergence
- Added visual `tqdm` progress tracking for training and evaluation loops

## Customization

### My Findings (Summary)

- Baseline feed-forward model (ReLU + dropout, no explicit pruning) reached a maximum accuracy of `61.15%`.
- Initial pruning attempts with weak lambda values gave poor sparsity, while strong lambda values caused model collapse.
- Stabilized training by using mid-range lambda values and separate learning rates for weights and gates.
- Added warmup before pruning and a ramp phase for smooth sparsity pressure.
- Froze BatchNorm during pruning to make pruning decisions more meaningful.

### Final Training Phase Design (Can be implemented later...)

| Phase | Epochs | λ_eff | BN status | What happens |
|---|---:|---|---|---|
| Warmup | 1-8 | 0 | Active | Learn features; BN collects running statistics |
| BN-adapt | 9-13 | 0 | Frozen | Weights re-adapt to frozen BN; no pruning pressure |
| Ramp | 14-28 | 0→λ | Frozen | Gradual pruning; gates start separating |
| Full | 29-80 | λ | Frozen | Full pruning pressure; bimodal gate distribution forms |

### Key Fixes vs v1

- `gate_scores` initialized to `+2.0` (gates start near `0.88`, not `0.5`) to give sparsity loss a clear direction.
- Added a 5-epoch BN-adapt buffer between BN freezing and lambda start to avoid accuracy collapse.
- Increased sparsity threshold to `0.1` (`sigmoid(x) < 0.01` needs `x < -4.6`, hard to reach in 80 epochs with `gate_lr=0.002`).
- Increased gate LR to `0.01` so gates can move from around `2.0` to below `-2.2` (for threshold `0.1`).
- Used tighter lambda range: `[5e-4, 2e-3, 6e-3]`; larger values were destructive with new gate initialization.
