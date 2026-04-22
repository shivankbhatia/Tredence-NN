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
1. Train five models with different λ values (0.0001, 0.001, 0.005, 0.01, 0.05)
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

You can modify:
- Network architecture: Change `hidden_sizes` in `SelfPruningNetwork`
- Lambda values: Adjust `lambda_values` in `main()`
- Training epochs: Change `num_epochs` parameter
- Pruning threshold: Modify `threshold` in `get_sparsity_stats()`

## Notes

- Training on GPU is recommended (automatically detected)
- CIFAR-10 dataset downloads automatically on first run
- Each training run takes approximately 5-10 minutes on GPU
- Results are reproducible with fixed random seed
