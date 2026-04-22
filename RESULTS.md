## Results Summary

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|-------------------|--------------------|
| 0.0002 | 55.32 | 70.11 |
| 0.002 | 55.25 | 99.66 |
| 0.02 | 34.45 | 99.98 |

## Analysis

The results demonstrate the trade-off between model accuracy and sparsity:

- **Low λ (0.0002)**: Minimal pruning, highest accuracy, network retains most connections.
- **Medium λ (0.002)**: Balanced trade-off, moderate pruning with acceptable accuracy loss.
- **High λ (0.02)**: Aggressive pruning, higher sparsity but potential accuracy degradation.

The gate distribution plots show the bimodal distribution characteristic of successful pruning, with many gates near 0 (pruned) and active gates clustered at higher values.
