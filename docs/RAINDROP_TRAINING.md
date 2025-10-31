# RainDrop Training Guide

Complete guide for training and evaluating RainDrop baseline for irregular time series forecasting.

## Overview

**Model**: RainDrop (modified for forecasting)
**Task**: Forecast 36 temporal variables at irregular timestamps in 24-30hr window
**Input**: 0-24hr patient history
**Output**: Sequence predictions at observed timestamps in forecast window

## Quick Start

```bash
# 1. Train model (50 epochs, ~30-60 min on GPU)
python src/models/raindrop/train_raindrop_forecasting.py --epochs 50 --batch-size 32

# 2. Train with WandB logging
python src/models/raindrop/train_raindrop_forecasting.py --epochs 50 --use-wandb

# 3. Evaluate on test set
python src/models/raindrop/evaluate_raindrop.py --checkpoint results/raindrop/best_model.pt
```

## Task Definition

Based on team discussion, the forecasting task is:

- **Input window**: First 24 hours (timestamps 0 to 0.5 normalized)
- **Forecast window**: 6 hours ahead (timestamps 0.5 to 0.625, i.e., hours 24-30)
- **Variables**: 36 temporal variables (not the 5 static ones)
  - Static (not predicted): Age, Gender, Height, ICUType, Weight
  - Temporal (predicted): HR, Temp, SysABP, Glucose, pH, etc. (36 variables)
- **Prediction type**: Sequence prediction
  - For each patient, predict value at each observed (timestamp, variable) pair in forecast window
  - NOT a single point prediction per variable
  - Each variable can have multiple observations at different irregular timestamps
- **Evaluation**: Only on observed values (respecting mask)

## Data Format

### Our Preprocessed Data
```python
record_id, timestamps, values, mask = train_data[0]
# timestamps: (T,) normalized [0, 1] (48hr → [0, 1])
# values: (T, 41) z-score normalized (5 static + 36 temporal)
# mask: (T, 41) binary (1=observed, 0=missing)
```

### RainDrop Format (after conversion)
```python
src: (T, B, 72)     # History: [values, mask] for 36 temporal vars
times: (T, B)       # History timestamps
static: (B, 9)      # Static features (padded to 9)
lengths: (B,)       # Sequence lengths per patient
targets: List[(T_i, 36)]  # Variable-length targets per patient
target_masks: List[(T_i, 36)]  # Masks per patient
```

## Training Options

### Basic Training
```bash
python src/models/raindrop/train_raindrop_forecasting.py
```

Default settings:
- 50 epochs
- Batch size: 32
- Learning rate: 0.0001
- Model dimension: 64
- Saves best model based on validation loss

### Custom Configuration
```bash
python src/models/raindrop/train_raindrop_forecasting.py \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0005 \
    --d-model 128 \
    --nlayers 3 \
    --dropout 0.4 \
    --output-dir results/raindrop_v2
```

### With WandB Logging
```bash
# First login to WandB
wandb login

# Then train with logging
python src/models/raindrop/train_raindrop_forecasting.py \
    --epochs 50 \
    --use-wandb \
    --wandb-project graphml-imts \
    --wandb-name raindrop_baseline_v1
```

WandB will log:
- Train/val loss curves
- All metrics (RMSE, MAE, R², RMSE/MAE ratio)
- Visualizations (loss curves, predictions vs actual)

## Device Selection

The script auto-detects the best available device:

### Auto (Recommended)
```bash
python src/models/raindrop/train_raindrop_forecasting.py --device auto
```
Priority: CUDA > MPS > CPU

### Force Specific Device
```bash
# For Apple Silicon Macs
python src/models/raindrop/train_raindrop_forecasting.py --device mps

# For CUDA GPUs
python src/models/raindrop/train_raindrop_forecasting.py --device cuda

# For CPU (slower)
python src/models/raindrop/train_raindrop_forecasting.py --device cpu
```

### MPS vs CUDA

**MPS (Apple Silicon)**:
- Works on M1/M2/M3 Macs
- Training should work but ~2-3x slower than CUDA
- Good for testing/debugging
- Expected time: ~1-2 hours for 50 epochs

**CUDA (NVIDIA GPU)**:
- Fastest option
- Recommended for final training
- Expected time: ~20-40 minutes for 50 epochs
- If you need CUDA: SSH into a server with GPU and run there

**Recommendation**: Test on MPS first, then train on CUDA for best results.

## Outputs

Training creates the following in `results/raindrop/`:

```
results/raindrop/
├── best_model.pt              # Best model checkpoint (lowest val loss)
├── final_model.pt             # Final model after all epochs
├── training_history.json      # All metrics per epoch
├── loss_curves.png            # Train/val loss over time
├── metrics_over_time.png      # RMSE/MAE over epochs
└── predictions_vs_actual.png  # Scatter plot of predictions
```

### Checkpoint Contents
```python
checkpoint = torch.load('results/raindrop/best_model.pt')
# Keys:
#   - model_state_dict: Model weights
#   - optimizer_state_dict: Optimizer state
#   - epoch: Epoch number
#   - train_metrics: Training metrics
#   - val_metrics: Validation metrics
#   - args: Training arguments
```

## Evaluation

After training, evaluate on test set:

```bash
python src/models/raindrop/evaluate_raindrop.py \
    --checkpoint results/raindrop/best_model.pt
```

This generates:

```
results/raindrop/test_results/
├── test_results.json           # Detailed metrics (JSON)
├── test_results.txt            # Human-readable report
├── predictions_vs_actual.png   # Scatter plot
├── error_analysis.png          # Residuals + error distribution
├── rmse_distribution.png       # Per-patient RMSE histogram
├── mae_distribution.png        # Per-patient MAE histogram
└── r2_distribution.png         # Per-patient R² histogram
```

## Metrics

All metrics computed on **normalized** (z-scored) values, **only** on observed data points (respecting mask).

### Overall Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **R²**: Coefficient of determination
- **RMSE/MAE Ratio**: Outlier sensitivity indicator
  - Ratio >> 1: Model struggles with outliers
  - Ratio ≈ 1: Errors uniformly distributed (robust)

### Per-Patient Statistics
For each metric, we compute:
- Mean, median, std
- Min, max
- Q25, Q75 (quartiles)

This shows how performance varies across patients.

## Comparing with Team Results

According to team discussion, Luca's results (50 epochs) were:
1. Learnable graphs (best)
2. KAFNet (second)
3. SeFT (third)

Your RainDrop baseline should be comparable. Expected performance on normalized values:
- **Val RMSE**: ~0.3-0.5
- **Val MAE**: ~0.2-0.4
- **Val R²**: ~0.3-0.6

If results are very different, check:
1. Are you using the same train/val/test splits? (should be, from preprocessing)
2. Are predictions and targets both normalized?
3. Is the mask properly applied during loss computation?

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python src/models/raindrop/train_raindrop_forecasting.py --batch-size 16
```

### Slow Training on MPS
```bash
# Expected on Apple Silicon. Options:
# 1. Use smaller model
python src/models/raindrop/train_raindrop_forecasting.py --d-model 32 --nlayers 1

# 2. Fewer epochs for testing
python src/models/raindrop/train_raindrop_forecasting.py --epochs 10

# 3. Switch to CUDA server
ssh gpu_server
cd GraphML-imts
python src/models/raindrop/train_raindrop_forecasting.py --device cuda
```

### NaN Loss
This can happen if:
1. Learning rate too high → Try `--lr 0.00005`
2. Gradient explosion → Already using gradient clipping (max_norm=1.0)
3. Bad initialization → Should be fine with default

### Low R² Score
R² can be negative or low on this task because:
- Highly irregular, sparse data
- Predicting normalized values (not raw)
- Some variables are very noisy (clinical data)

Focus on RMSE and MAE as primary metrics.

## Advanced Usage

### Resume Training
```python
# Load checkpoint and continue training
checkpoint = torch.load('results/raindrop/final_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
# Continue training...
```

### Hyperparameter Search
```bash
# Try different configurations
for lr in 0.0001 0.0005 0.001; do
    for d_model in 32 64 128; do
        python src/models/raindrop/train_raindrop_forecasting.py \
            --lr $lr --d-model $d_model \
            --output-dir results/raindrop_lr${lr}_d${d_model} \
            --use-wandb --wandb-name "raindrop_lr${lr}_d${d_model}"
    done
done
```

### Export Predictions
```python
# In evaluate_raindrop.py, add:
torch.save({
    'predictions': all_preds,
    'targets': all_targets,
    'patient_ids': [data[i][0] for i in range(len(data))]
}, 'predictions.pt')
```

## Graph Structure

RainDrop uses a **sensor graph** (not observation graph like Hi-Patch):

- **Nodes**: 36 variables (sensors)
- **Edges**: Initially fully connected (`global_structure = torch.ones(36, 36)`)
- **Message passing**: Graph attention between variables at each timestamp
- **Learning**: Edge weights learned via attention mechanism

This is different from your preprocessing graphs where nodes are observations!

## Next Steps

1. **Train baseline**: Get initial results with default settings
2. **Compare with team**: See how it compares to Luca's baselines
3. **Tune hyperparameters**: Try different learning rates, model sizes
4. **Visualizations for report**: Use generated plots
5. **Write results section**: Document performance

## Questions?

- RainDrop paper: https://arxiv.org/abs/2110.05357
- Original code: `src/models/raindrop/Raindrop/`
- Our modifications: `src/models/raindrop/Raindrop/code/models_rd.py` lines 92-107, 204-208

