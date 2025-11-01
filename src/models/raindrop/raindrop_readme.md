# RainDrop for Irregular Time Series Forecasting

Official RainDrop implementation ([Zhang et al., 2021](https://arxiv.org/abs/2110.05357)) adapted for multivariate time series forecasting.

## Overview

RainDrop is a graph neural network that learns latent sensor dependency graphs through neural message passing. Unlike traditional sequential models, RainDrop:

- Constructs a **sensor graph** where nodes represent variables (sensors)
- Learns directed edges representing dependencies between sensors
- Handles irregular sampling by passing messages when observations arrive
- Aggregates information hierarchically: observations → sensors → sample

**Original Task**: Classification (mortality prediction, activity recognition)  
**Our Task**: Forecasting (predict 36 vitals 6 hours ahead)

## How RainDrop Works

### 1. Sensor Graph Construction

For each patient sample $\mathcal{S}_i$, RainDrop builds a sensor dependency graph $\mathcal{G}_i$:
- **Nodes**: 36 temporal variables (HR, Temp, Glucose, pH, etc.)
- **Edges**: $e_{i,uv} \in [0,1]$ - learned weights from sensor $u$ to sensor $v$
- **Initialization**: Fully connected (all edges start at 1.0)
- **Learning**: Edge weights updated via attention during training

### 2. Message Passing (When Observation Arrives)

When observation $x_{i,u}^t$ is recorded for sensor $u$ at time $t$:

**Step 1**: Embed the observation
$$\bm{h}_{i,u}^t = \sigma(x_{i,u}^t \bm{R}_u)$$
where $\bm{R}_u$ is a sensor-specific weight vector

**Step 2**: Propagate to neighboring sensors
$$\bm{h}_{i,v}^t = \sigma(\bm{h}_{i,u}^t \bm{w}_u \bm{w}_v^T \alpha_{i,uv}^t e_{i,uv})$$
where:
- $\alpha_{i,uv}^t$ = time-varying attention weight (depends on timestamp)
- $e_{i,uv}$ = learned edge weight (shared across time)
- $\bm{w}_u, \bm{w}_v$ = sensor-specific transformations

This allows **unobserved sensors** to get embeddings via message passing!

### 3. Temporal Aggregation

Aggregate observation embeddings for each sensor across all timestamps using temporal self-attention:
$$\bm{z}_{i,v} = \sum_{t \in \mathcal{T}_{i,v}} \beta^t_{i,v} [\bm{h}_{i,v}^t || \bm{p}^t_i]$$
where $\beta^t_{i,v}$ are learned temporal attention weights

### 4. Sample Embedding

Aggregate all sensor embeddings into a single sample embedding:
$$\bm{z}_i = g(\bm{z}_{i,1}, \bm{z}_{i,2}, \ldots, \bm{z}_{i,36})$$

### 5. Prediction

**Original (Classification)**: $\hat{y}_i = \text{FC}(\bm{z}_i) \rightarrow 2$ classes

**Ours (Forecasting)**: $\hat{Y}_i = \text{FC}(\bm{z}_i) \rightarrow (36 \text{ vars}, 6 \text{ timesteps})$

## Our Modifications

We adapted RainDrop from classification to forecasting with **minimal code changes**:

### Changes Made

**File**: `Raindrop/code/models_rd.py`

**Lines Changed**: ~20 lines in two classes

**Modifications**:
1. Added `forecasting_mode` parameter to `__init__`
2. Modified output layer:
   ```python
   # Original
   self.output = nn.Linear(d_model, n_classes)  # 2 classes
   
   # Our forecasting version
   if forecasting_mode:
       self.output = nn.Linear(d_model, d_inp * n_forecast_steps)  # 36*6=216
   ```
3. Reshaped output:
   ```python
   # Reshape to (batch, 36, 6)
   output = output.view(batch_size, d_inp, n_forecast_steps)
   ```
4. Made device-agnostic (replaced `.cuda()` with `.to(device)`)

**Everything else unchanged**: Message passing, sensor graphs, temporal attention, all original

### Additional Fixes for Production

Beyond forecasting adaptation, we fixed several issues:
1. **Device compatibility**: Hardcoded `.cuda()` → device-agnostic `.to(device)`
2. **Sequence length**: Hardcoded 215 → dynamic `x.shape[0]`  
3. **Dimension requirements**: `d_model` must be multiple of `d_inp=36`
4. **Target device placement**: Ensured all tensors on same device

## Installation & Setup

### Prerequisites

```bash
# Create environment
conda env create -f ../../../environment.yml
conda activate graphml-imts

# Install PyTorch (auto-detects platform)
python ../../../install_pytorch.py
```

### Data Preparation

RainDrop uses our preprocessed data:
```bash
python ../../../scripts/preprocess_data.py
```

This creates:
- `data/physionet/processed/train.pt` (7,200 patients)
- `data/physionet/processed/val.pt` (2,400 patients)  
- `data/physionet/processed/test.pt` (2,400 patients)

Format: `(record_id, timestamps, values, mask)` per patient
- `timestamps`: (T,) normalized [0, 1]
- `values`: (T, 41) z-score normalized
- `mask`: (T, 41) binary (1=observed, 0=missing)

## Training

### Quick Test

```bash
# Verify setup works
python test_raindrop_training.py
```

This runs a minimal training loop to catch any issues.

### Basic Training

```bash
# Train 50 epochs with default settings
python train_raindrop_forecasting.py --epochs 50
```

**Default Configuration**:
- Batch size: 64
- Learning rate: 1e-4
- Model: d_model=72, 4 heads, 2 layers
- Dropout: 0.3
- Device: Auto (CUDA > MPS > CPU)

### GPU-Optimized Training

```bash
# Find optimal batch size for your GPU
python find_optimal_batch_size.py

# Train with optimized settings
python train_raindrop_forecasting.py \
    --epochs 50 \
    --batch-size 512 \
    --use-amp \
    --use-wandb
```

**GPU Batch Size Recommendations**:
| GPU | Batch Size | Memory | Time (50 epochs) |
|-----|-----------|---------|-----------------|
| A100 80GB | 2048-7200 | 2-8 GB | ~10-20 min |
| RTX 3090 24GB | 512-1024 | 2-4 GB | ~30-45 min |
| Apple M1 Max | 128-256 | 1-2 GB | ~2-3 hrs |
| CPU | 32-64 | <1 GB | ~10 hrs |

### All Training Arguments

```bash
python train_raindrop_forecasting.py \
    --data-dir data/physionet/processed \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.0001 \
    --weight-decay 0.0 \
    --distance-weight 0.02 \       # RainDrop graph regularization
    --d-model 72 \                 # Must be multiple of 36
    --nhead 4 \
    --nlayers 2 \
    --dropout 0.3 \
    --device auto \                # auto|cuda|mps|cpu
    --use-amp \                    # Mixed precision (GPU only)
    --use-wandb \
    --wandb-project graphml-imts \
    --output-dir results/raindrop \
    --save-best
```

## Evaluation

### Test Set Evaluation

```bash
python evaluate_raindrop.py --checkpoint results/raindrop/best_model.pt
```

Generates in `results/raindrop/test_results/`:
- `test_results.json` - All metrics (JSON)
- `test_results.txt` - Human-readable report
- `predictions_vs_actual.png` - Scatter plot
- `error_analysis.png` - Residuals and distributions
- `*_distribution.png` - Per-patient metric histograms

### Custom Evaluation

```bash
# Evaluate with different batch size
python evaluate_raindrop.py \
    --checkpoint results/raindrop/best_model.pt \
    --batch-size 128 \
    --output-dir results/raindrop/custom_eval
```

## Results

### Overall Performance (50 Epochs)

**Validation Set** (2,400 patients):
- RMSE: **0.561**
- MAE: **0.365**
- R²: **0.519**
- RMSE/MAE: 1.538
- Loss (MSE): 0.291

**Test Set** (2,368 patients):
- RMSE: **0.596** 
- MAE: **0.368**
- R²: **0.494**
- RMSE/MAE: 1.621
- Loss (MSE): 0.337

**Training Set** (7,200 patients):
- RMSE: 0.638
- MAE: 0.371
- R²: 0.479
- RMSE/MAE: 1.719

All metrics computed on z-score normalized values, only on observed data points (respecting mask).

### Training Dynamics

**Convergence**:
- Train loss: 0.719 → 0.371 (48% reduction)
- Val loss: 0.538 → 0.291 (46% reduction)
- Monotonic improvement, no plateau
- Stable RMSE/MAE ratio (1.5-1.7)

**Generalization**:
- Val-Test gap: 0.561 → 0.596 RMSE (6% increase)
- Indicates good generalization without severe overfitting

**Training Time**:
- A100 80GB: ~8.5 min (batch=64)
- RTX 3090 24GB: ~30 min (batch=64)
- Apple M1 Max: ~2-3 hrs

### Per-Patient Analysis (Test Set)

**RMSE Distribution**:
- Mean: 0.484 ± 0.303
- Median: 0.435
- Range: 0.014 to 6.053
- Q25: 0.336, Q75: 0.554

**MAE Distribution**:
- Mean: 0.356 ± 0.161
- Median: 0.328
- Range: 0.014 to 2.053

**R² Distribution**:
- Median: 0.517 (model explains ~50% variance)
- Q25: 0.256, Q75: 0.701
- 25-70% variance explained for most patients

**Interpretation**:
- High variance reflects clinical heterogeneity
- Stable patients (predictable vitals) → low RMSE
- Unstable/sparse patients → high RMSE
- Long tail driven by extreme cases

## Model Architecture Details

### Hyperparameters

**Used in our experiments**:
```python
d_inp = 36           # Number of temporal variables
d_model = 72         # Model dimension (must be multiple of d_inp)
nhead = 4            # Attention heads
nlayers = 2          # Number of graph layers
dropout = 0.3
max_len = 216        # Max sequence length in data
d_static = 9         # Static features (Age, Gender, etc.)
n_forecast_steps = 6 # Predict 6 future irregular timesteps
```

**Model Size**: 250,166 trainable parameters

### Graph Structure

- **Initialization**: Fully connected (36×36 = 1,296 edges)
- **Pruning**: Bottom K% edges pruned per layer (from paper)
- **Learning**: Edge weights updated via attention aggregation
- **Regularization**: Distance loss encourages similar graphs for similar patients

### Loss Function

```python
loss = MSE_loss + distance_weight * distance_regularization
```

where:
- `MSE_loss`: Mean squared error on predictions vs targets
- `distance_regularization`: Euclidean distance between attention patterns (encourages consistency)
- `distance_weight`: 0.02 (from RainDrop paper)

## Data Format

### Input Format (Our Preprocessing)

```python
record_id, timestamps, values, mask = train_data[i]
# timestamps: (T,) float, normalized [0, 1]
# values: (T, 41) float, z-score normalized
# mask: (T, 41) bool, 1=observed, 0=missing
```

### RainDrop Format (After Conversion)

```python
src: (T_max, B, 72)      # History: [values, mask] for 36 temporal vars
times: (T_max, B)        # History timestamps  
static: (B, 9)           # Static features (Age, Gender, Height, ...)
lengths: (B,)            # Actual sequence lengths
targets: List[(T_i, 36)] # Forecast targets (variable length per patient)
target_masks: List[(T_i, 36)]  # Observation masks
target_times: List[(T_i,)]     # Forecast timestamps
```

**Conversion** handled automatically by `convert_to_raindrop_format()` in `train_raindrop_forecasting.py`.

### Forecasting Task

- **Input window**: 0-24 hours (normalized: 0.0 to 0.5)
- **Forecast window**: 24-30 hours (normalized: 0.5 to 0.625)
- **Prediction**: Value at each observed (timestamp, variable) pair in forecast window
- **Evaluation**: Only on observed values (respecting mask)

## Code Structure

```
src/models/raindrop/
├── train_raindrop_forecasting.py   # Training script
├── evaluate_raindrop.py            # Evaluation script
├── environment.yml                 # Conda environment (legacy)
└── Raindrop/                       # Official RainDrop repository
    ├── code/
    │   ├── models_rd.py           # Main model (OUR MODIFICATIONS HERE)
    │   ├── transformer_conv.py    # Graph conv layer
    │   └── baselines/             # GRU-D, SeFT, mTAND, etc.
    └── paper/                      # Original paper LaTeX

Modified: ~20 lines in models_rd.py
Unchanged: Everything else from official repo
```

## Usage Examples

### Basic Training

```python
from models_rd import Raindrop
import torch

# Create model
global_structure = torch.ones(36, 36)  # Initial fully-connected
model = Raindrop(
    d_inp=36,
    d_model=72,
    nhead=4,
    nhid=144,
    nlayers=2,
    dropout=0.3,
    max_len=216,
    d_static=9,
    n_classes=2,  # Ignored in forecasting mode
    forecasting_mode=True,
    n_forecast_steps=6,
    global_structure=global_structure
)

# Forward pass
predictions, distance, _ = model(src, static, times, lengths)
# predictions: (batch, 36, 6)
# distance: scalar (graph regularization term)

# Loss
criterion = nn.MSELoss()
loss = criterion(predictions, targets) + 0.02 * distance
```

### Loading from Checkpoint

```python
import torch
from models_rd import Raindrop

# Load checkpoint
checkpoint = torch.load('results/raindrop/best_model.pt')

# Recreate model with same config
model = Raindrop(**checkpoint['args'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    predictions, _, _ = model(src, static, times, lengths)
```

## Performance Analysis

### Learning Curves

Training shows steady, monotonic improvement:
- No overfitting (val loss tracks train loss)
- No plateau (continued improving at epoch 50)
- Stable RMSE/MAE ratio (no extreme outlier sensitivity)

See: `results/raindrop/loss_curves.png`, `results/raindrop/metrics_over_time.png`

### Error Patterns

**By Variable Coverage**:
- High coverage (HCT 99%, BUN 98%) → better accuracy
- Low coverage (TroponinI 7%, Cholesterol 10%) → higher errors
- Sensor graph needs temporal context for effective message passing

**By Patient**:
- Stable vitals → low error (RMSE ~0.3-0.4)
- Rapidly changing → high error (RMSE >1.0)
- Very sparse observations → unpredictable (RMSE >2.0)

**Extreme Values**:
- Slight underprediction of extremes (typical MSE behavior)
- Model conservative with outliers (HR=300, Glucose=800)
- RMSE/MAE ratio 1.5-1.6 indicates balanced errors

See: `results/raindrop/test_results/error_analysis.png`

## Computational Efficiency

### Training Speed

**50 Epochs Benchmark**:
| Platform | Batch Size | Time | GPU Util |
|----------|-----------|------|----------|
| A100 80GB | 64 | 8.5 min | 45% |
| A100 80GB | 512 | 12 min | 75% |
| A100 80GB | 2048 | 15 min | 85% |
| RTX 3090 24GB | 64 | 30 min | 80% |
| M1 Max 32GB | 128 | 2.5 hrs | N/A |

### Memory Efficiency

**Model is tiny compared to modern standards**:
- Parameters: 250K (~1 MB)
- Can fit entire dataset (7,200 patients) in one batch on A100
- Batch size 7200 uses only 7.3 GB / 80 GB (8.5%)

### Optimization Tips

1. **Increase batch size**: Default 64 is conservative
   ```bash
   python find_optimal_batch_size.py  # Find your max
   ```

2. **Enable mixed precision**: ~30% speedup
   ```bash
   python train_raindrop_forecasting.py --use-amp
   ```

3. **Monitor GPU**: Watch utilization
   ```bash
   nvidia-smi -l 1  # Update every second
   ```

## Troubleshooting

### Common Issues

**1. Dimension mismatch error**
```
AssertionError: was expecting embedding dimension of 100, but got 72
```
**Solution**: Ensure `d_model` is multiple of `d_inp=36`. Use 72, 108, 144, etc.

**2. Out of memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or enable AMP
```bash
python train_raindrop_forecasting.py --batch-size 32 --use-amp
```

**3. libstdc++ error (Linux)**
```
ImportError: libstdc++.so.6: version `CXXABI_1.3.15' not found
```
**Solution**: Already fixed! Conda activation sets `LD_LIBRARY_PATH` automatically

**4. Device placement error**
```
RuntimeError: Expected all tensors to be on the same device
```
**Solution**: Already fixed in our version. Update to latest code.

### Platform-Specific Notes

**Linux + NVIDIA GPU**:
- Best performance
- Use batch size 512+ for good GPU utilization
- Enable AMP for free 30% speedup

**macOS + Apple Silicon**:
- MPS support works
- ~2-3x slower than CUDA
- Use batch size 128-256
- Some operations may fall back to CPU

**CPU-only**:
- Works but slow (~100x slower than A100)
- Use small batches (32-64)
- Good for development/testing

## Understanding the Outputs

### Training History JSON

```json
{
  "train_losses": [0.719, 0.616, ..., 0.371],
  "val_losses": [0.538, 0.481, ..., 0.291],
  "train_metrics": [{
    "train_rmse": 0.869,
    "train_mae": 0.569,
    "train_r2": 0.035,
    ...
  }, ...],
  "val_metrics": [...],
  "args": {...}  // All hyperparameters
}
```

### Test Results JSON

```json
{
  "overall_metrics": {
    "test_rmse": 0.596,
    "test_mae": 0.368,
    "test_r2": 0.494,
    ...
  },
  "per_patient_statistics": {
    "rmse": {"mean": 0.484, "median": 0.435, ...},
    "mae": {...},
    "r2": {...}
  },
  "n_patients_evaluated": 2368
}
```

## Comparison with Other Models

RainDrop serves as our graph-based baseline. Key differences from other approaches:

**vs GRU-D**: 
- RainDrop: Learns sensor graphs, parallel processing
- GRU-D: Sequential RNN with time decay

**vs Latent ODE**:
- RainDrop: Discrete message passing on learned graphs
- ODE: Continuous-time dynamics with neural ODEs

**vs SeFT**:
- RainDrop: Structured (graph) with learned edges
- SeFT: Permutation-invariant set functions

**vs Hi-Patch**:
- RainDrop: Sensor graph (variables as nodes)
- Hi-Patch: Observation graph (measurements as nodes)

## Extending RainDrop

### Modify for Different Tasks

**Classification** (original):
```python
model = Raindrop(..., forecasting_mode=False, n_classes=2)
```

**Regression** (single value per variable):
```python
model = Raindrop(..., forecasting_mode=True, n_forecast_steps=1)
```

**Longer forecasts**:
```python
model = Raindrop(..., n_forecast_steps=12)  # 12 timesteps ahead
```

### Hyperparameter Tuning

**Suggested ranges**:
- `d_model`: {72, 108, 144} - must be multiple of 36
- `nhead`: {2, 4, 8}
- `nlayers`: {1, 2, 3}
- `dropout`: {0.1, 0.3, 0.5}
- `lr`: {5e-5, 1e-4, 5e-4}

**Grid search example**:
```bash
for lr in 0.00005 0.0001 0.0005; do
    for d_model in 72 108 144; do
        python train_raindrop_forecasting.py \
            --lr $lr --d-model $d_model \
            --output-dir results/raindrop_lr${lr}_d${d_model} \
            --use-wandb --wandb-name "lr${lr}_d${d_model}"
    done
done
```

## Citations

If you use this implementation, cite:

**RainDrop (original)**:
```bibtex
@inproceedings{zhang2021graph,
  title={Graph-Guided Network for Irregularly Sampled Multivariate Time Series},
  author={Zhang, Xiang and Zeman, Marko and Tsiligkaridis, Theodoros and Zitnik, Marinka},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

**Our Forecasting Adaptation**:
```bibtex
@misc{graphml-imts-2024,
  title={Graph Machine Learning for Irregular Multivariate Time Series Forecasting},
  author={Roudbari, Asal and Gianantonio, Luca and Sanganeria, Saahil and Reza, Nawal},
  year={2024}
}
```

## Additional Resources

- **Original Paper**: https://arxiv.org/abs/2110.05357
- **Original Code**: https://github.com/mims-harvard/Raindrop
- **Our Repository**: See main README.md
- **Results**: `results/raindrop/`
- **Documentation**: This file + main README.md

## Support

For questions or issues:
1. Check this README
2. Run `python test_raindrop_training.py` to verify setup
3. Check `results/raindrop/training_history.json` for training details
4. See main project README for platform-specific setup

---

**Last Updated**: November 1, 2025  
**Status**: ✅ Fully functional, tested on CUDA/MPS/CPU  
**Performance**: Competitive baseline for irregular time series forecasting

