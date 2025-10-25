# Data Preprocessing

How I preprocessed the PhysioNet data for our project.

## What It Does

Takes raw PhysioNet text files → clean PyTorch tensors + observation graphs

```bash
python scripts/preprocess_data.py  # ~20 min
```

## Output

```
data/physionet/processed/
├── train.pt, val.pt, test.pt              # 60/20/20 split
├── train_graphs.pt, val_graphs.pt, test_graphs.pt  # Pre-built graphs
├── all_data.pt, all_graphs.pt             # Full dataset
├── split_indices.pt                        # Which patient in which split
└── statistics.pt                           # Mean/std for normalization
```

## Pipeline

1. **Download** - Gets sets A, B, C from PhysioNet (~1GB, 12K patients)

2. **Parse** - Reads text files, converts `HH:MM` to hours, creates masks

3. **Validate** - Only removes NaN/inf. **Keeps outliers** (HR=300, Glucose=800 are real ICU patients, not errors)

4. **Split** - Random 60/20/20, seed=42 for reproducibility

5. **Normalize** - Z-score using train set stats only (no data leakage)

6. **Build Graphs** - Creates observation graphs for all patients
   
   **What is an observation graph?**
   
   Each measurement becomes a node in the graph. So if a patient has 150 measurements (HR readings, BP readings, lab values, etc.) over 48 hours, that's 150 nodes.
   
   **Node features**: `[timestamp, variable_id, value]`
   - Example node: `[0.5, 19, 88.5]` = HR measured at 24hr with value 88.5 bpm
   
   **Edges connect related observations:**
   
   *Temporal edges* (intra-series):
   - Same variable, close in time
   - Example: HR at 10:00 → HR at 10:30 (same sensor, 30 min apart)
   - Threshold: 2 hours (based on EDA - mean interval is 0.64hr)
   - Captures temporal dependencies within each variable
   
   *Variable edges* (inter-series):
   - Different variables, same timestamp
   - Example: HR at 10:00 → BP at 10:00 (measured together)
   - Captures physiological correlations (when vitals measured simultaneously)
   
   *Self-loops*:
   - Each node to itself (helps with stability)
   
   **Stats**: ~150 nodes, ~500 edges per patient (average)
   
   **Why 2-hour threshold?**
   - EDA shows mean interval = 0.64 ± 0.49 hours
   - 2 hours catches most consecutive measurements
   - Not too small (misses connections) or too large (connects everything)
   - Can tune this as hyperparameter later
   
   **Implementation**:
   - Fully vectorized (PyTorch broadcasting)
   - Fast: ~0.01-0.05 sec per patient vs 0.5-1 sec with Python loops
   - See code in `src/data/preprocessing.py` line ~340
   
   **Why pre-generate?**
   - Much faster training (graphs built once, not every epoch)
   - Consistent across all model runs
   - Team requested this ("waiting for graph generation")
   - Models just load and use them

7. **Save** - PyTorch tensors for fast loading

## Why No Outlier Removal?

From our report: *"robust models capable of handling extreme values that, while statistically outlying, may be clinically significant"*

We're testing if models can handle real ICU extremes. RMSE vs MAE ratio measures this.

## Data Format

```python
record_id, timestamps, values, mask = data[0]
# timestamps: (T,) normalized to [0,1]
# values: (T, 41) z-score normalized
# mask: (T, 41) 1=observed, 0=missing
```

## Using It

```python
from src.data.dataset import PhysioNetDataset
dataset = PhysioNetDataset('data/physionet/processed', split='train')

# Or for graph models
from src.data.dataset import PhysioNetGraphDataset
dataset = PhysioNetGraphDataset('data/physionet/processed', split='train')
# Loads pre-built graphs automatically
```

## Notes

- Preprocessing runs once, everyone uses same data (fair comparison)
- Graphs pre-built during preprocessing (faster training)
- Stats from train set only (proper ML practice)
- Extreme values kept (testing robustness)
