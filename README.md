# Graph ML for Irregular Time Series

Forecasting ICU patient vitals using graph neural networks. Class project for GML.

**Team**: Asal Roudbari, Luca Gianantonio, Saahil Sanganeria, Nawal Reza

## Quick Start

```bash
# Install
mamba env create -f environment.yml
conda activate graphml-imts
pip install torch-geometric torch-scatter torch-sparse

# Preprocess data (~20 min)
python scripts/preprocess_data.py

# Verify it worked
python test_setup.py
```

## What We're Doing

**Task**: Predict patient vitals/labs 6-24 hours ahead using 24hr history

**Dataset**: PhysioNet Challenge 2012
- 12,000 ICU patients
- 48-hour windows
- 41 variables (vitals + labs)
- ~86% missing data (irregular sampling)


## Repository

```
├── data/physionet/          # Raw & processed data
├── src/
│   ├── data/                # Preprocessing, dataset classes
│   ├── models/raindrop/Raindrop/  # Official RainDrop (modified)
│   ├── models/hipatch/      # Hi-Patch reference
│   └── utils/               # Metrics
├── scripts/preprocess_data.py  # Main preprocessing
├── eda_analysis.py          # Exploratory analysis
├── test_setup.py            # Verify setup
├── docs/                    # DATA_PREPROCESSING.md, RAINDROP.md
└── Reports/                 # LaTeX files
```

## Preprocessing (Run Once)

```bash
python scripts/preprocess_data.py
```

**What it does**:
1. Downloads PhysioNet data (12K patients)
2. Validates (removes NaN/inf only - keeps outliers!)
3. Splits (60% train, 20% val, 20% test, seed=42)
4. Normalizes (z-score, train stats only)
5. **Builds observation graphs** for all patients (~150 nodes, ~500 edges each)
6. Saves everything

**Creates**:
```
data/physionet/processed/
├── train.pt, val.pt, test.pt                    # Patient data
├── train_graphs.pt, val_graphs.pt, test_graphs.pt  # Pre-built graphs
└── statistics.pt                                 # Normalization stats
```

**Important**: Keeps extreme values (HR=300, Glucose=800)! These are real ICU patients in critical condition, not measurement errors. Per our report: *"robust models capable of handling extreme values that, while statistically outlying, may be clinically significant"*

See `docs/DATA_PREPROCESSING.md` for details on graph construction.

## Observation Graphs

Each patient gets a graph where:
- **Nodes**: Individual measurements (HR at 10:00, BP at 10:30, etc.)
- **Edges**: 
  - Temporal: same variable, <2hr apart (HR 10:00 → HR 10:30)
  - Variable: different variables, same time (HR 10:00 → BP 10:00)

Pre-generated during preprocessing = faster training.

## Using Data in Your Model

```python
from src.data.dataset import PhysioNetDataset

dataset = PhysioNetDataset('data/physionet/processed', split='train')
for sample in dataset:
    timestamps = sample['timestamps']  # (T,) normalized [0,1]
    values = sample['values']          # (T, 41) z-score normalized
    mask = sample['mask']              # (T, 41) 1=observed, 0=missing
    # Your model here

# For graph models (RainDrop, Hi-Patch)
from src.data.dataset import PhysioNetGraphDataset
dataset = PhysioNetGraphDataset('data/physionet/processed', split='train')
# Automatically loads pre-built graphs
```


## EDA (Optional)

```bash
python eda_analysis.py  # Run before preprocessing to see raw data
```

Creates 10+ visualizations in `eda_results/`. See `EDA_SUMMARY.md` for stats.


## Key Points

- **Preprocessing**: Run once, everyone uses same data (reproducibility)
- **No outlier removal**: Testing model robustness to extremes
- **Graphs pre-built**: Faster training, consistent across models
- **Fair comparison**: All models use same splits, same normalization
- **Device**: Auto-detects MPS (Mac M1/M2) / CUDA / CPU

## Files

**Run these**:
- `eda_analysis.py` - Visualize data
- `scripts/preprocess_data.py` - Process data
- `test_setup.py` - Verify it worked

**Read these**:
- `docs/DATA_PREPROCESSING.md` - How preprocessing works
- `docs/RAINDROP.md` - RainDrop baseline
- `EDA_SUMMARY.md` - Dataset stats

**Code**:
- `src/data/preprocessing.py` - Preprocessing pipeline
- `src/data/dataset.py` - Dataset loaders
- `src/utils/metrics.py` - Evaluation metrics

## References

LaTeX reports in `Reports/` (proposal, midterm, references)
