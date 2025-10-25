# RainDrop Baseline - Official Implementation Modified for Forecasting

## What This Is

Using the **official RainDrop repository** with minimal modifications to change from classification → forecasting.

**Location**: `Raindrop/` folder

## Modifications Made

**File**: `Raindrop/code/models_rd.py`

**Changes** (~15 lines):
1. Added `forecasting_mode` and `n_forecast_steps` parameters to `__init__`
2. Modified output head to predict D×n_forecast_steps values
3. Reshaped output to (B, D, n_forecast_steps)

**Everything else is original**:
- Sensor graph construction
- Transformer encoder  
- Message passing
- Their preprocessing

## Usage

### Classification (Original)
```bash
cd Raindrop/code
python Raindrop.py --dataset P12
```

### Forecasting (Modified)
```python
from models_rd import Raindrop

# Create model with forecasting mode
model = Raindrop(
    d_inp=36,
    d_model=64,
    forecasting_mode=True,
    n_forecast_steps=6,
    global_structure=global_structure
)

# Forward pass
predictions, distance, _ = model(src, static, times, lengths)
# predictions: (B, 36, 6) instead of (B, 2)

# Use MSE loss instead of CrossEntropy
criterion = nn.MSELoss()
loss = criterion(predictions[mask], targets[mask])
```

## Data Format (Their Preprocessing)

They use:
- `PTdict_list.npy` - Time series data
- `arr_outcomes.npy` - Labels (we can use for splits)
- Format: `dict['arr']` = (T, D), `dict['time']` = (T, D)

**Same P12 dataset as ours**, just different preprocessing format.

## Next Steps

**Option A**: Use their preprocessing
- Run `Raindrop/P12data/process_scripts/ParseData.py`
- Proven to work with their model
- Create forecast targets (split at 24hr)

**Option B**: Adapt to our preprocessing
- Use `data/physionet/processed/`
- Need data loader adapter
- All baselines use same data (fair comparison)

## Recommendation

For quick baseline: Use their preprocessing (Option A)

For final comparison: Adapt to our preprocessing (Option B) so all models use same data.

## What's Included

Their repo includes baselines:
- GRU-D (can use for our project!)
- SeFT (can use!)
- mTAND
- Transformer

Located in `Raindrop/code/baselines/`

These can also be modified for forecasting the same way.
