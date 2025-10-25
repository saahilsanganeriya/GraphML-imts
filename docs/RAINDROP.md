# RainDrop Baseline

Using official RainDrop code, modified to do forecasting instead of classification.

## Quick Start

```bash
# Install torch-sparse first if needed
pip install torch-sparse

# Train
python src/models/raindrop/train_raindrop_forecasting.py
```

## What I Changed

Modified official code in `src/models/raindrop/Raindrop/code/models_rd.py`:

1. Added `forecasting_mode` parameter
2. Output head: 2 classes → 36 sensors × 6 forecast steps
3. Reshaped output to (batch, 36, 6)

That's it - ~15 lines changed. Everything else is their original code.

## Why Use Official Code?

- They work with P12 (same dataset!)
- Architecture is tested
- Just changing output head is way easier than reimplementing
- Can use their baselines too (GRU-D, SeFT in `Raindrop/code/baselines/`)

## How It Works

```python
model = Raindrop(
    d_inp=36,
    forecasting_mode=True,  # NEW
    n_forecast_steps=6      # NEW
)

predictions, _, _ = model(src, static, times, lengths)
# predictions: (B, 36, 6) instead of (B, 2)

# Loss
criterion = nn.MSELoss()
loss = criterion(predictions[mask], targets[mask])
```

## Data Conversion

Official RainDrop uses format: `(T, B, D*2)` batch tensors

Our preprocessing uses: `(record_id, timestamps, values, mask)` per patient

Training script (`train_raindrop_forecasting.py`) converts automatically:
- Loads our data
- Pads to batch tensors
- Splits history (0-24hr) vs forecast (24-30hr)
- Feeds to model

## Training

```bash
python src/models/raindrop/train_raindrop_forecasting.py
```

Trains on our preprocessed data, saves checkpoint to `results/checkpoints/`

## Architecture

Official RainDrop (all kept the same):
1. Observation embedding
2. Sensor graph (message passing)
3. Transformer (temporal)
4. Aggregation
5. Output head ← only this changed

## Notes

- Official RainDrop does classification, ours does forecasting
- Same P12 dataset (11,988 patients, 36 sensors, 48hr)
- Modification is minimal (15 lines)
- Uses our preprocessed data (fair comparison with other models)
