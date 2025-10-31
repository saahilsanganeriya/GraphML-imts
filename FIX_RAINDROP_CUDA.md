# Fix RainDrop CUDA Hardcoding

## Problem

The official RainDrop code has hardcoded `.cuda()` calls that will cause **segmentation faults** on MPS or CPU.

Found 14+ instances of `.cuda()` in `src/models/raindrop/Raindrop/code/models_rd.py`:
- Line 43: `pe = pe.cuda()`
- Line 83-84: Edge tensors
- Line 156: Mask
- Line 163: Adjacency
- Line 169-170: Output tensors
- And more...

## Solution

You have 2 options:

### Option 1: Quick Fix (Use Official RainDrop's Data Format)

Train on a **CUDA machine** using their original preprocessing:

```bash
# SSH into CUDA server
ssh your_cuda_server

# Clone and setup
cd GraphML-imts/src/models/raindrop/Raindrop
cd P12data/process_scripts

# Run their preprocessing
python ParseData.py

# Then use their training script
cd ../../code
python Raindrop.py --dataset P12
```

This avoids the device issue entirely.

### Option 2: Patch models_rd.py (Device-Agnostic)

Replace all `.cuda()` calls with device-aware code:

```python
# Instead of: tensor.cuda()
# Use: tensor.to(device)
```

**Steps**:

1. Open `src/models/raindrop/Raindrop/code/models_rd.py`

2. Add device parameter to `__init__`:
```python
def __init__(self, d_inp=36, d_model=64, ..., global_structure=None, device='cuda'):
    super(Raindrop, self).__init__()
    self.device = device  # ADD THIS
    # ... rest of init
```

3. Replace all `.cuda()` with `.to(self.device)`:
```python
# Line 43
pe = pe.to(self.device)

# Line 83-84
self.edge_type_train = torch.ones([36*36*2], dtype=torch.int64).to(self.device)
self.adj = torch.ones([36, 36]).to(self.device)

# Line 156
mask = mask.squeeze(1).to(self.device)

# ... and so on for all 14+ occurrences
```

4. Update training script to pass device:
```python
model = Raindrop(
    d_inp=36,
    d_model=64,
    # ... other args ...
    global_structure=global_structure,
    device=device  # ADD THIS
).to(device)
```

**Full list of lines to change**:
- 43, 83, 84, 156, 163, 169, 170, 257, 259, 317, 325, 333, 337, 339, 387

### Option 3: Use Their Baselines Instead

The official repo includes GRU-D and SeFT in `Raindrop/code/baselines/` which might be easier to adapt.

## Recommended Approach

**For quick results**: Use Option 1 (CUDA server + their preprocessing)

**For your pipeline**: Use Option 2 (patch the code) - I can create a patched version if needed

**For debugging**: Test with simple forward pass to isolate the issue

## Testing After Fix

```bash
# Test on MPS
python test_raindrop_training.py --device mps

# Test on CPU
python test_raindrop_training.py --device cpu

# Should work on CUDA
python test_raindrop_training.py --device cuda
```

## Alternative: Use mTAND or SeFT

The `Raindrop/code/baselines/` folder has other models that might be easier to adapt:
- **mTAND**: Multi-time attention networks
- **SeFT**: Set functions for time series  
- **GRU-D**: Temporal decay RNN

These might have better device handling.

## Quick Workaround for Testing

If you just want to test the pipeline without fixing everything:

```bash
# Force CPU mode in training script
export CUDA_VISIBLE_DEVICES=""

# Then run
python src/models/raindrop/train_raindrop_forecasting.py --device cpu --epochs 2
```

But this will still crash due to hardcoded `.cuda()` calls.

## Need Help?

I can:
1. Create a fully patched version of `models_rd.py`
2. Write a device-aware wrapper class
3. Help adapt SeFT or GRU-D instead (might be simpler)

Let me know which approach you prefer!

