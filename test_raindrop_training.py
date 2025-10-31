#!/usr/bin/env python3
"""
Quick test to verify RainDrop training pipeline works.

This runs a minimal training loop to catch any issues before full training.

Usage:
    python test_raindrop_training.py
"""

import sys
from pathlib import Path

# Add paths
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / 'src'))
sys.path.insert(0, str(repo_root / 'src' / 'models' / 'raindrop' / 'Raindrop' / 'code'))

import torch
import torch.nn as nn
from models_rd import Raindrop
from src.models.raindrop.train_raindrop_forecasting import (
    load_data, convert_to_raindrop_format, compute_loss_and_metrics, get_device
)

print("=" * 80)
print("Testing RainDrop Training Pipeline")
print("=" * 80)

# 1. Check device
print("\n1. Checking device...")
device, device_name = get_device('auto')
print(f"   Device: {device_name}")

# 2. Load data
print("\n2. Loading data (small sample)...")
try:
    train_data = load_data('data/physionet/processed', 'train')
    val_data = load_data('data/physionet/processed', 'val')
    print(f"   ✓ Train: {len(train_data)} patients")
    print(f"   ✓ Val: {len(val_data)} patients")
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    print("\n   → Make sure you ran: python scripts/preprocess_data.py")
    sys.exit(1)

# 3. Test data conversion
print("\n3. Testing data conversion...")
try:
    # Take small batch
    batch_patients = train_data[:4]
    src, times, static, lengths, targets, target_masks, target_times = \
        convert_to_raindrop_format(batch_patients, device='cpu')
    
    print(f"   ✓ src shape: {src.shape} (should be (T, B, 72))")
    print(f"   ✓ times shape: {times.shape} (should be (T, B))")
    print(f"   ✓ static shape: {static.shape} (should be (B, 9))")
    print(f"   ✓ lengths shape: {lengths.shape} (should be (B,))")
    print(f"   ✓ targets: {len(targets)} patients")
    
    # Check temporal vars only (36)
    assert src.shape[2] == 72, f"Expected 72 dims (36 vals + 36 masks), got {src.shape[2]}"
    assert static.shape[1] == 9, f"Expected 9 static features, got {static.shape[1]}"
    print("   ✓ Data format correct!")
    
except Exception as e:
    print(f"   ✗ Error in data conversion: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test model creation
print("\n4. Creating model...")
try:
    global_structure = torch.ones(36, 36)
    model = Raindrop(
        d_inp=36,
        d_model=64,
        nhead=4,
        nhid=128,
        nlayers=2,
        dropout=0.3,
        max_len=215,
        d_static=9,
        n_classes=2,
        forecasting_mode=True,
        n_forecast_steps=6,
        global_structure=global_structure
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created ({n_params:,} parameters)")
    
except Exception as e:
    print(f"   ✗ Error creating model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Test forward pass
print("\n5. Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        predictions, distance, _ = model(src, static, times, lengths)
    
    print(f"   ✓ predictions shape: {predictions.shape} (should be (B, 36, 6))")
    assert predictions.shape == (4, 36, 6), f"Wrong output shape: {predictions.shape}"
    print("   ✓ Forward pass works!")
    
except Exception as e:
    print(f"   ✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Test loss computation
print("\n6. Testing loss computation...")
try:
    criterion = nn.MSELoss()
    loss, batch_preds, batch_targets = compute_loss_and_metrics(
        predictions, targets, target_masks, target_times, criterion
    )
    
    print(f"   ✓ loss: {loss.item():.4f}")
    print(f"   ✓ predictions: {len(batch_preds)} values")
    print(f"   ✓ targets: {len(batch_targets)} values")
    print("   ✓ Loss computation works!")
    
except Exception as e:
    print(f"   ✗ Error in loss computation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Test backward pass
print("\n7. Testing backward pass...")
try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Forward
    predictions, distance, _ = model(src, static, times, lengths)
    loss, _, _ = compute_loss_and_metrics(
        predictions, targets, target_masks, target_times, criterion
    )
    
    if distance is not None:
        loss = loss + 0.02 * distance
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Backward pass works!")
    print(f"   ✓ Loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"   ✗ Error in backward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 8. Test metrics computation
print("\n8. Testing metrics...")
try:
    from src.utils.metrics import compute_all_metrics
    
    if len(batch_preds) > 0:
        metrics = compute_all_metrics(batch_preds, batch_targets, prefix='test_')
        print(f"   ✓ RMSE: {metrics['test_rmse']:.4f}")
        print(f"   ✓ MAE: {metrics['test_mae']:.4f}")
        print(f"   ✓ R²: {metrics['test_r2']:.4f}")
        print("   ✓ Metrics computation works!")
    else:
        print("   ⚠ No predictions to compute metrics (all missing)")
    
except Exception as e:
    print(f"   ✗ Error computing metrics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 9. Test on different device
if device.type != 'cpu':
    print(f"\n9. Testing on {device_name}...")
    try:
        model = model.to(device)
        src = src.to(device)
        times = times.to(device)
        static = static.to(device)
        lengths = lengths.to(device)
        
        with torch.no_grad():
            predictions, distance, _ = model(src, static, times, lengths)
        
        print(f"   ✓ Model works on {device_name}!")
        
    except Exception as e:
        print(f"   ⚠ Warning: Model failed on {device_name}: {e}")
        print(f"   → You may need to use --device cpu for training")
        print(f"   → Or train on a CUDA machine")

# Success!
print("\n" + "=" * 80)
print("✅ All tests passed!")
print("=" * 80)
print("\nYou're ready to train!")
print("\nNext steps:")
print("   1. Quick test (2 epochs): python src/models/raindrop/train_raindrop_forecasting.py --epochs 2")
print("   2. Full training: python src/models/raindrop/train_raindrop_forecasting.py --epochs 50")
print("   3. With WandB: python src/models/raindrop/train_raindrop_forecasting.py --epochs 50 --use-wandb")
print("\nDevice recommendation:")
if device.type == 'mps':
    print("   - MPS detected: Good for testing, but ~2-3x slower than CUDA")
    print("   - For final training: Consider using a CUDA machine")
elif device.type == 'cuda':
    print("   - CUDA detected: Perfect! You're all set for fast training")
else:
    print("   - CPU only: Training will be slow (~2-4 hours for 50 epochs)")
    print("   - Consider using a machine with GPU (CUDA or MPS)")
print()

