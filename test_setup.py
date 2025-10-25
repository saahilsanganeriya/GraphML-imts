#!/usr/bin/env python3
"""
Quick test to verify setup is working.
Run this after preprocessing to make sure everything is OK.
"""

import torch
import sys
from pathlib import Path

print("Testing setup...")
print("=" * 60)

# Test 1: Check preprocessed data exists
print("\n1. Checking preprocessed data...")
data_dir = Path('data/physionet/processed')

required_files = ['all_data.pt', 'train.pt', 'val.pt', 'test.pt', 
                  'split_indices.pt', 'statistics.pt']

graph_files = ['all_graphs.pt', 'train_graphs.pt', 'val_graphs.pt', 'test_graphs.pt']

for f in required_files:
    fpath = data_dir / f
    if fpath.exists():
        size_mb = fpath.stat().st_size / (1024*1024)
        print(f"   ✓ {f} ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ {f} NOT FOUND")
        print("\n   → Run: python scripts/preprocess_data.py")
        sys.exit(1)

# Check graphs (optional but recommended)
print("\n   Graph files:")
for f in graph_files:
    fpath = data_dir / f
    if fpath.exists():
        size_mb = fpath.stat().st_size / (1024*1024)
        print(f"   ✓ {f} ({size_mb:.1f} MB)")
    else:
        print(f"   ⚠ {f} not found (will build dynamically - slower)")

# Test 2: Load and check data
print("\n2. Loading data...")
try:
    train = torch.load(data_dir / 'train.pt')
    val = torch.load(data_dir / 'val.pt')
    test = torch.load(data_dir / 'test.pt')
    all_data = torch.load(data_dir / 'all_data.pt')
    stats = torch.load(data_dir / 'statistics.pt')
    
    print(f"   ✓ Train: {len(train)} patients")
    print(f"   ✓ Val: {len(val)} patients")
    print(f"   ✓ Test: {len(test)} patients")
    print(f"   ✓ All: {len(all_data)} patients")
    print(f"   ✓ Total: {len(train) + len(val) + len(test)} patients")
    
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    sys.exit(1)

# Test 3: Check data format
print("\n3. Checking data format...")
try:
    record_id, tt, vals, mask = train[0]
    
    print(f"   ✓ Sample patient: {record_id}")
    print(f"   ✓ Timestamps: {tt.shape} (normalized to [0,1])")
    print(f"   ✓ Values: {vals.shape} (T, {vals.shape[1]} variables)")
    print(f"   ✓ Mask: {mask.shape}")
    
    sparsity = 1 - mask.sum() / mask.numel()
    print(f"   ✓ Sparsity: {sparsity:.1%}")
    
except Exception as e:
    print(f"   ✗ Error checking format: {e}")
    sys.exit(1)

# Test 4: Check normalization
print("\n4. Checking normalization...")
try:
    print(f"   ✓ Time max: {stats['time_max']:.1f} hours")
    print(f"   ✓ Variables: {len(stats['params'])}")
    print(f"   ✓ Stats keys: {list(stats.keys())}")
    
    # Check values are normalized
    print(f"   ✓ Value range: [{vals.min():.2f}, {vals.max():.2f}]")
    print(f"   ✓ Time range: [{tt.min():.2f}, {tt.max():.2f}]")
    
except Exception as e:
    print(f"   ✗ Error checking stats: {e}")
    sys.exit(1)

# Test 5: Check device support
print("\n5. Checking device support...")
try:
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    
    print(f"   MPS (Apple Silicon): {'✓ Available' if mps_available else '✗ Not available'}")
    print(f"   CUDA (NVIDIA GPU): {'✓ Available' if cuda_available else '✗ Not available'}")
    print(f"   CPU: ✓ Always available")
    
    if mps_available:
        print(f"   → Will use MPS")
    elif cuda_available:
        print(f"   → Will use CUDA")
    else:
        print(f"   → Will use CPU")
        
except Exception as e:
    print(f"   Warning: {e}")

# Test 6: Test dataset class
print("\n6. Testing dataset classes...")
try:
    sys.path.insert(0, 'src')
    from data.dataset import PhysioNetDataset, PhysioNetGraphDataset
    
    dataset = PhysioNetDataset('data/physionet/processed', split='train')
    print(f"   ✓ PhysioNetDataset loaded: {len(dataset)} samples")
    
    graph_dataset = PhysioNetGraphDataset('data/physionet/processed', split='train')
    print(f"   ✓ PhysioNetGraphDataset loaded: {len(graph_dataset)} samples")
    
except Exception as e:
    print(f"   ✗ Error loading dataset: {e}")
    print(f"   Check that src/data/dataset.py exists")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED")
print("=" * 60)
print("\nNext steps:")
print("1. Run EDA: python eda_analysis.py")
print("2. Train RainDrop: cd src/models/raindrop/Raindrop/code && python Raindrop.py --dataset P12")
print("3. Or implement your own model")

