#!/usr/bin/env python3
"""
Train RainDrop for Forecasting

Uses official RainDrop model (modified) with our preprocessed data.

Usage:
    python src/models/raindrop/train_raindrop_forecasting.py
"""

import sys
import os
from pathlib import Path

# MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add paths
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / 'src'))
sys.path.insert(0, str(repo_root / 'src' / 'models' / 'raindrop' / 'Raindrop' / 'code'))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models_rd import Raindrop
from utils.metrics import compute_all_metrics

# Load our preprocessed data
def load_our_data(split='train'):
    """Load our preprocessed data."""
    data = torch.load(f'data/physionet/processed/{split}.pt')
    return data

def convert_to_raindrop_format(patients, max_len=215):
    """
    Convert our format → RainDrop's format.
    
    Our format: (record_id, timestamps, values, mask) per patient
    Their format: (T, B, D*2) batch tensors
    
    Returns:
        src: (T, B, D*2) - values + mask concatenated
        times: (T, B) - timestamps  
        static: (B, d_static) - static features
        lengths: (B,) - sequence lengths
        targets: (T, B, D) - forecast targets
        target_mask: (T, B, D) - forecast mask
    """
    from torch.nn.utils.rnn import pad_sequence
    
    batch_data = []
    batch_times = []
    batch_static = []
    batch_lengths = []
    batch_targets = []
    batch_target_masks = []
    
    for record_id, tt, vals, mask in patients:
        # Split history (0-24hr) and forecast (24-30hr)
        history_mask = tt <= 0.5  # Normalized: 24/48 = 0.5
        forecast_mask = (tt > 0.5) & (tt <= 0.625)  # 30/48 = 0.625
        
        # History
        hist_times = tt[history_mask]
        hist_vals = vals[history_mask]
        hist_mask = mask[history_mask]
        
        # Concatenate values and mask (their format)
        hist_data = torch.cat([hist_vals, hist_mask], dim=1)  # (T, D*2)
        
        # Static features (first row, first 9 columns: Age, Gender, Height, ICUType, Weight, etc.)
        static = vals[0, :9]
        
        # Forecast targets
        forecast_vals = vals[forecast_mask]
        forecast_mask_vals = mask[forecast_mask]
        
        batch_data.append(hist_data)
        batch_times.append(hist_times)
        batch_static.append(static)
        batch_lengths.append(len(hist_times))
        batch_targets.append(forecast_vals)
        batch_target_masks.append(forecast_mask_vals)
    
    # Pad sequences
    src = pad_sequence(batch_data, batch_first=False)  # (T, B, D*2)
    times = pad_sequence(batch_times, batch_first=False)  # (T, B)
    static = torch.stack(batch_static)  # (B, 9)
    lengths = torch.tensor(batch_lengths)  # (B,)
    targets = pad_sequence(batch_targets, batch_first=False)  # (T, B, D)
    target_masks = pad_sequence(batch_target_masks, batch_first=False)  # (T, B, D)
    
    return src, times, static, lengths, targets, target_masks

def train_epoch(model, train_data, optimizer, device):
    """Train one epoch."""
    model.train()
    
    # Convert data
    src, times, static, lengths, targets, target_masks = convert_to_raindrop_format(train_data[:128])  # Batch
    
    src = src.to(device)
    times = times.to(device)
    static = static.to(device)
    targets = targets.to(device)
    target_masks = target_masks.to(device)
    
    # Forward
    predictions, distance, _ = model(src, static, times, lengths)
    # predictions: (B, D, n_forecast_steps)
    
    # Average predictions over forecast steps
    pred_mean = predictions.mean(dim=-1)  # (B, D)
    
    # Target: average over forecast window
    target_mean = (targets * target_masks).sum(dim=0) / (target_masks.sum(dim=0) + 1e-8)  # (B, D)
    
    # Mask: only variables with observations
    valid_mask = target_masks.sum(dim=0) > 0  # (B, D)
    
    # MSE loss
    criterion = nn.MSELoss()
    loss = criterion(pred_mean[valid_mask], target_mean[valid_mask])
    
    # Add distance loss (from official RainDrop)
    if distance is not None:
        loss = loss + 0.02 * distance
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    print("Training RainDrop for Forecasting")
    print("="*60)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_data = load_our_data('train')
    val_data = load_our_data('val')
    print(f"Train: {len(train_data)} patients")
    print(f"Val: {len(val_data)} patients")
    
    # Create global structure (fully connected initially)
    global_structure = torch.ones(36, 36)
    
    # Create model
    print("\nCreating model...")
    model = Raindrop(
        d_inp=36,
        d_model=64,
        nhead=4,
        nhid=128,
        nlayers=2,
        dropout=0.3,
        max_len=215,
        d_static=9,
        n_classes=2,  # Ignored in forecasting mode
        forecasting_mode=True,  # NEW
        n_forecast_steps=6,     # NEW
        global_structure=global_structure
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Train
    print("\nTraining...")
    for epoch in range(1, 6):  # 5 epochs for testing
        loss = train_epoch(model, train_data, optimizer, device)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    print("\n✓ Training complete!")
    print("Save checkpoint: results/raindrop_forecasting.pt")
    
    # Save
    os.makedirs('results/checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'results/checkpoints/raindrop_forecasting.pt')

if __name__ == '__main__':
    main()

