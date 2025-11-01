#!/usr/bin/env python3
"""
Train RainDrop for Forecasting

Uses official RainDrop model (modified) with our preprocessed data.
Task: Forecast 36 temporal variables at irregular timestamps in 24-30hr window
Input: 0-24hr history
Forecast: 24-30hr future (sequence prediction at observed timestamps)

Usage:
    python src/models/raindrop/train_raindrop_forecasting.py
    python src/models/raindrop/train_raindrop_forecasting.py --epochs 100 --batch-size 32
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

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
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns

from models_rd import Raindrop
from utils.metrics import compute_all_metrics

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RainDrop for Forecasting')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/physionet/processed',
                        help='Path to preprocessed data directory')
    
    # Model
    parser.add_argument('--d-model', type=int, default=72,
                        help='Model dimension (must be multiple of d_inp=36)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--distance-weight', type=float, default=0.02,
                        help='Weight for distance loss (RainDrop regularization)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (auto selects best available)')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision (AMP) for faster training on GPU')
    
    # Logging
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='graphml-imts',
                        help='WandB project name')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='WandB run name (default: auto-generated)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N batches')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/raindrop',
                        help='Output directory for checkpoints and plots')
    parser.add_argument('--save-best', action='store_true', default=True,
                        help='Save best model checkpoint')
    
    return parser.parse_args()


def get_device(device_arg='auto'):
    """Get PyTorch device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            device_name = "MPS (Apple Silicon)"
        else:
            device = torch.device('cpu')
            device_name = "CPU"
    else:
        device = torch.device(device_arg)
        device_name = device_arg.upper()
    
    return device, device_name


def load_data(data_dir, split='train'):
    """Load our preprocessed data."""
    data_path = Path(data_dir) / f'{split}.pt'
    data = torch.load(data_path)
    return data


def convert_to_raindrop_format(patients, device='cpu'):
    """
    Convert our format → RainDrop's format.
    
    Our format: (record_id, timestamps, values, mask) per patient
    - timestamps: (T,) normalized [0, 1] (48hr → [0, 1])
    - values: (T, 41) z-score normalized (5 static + 36 temporal)
    - mask: (T, 41) binary (1=observed, 0=missing)
    
    Their format: (T, B, D*2) batch tensors
    - src: (T, B, D*2) where D=36 (only temporal vars), last D dims are mask
    - times: (T, B) timestamps
    - static: (B, 9) static features
    - lengths: (B,) sequence lengths
    
    Task: Predict values at observed (timestamp, variable) pairs in 24-30hr window
    
    Returns:
        src: (T, B, D*2) - history values + mask (D=36 temporal vars)
        times: (T, B) - history timestamps
        static: (B, d_static) - static features (Age, Gender, Height, ICUType, Weight, etc.)
        lengths: (B,) - sequence lengths
        targets: List[(T_i, D)] - per-patient forecast values (variable length)
        target_masks: List[(T_i, D)] - per-patient forecast masks
        target_times: List[(T_i,)] - per-patient forecast timestamps
    """
    batch_data = []
    batch_times = []
    batch_static = []
    batch_lengths = []
    batch_targets = []
    batch_target_masks = []
    batch_target_times = []
    
    for record_id, tt, vals, mask in patients:
        # Split history (0-24hr) and forecast (24-30hr)
        history_mask = tt <= 0.5  # Normalized: 24/48 = 0.5
        forecast_mask = (tt > 0.5) & (tt <= 0.625)  # 30/48 = 0.625
        
        # Extract temporal variables only (columns 5-41 are the 36 temporal vars)
        # First 5 are static: Age, Gender, Height, ICUType, Weight
        temporal_vals = vals[:, 5:]  # (T, 36)
        temporal_mask = mask[:, 5:]  # (T, 36)
        
        # History
        hist_times = tt[history_mask]
        hist_vals = temporal_vals[history_mask]
        hist_mask = temporal_mask[history_mask]
        
        # Concatenate values and mask (RainDrop format: [values, mask])
        hist_data = torch.cat([hist_vals, hist_mask], dim=1)  # (T_hist, 72)
        
        # Static features (take first observation, first 5 columns)
        # But RainDrop expects 9 static features, so we'll pad or use what we have
        static = vals[0, :5]  # (5,)
        # Pad to 9 dimensions (RainDrop's d_static=9)
        static = torch.cat([static, torch.zeros(4)], dim=0)  # (9,)
        
        # Forecast targets (only temporal variables)
        forecast_times = tt[forecast_mask]
        forecast_vals = temporal_vals[forecast_mask]
        forecast_mask_vals = temporal_mask[forecast_mask]
        
        batch_data.append(hist_data)
        batch_times.append(hist_times)
        batch_static.append(static)
        batch_lengths.append(len(hist_times))
        batch_targets.append(forecast_vals)  # Variable length!
        batch_target_masks.append(forecast_mask_vals)
        batch_target_times.append(forecast_times)
    
    # Pad history sequences
    src = pad_sequence(batch_data, batch_first=False)  # (T_max, B, 72)
    times = pad_sequence(batch_times, batch_first=False)  # (T_max, B)
    static = torch.stack(batch_static)  # (B, 9)
    lengths = torch.tensor(batch_lengths)  # (B,)
    
    # Keep targets as lists (variable length per patient)
    # We'll handle them separately during loss computation
    
    # Move to device
    src = src.to(device)
    times = times.to(device)
    static = static.to(device)
    lengths = lengths.to(device)
    
    # Move targets to device (they're lists, so move each element)
    batch_targets = [t.to(device) for t in batch_targets]
    batch_target_masks = [t.to(device) for t in batch_target_masks]
    batch_target_times = [t.to(device) for t in batch_target_times]
    
    return src, times, static, lengths, batch_targets, batch_target_masks, batch_target_times


def compute_loss_and_metrics(predictions, targets, target_masks, target_times, criterion):
    """
    Compute loss and metrics for sequence forecasting.
    
    predictions: (B, 36, n_forecast_steps) - model output
    targets: List[(T_i, 36)] - ground truth per patient (variable length)
    target_masks: List[(T_i, 36)] - masks per patient
    target_times: List[(T_i,)] - timestamps per patient
    
    Strategy: For each patient, average predictions over forecast steps,
    then average target values over all observed timestamps in forecast window.
    Only compute loss on variables that have at least one observation.
    """
    batch_size = predictions.shape[0]
    
    # Average predictions over forecast steps: (B, 36)
    pred_mean = predictions.mean(dim=-1)  # (B, 36)
    
    # For each patient, compute average target over observed timestamps
    losses = []
    all_preds = []
    all_targets = []
    
    for i in range(batch_size):
        patient_targets = targets[i]  # (T_i, 36)
        patient_mask = target_masks[i]  # (T_i, 36)
        
        # Average over time for each variable (only where observed)
        # Sum over time, divide by count per variable
        sum_targets = (patient_targets * patient_mask).sum(dim=0)  # (36,)
        count_obs = patient_mask.sum(dim=0)  # (36,)
        
        # Variables with at least one observation
        valid_vars = count_obs > 0  # (36,)
        
        if valid_vars.sum() > 0:
            # Average target per variable
            avg_target = sum_targets[valid_vars] / count_obs[valid_vars]  # (n_valid,)
            pred_for_patient = pred_mean[i][valid_vars]  # (n_valid,)
            
            # Compute loss for this patient
            loss = criterion(pred_for_patient, avg_target)
            losses.append(loss)
            
            # Collect for metrics
            all_preds.append(pred_for_patient.detach())
            all_targets.append(avg_target.detach())
    
    # Average loss over batch
    if len(losses) > 0:
        total_loss = torch.stack(losses).mean()
        
        # Concatenate all predictions and targets for metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
    else:
        total_loss = torch.tensor(0.0, device=predictions.device)
        all_preds = torch.tensor([], device=predictions.device)
        all_targets = torch.tensor([], device=predictions.device)
    
    return total_loss, all_preds, all_targets


def train_epoch(model, data, batch_size, optimizer, criterion, device, distance_weight=0.02, use_amp=False, scaler=None):
    """Train one epoch."""
    model.train()
    
    # Shuffle data
    indices = torch.randperm(len(data))
    
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []
    
    # Process in batches
    for i in range(0, len(data), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_patients = [data[idx] for idx in batch_indices]
        
        # Convert to RainDrop format
        src, times, static, lengths, targets, target_masks, target_times = \
            convert_to_raindrop_format(batch_patients, device=device)
        
        # Forward pass (with optional AMP)
        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                predictions, distance, _ = model(src, static, times, lengths)
                loss, batch_preds, batch_targets = compute_loss_and_metrics(
                    predictions, targets, target_masks, target_times, criterion
                )
                if distance is not None:
                    loss = loss + distance_weight * distance
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular forward pass
            predictions, distance, _ = model(src, static, times, lengths)
            loss, batch_preds, batch_targets = compute_loss_and_metrics(
                predictions, targets, target_masks, target_times, criterion
            )
            if distance is not None:
                loss = loss + distance_weight * distance
            
            # Regular backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        n_batches += 1
        
        if len(batch_preds) > 0:
            all_preds.append(batch_preds.cpu())
            all_targets.append(batch_targets.cpu())
    
    # Average loss
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    # Compute metrics
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_all_metrics(all_preds, all_targets, prefix='train_')
    else:
        metrics = {}
    
    metrics['train_loss'] = avg_loss
    
    return metrics


def evaluate(model, data, batch_size, criterion, device, distance_weight=0.02):
    """Evaluate on validation/test set."""
    model.eval()
    
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch_patients = data[i:i + batch_size]
            
            # Convert to RainDrop format
            src, times, static, lengths, targets, target_masks, target_times = \
                convert_to_raindrop_format(batch_patients, device=device)
            
            # Forward pass
            predictions, distance, _ = model(src, static, times, lengths)
            
            # Compute loss
            loss, batch_preds, batch_targets = compute_loss_and_metrics(
                predictions, targets, target_masks, target_times, criterion
            )
            
            if distance is not None:
                loss = loss + distance_weight * distance
            
            total_loss += loss.item()
            n_batches += 1
            
            if len(batch_preds) > 0:
                all_preds.append(batch_preds.cpu())
                all_targets.append(batch_targets.cpu())
    
    # Average loss
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    # Compute metrics
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_all_metrics(all_preds, all_targets, prefix='val_')
    else:
        metrics = {}
    
    metrics['val_loss'] = avg_loss
    
    return metrics, all_preds, all_targets


def plot_loss_curves(train_losses, val_losses, output_dir):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'loss_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved loss curves to {output_path}")
    
    return str(output_path)


def plot_predictions_vs_actual(predictions, targets, output_dir, n_samples=1000):
    """Plot predictions vs actual values."""
    # Subsample if too many points
    if len(predictions) > n_samples:
        indices = np.random.choice(len(predictions), n_samples, replace=False)
        predictions = predictions[indices]
        targets = targets[indices]
    
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.3, s=10)
    
    # Diagonal line (perfect predictions)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values (Normalized)', fontsize=12)
    plt.ylabel('Predicted Values (Normalized)', fontsize=12)
    plt.title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'predictions_vs_actual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved predictions vs actual to {output_path}")
    
    return str(output_path)


def plot_metrics_over_time(train_metrics_history, val_metrics_history, output_dir):
    """Plot RMSE and MAE over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_metrics_history) + 1)
    
    # RMSE
    train_rmse = [m.get('train_rmse', np.nan) for m in train_metrics_history]
    val_rmse = [m.get('val_rmse', np.nan) for m in val_metrics_history]
    ax1.plot(epochs, train_rmse, 'b-', label='Train RMSE', linewidth=2)
    ax1.plot(epochs, val_rmse, 'r-', label='Val RMSE', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # MAE
    train_mae = [m.get('train_mae', np.nan) for m in train_metrics_history]
    val_mae = [m.get('val_mae', np.nan) for m in val_metrics_history]
    ax2.plot(epochs, train_mae, 'b-', label='Train MAE', linewidth=2)
    ax2.plot(epochs, val_mae, 'r-', label='Val MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'metrics_over_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved metrics over time to {output_path}")
    
    return str(output_path)


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Training RainDrop for Forecasting")
    print("=" * 80)
    
    # Device
    device, device_name = get_device(args.device)
    print(f"\nDevice: {device_name}")
    
    # Check if MPS is used
    if device.type == 'mps':
        print("⚠️  Using MPS (Apple Silicon). Training should work but might be slower than CUDA.")
        print("   If you encounter issues, use --device cpu or train on a CUDA machine.")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize WandB
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_name or f"raindrop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
        print(f"✓ WandB initialized (project: {args.wandb_project}, run: {run_name})")
    elif args.use_wandb:
        print("⚠️  WandB requested but not available. Install with: pip install wandb")
    
    # Load data
    print("\nLoading data...")
    train_data = load_data(args.data_dir, 'train')
    val_data = load_data(args.data_dir, 'val')
    print(f"   Train: {len(train_data)} patients")
    print(f"   Val: {len(val_data)} patients")
    
    # Create global structure (fully connected graph for sensors)
    global_structure = torch.ones(36, 36)  # 36 temporal variables
    
    # Create model
    print("\nCreating model...")
    model = Raindrop(
        d_inp=36,  # 36 temporal variables
        d_model=args.d_model,
        nhead=args.nhead,
        nhid=args.d_model * 2,  # Typical choice
        nlayers=args.nlayers,
        dropout=args.dropout,
        max_len=216,  # Max sequence length in data
        d_static=9,
        n_classes=2,  # Ignored in forecasting mode
        forecasting_mode=True,
        n_forecast_steps=6,  # Predict 6 future time steps
        global_structure=global_structure
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {n_params:,}")
    print(f"   Trainable parameters: {n_trainable:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    
    # Mixed precision training setup
    scaler = None
    if args.use_amp and (torch.cuda.is_available() or torch.backends.mps.is_available()):
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else torch.amp.GradScaler('cpu')
        print(f"\n✓ Using Automatic Mixed Precision (AMP) for faster training")
    
    # Training loop
    print("\nTraining...")
    print("-" * 80)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_data, args.batch_size, optimizer, criterion, device, 
            args.distance_weight, args.use_amp, scaler
        )
        
        # Evaluate
        val_metrics, val_preds, val_targets = evaluate(
            model, val_data, args.batch_size, criterion, device, args.distance_weight
        )
        
        # Store losses
        train_losses.append(train_metrics['train_loss'])
        val_losses.append(val_metrics['val_loss'])
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # Print metrics
        print(f"Epoch {epoch:3d}/{args.epochs}:")
        print(f"   Train Loss: {train_metrics['train_loss']:.4f} | "
              f"RMSE: {train_metrics.get('train_rmse', float('nan')):.4f} | "
              f"MAE: {train_metrics.get('train_mae', float('nan')):.4f} | "
              f"R²: {train_metrics.get('train_r2', float('nan')):.4f}")
        print(f"   Val   Loss: {val_metrics['val_loss']:.4f} | "
              f"RMSE: {val_metrics.get('val_rmse', float('nan')):.4f} | "
              f"MAE: {val_metrics.get('val_mae', float('nan')):.4f} | "
              f"R²: {val_metrics.get('val_r2', float('nan')):.4f}")
        
        # Log to WandB
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({**train_metrics, **val_metrics, 'epoch': epoch})
        
        # Save best model
        if args.save_best and val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, checkpoint_path)
            print(f"   ✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    print("-" * 80)
    print("✓ Training complete!")
    
    # Save final model
    final_checkpoint_path = output_dir / 'final_model.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'args': vars(args)
    }, final_checkpoint_path)
    print(f"\n✓ Saved final model to {final_checkpoint_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history,
            'args': vars(args)
        }, f, indent=2)
    print(f"✓ Saved training history to {history_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Loss curves
    loss_plot = plot_loss_curves(train_losses, val_losses, output_dir)
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({"loss_curves": wandb.Image(loss_plot)})
    
    # Metrics over time
    metrics_plot = plot_metrics_over_time(train_metrics_history, val_metrics_history, output_dir)
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({"metrics_over_time": wandb.Image(metrics_plot)})
    
    # Predictions vs actual
    if len(val_preds) > 0:
        pred_plot = plot_predictions_vs_actual(val_preds, val_targets, output_dir)
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({"predictions_vs_actual": wandb.Image(pred_plot)})
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    if len(val_metrics_history) > 0:
        best_epoch_idx = np.argmin(val_losses)
        best_epoch = best_epoch_idx + 1
        best_val_metrics = val_metrics_history[best_epoch_idx]
        print(f"Best epoch: {best_epoch}")
        print(f"   Val RMSE: {best_val_metrics.get('val_rmse', float('nan')):.4f}")
        print(f"   Val MAE: {best_val_metrics.get('val_mae', float('nan')):.4f}")
        print(f"   Val R²: {best_val_metrics.get('val_r2', float('nan')):.4f}")
    print("=" * 80)
    
    # Finish WandB
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"\n✓ All results saved to {output_dir}")
    print("\n🎉 Done!")


if __name__ == '__main__':
    main()
