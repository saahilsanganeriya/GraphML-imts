import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from datasets.dataset import PhysioNetDataset, split_history_forecast
from models.seft.model import SeFT
from utils.metrics import compute_all_metrics


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    return batch  # Return list of tuples


def prepare_batch_for_seft(batch, device, n_variables=36):
    """
    Convert batch to SeFT format.

    Args:
        batch: List of (record_id, timestamps, values, mask)
        device: torch device
        n_variables: Number of time-series variables (excluding static)

    Returns:
        Dictionary with batched inputs
    """
    batch_size = len(batch)

    # Prepare lists for history observations
    all_times = []
    all_vars = []
    all_vals = []
    all_masks = []

    # Prepare lists for forecast queries
    all_query_times = []
    all_query_vars = []
    all_query_vals = []
    all_query_masks = []

    max_hist_len = 0
    max_fore_len = 0

    for record_id, timestamps, values, mask in batch:
        # Split history vs forecast
        hist_t, hist_v, hist_m, fore_t, fore_v, fore_m = split_history_forecast(
            timestamps, values, mask
        )

        # Focus on time-series variables only (skip first 5 static vars)
        hist_v_ts = hist_v[:, 5:]  # (T_hist, 36)
        hist_m_ts = hist_m[:, 5:]  # (T_hist, 36)
        fore_v_ts = fore_v[:, 5:]  # (T_fore, 36)
        fore_m_ts = fore_m[:, 5:]  # (T_fore, 36)

        # Convert to observation format: (time, var, value) for observed points
        hist_obs_t = []
        hist_obs_v = []
        hist_obs_val = []

        for t_idx in range(len(hist_t)):
            for v_idx in range(n_variables):
                if hist_m_ts[t_idx, v_idx] > 0:
                    hist_obs_t.append(hist_t[t_idx].item())
                    hist_obs_v.append(v_idx)
                    hist_obs_val.append(hist_v_ts[t_idx, v_idx].item())

        # Convert forecast to queries
        fore_obs_t = []
        fore_obs_v = []
        fore_obs_val = []
        fore_obs_m = []

        for t_idx in range(len(fore_t)):
            for v_idx in range(n_variables):
                if fore_m_ts[t_idx, v_idx] > 0:
                    fore_obs_t.append(fore_t[t_idx].item())
                    fore_obs_v.append(v_idx)
                    fore_obs_val.append(fore_v_ts[t_idx, v_idx].item())
                    fore_obs_m.append(1.0)

        # Skip patients with no history OR no forecast observations
        if len(hist_obs_t) == 0 or len(fore_obs_t) == 0:
            continue

        all_times.append(torch.tensor(hist_obs_t))
        all_vars.append(torch.tensor(hist_obs_v))
        all_vals.append(torch.tensor(hist_obs_val))
        all_masks.append(torch.ones(len(hist_obs_t)))

        all_query_times.append(torch.tensor(fore_obs_t))
        all_query_vars.append(torch.tensor(fore_obs_v))
        all_query_vals.append(torch.tensor(fore_obs_val))
        all_query_masks.append(torch.tensor(fore_obs_m))

        max_hist_len = max(max_hist_len, len(hist_obs_t))
        max_fore_len = max(max_fore_len, len(fore_obs_t))

    # Check if batch is empty (all patients skipped)
    if len(all_times) == 0 or len(all_query_times) == 0:
        return None

    # Pad sequences
    def pad_list(tensors, max_len):
        padded = torch.zeros(len(tensors), max_len)
        for i, t in enumerate(tensors):
            padded[i, :len(t)] = t
        return padded

    batch_times = pad_list(all_times, max_hist_len).to(device)
    batch_vars = pad_list(all_vars, max_hist_len).long().to(device)
    batch_vals = pad_list(all_vals, max_hist_len).to(device)
    batch_masks = pad_list(all_masks, max_hist_len).to(device)

    batch_query_times = pad_list(all_query_times, max_fore_len).to(device)
    batch_query_vars = pad_list(all_query_vars, max_fore_len).long().to(device)
    batch_query_vals = pad_list(all_query_vals, max_fore_len).to(device)
    batch_query_masks = pad_list(all_query_masks, max_fore_len).to(device)

    return {
        'timestamps': batch_times,
        'variables': batch_vars,
        'values': batch_vals,
        'mask': batch_masks,
        'query_times': batch_query_times,
        'query_vars': batch_query_vars,
        'query_values': batch_query_vals,
        'query_mask': batch_query_masks
    }


def train_epoch(model, dataloader, optimizer, device, n_variables=36):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        try:
            batch_data = prepare_batch_for_seft(batch, device, n_variables)

            # Skip empty batches
            if batch_data is None:
                continue

            # Forward pass
            predictions = model(
                batch_data['timestamps'],
                batch_data['variables'],
                batch_data['values'],
                batch_data['mask'],
                batch_data['query_times'],
                batch_data['query_vars']
            )

            # Compute loss (only on observed forecast points)
            query_mask = batch_data['query_mask']
            loss = F.mse_loss(
                predictions[query_mask > 0],
                batch_data['query_values'][query_mask > 0]
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': loss.item()})

        except Exception as e:
            print(f"Error in batch: {e}")
            continue

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, n_variables=36):
    """Evaluate model."""
    model.eval()
    all_predictions = []
    all_targets = []

    for batch in tqdm(dataloader, desc='Evaluating'):
        try:
            batch_data = prepare_batch_for_seft(batch, device, n_variables)

            # Skip empty batches
            if batch_data is None:
                continue

            # Forward pass
            predictions = model(
                batch_data['timestamps'],
                batch_data['variables'],
                batch_data['values'],
                batch_data['mask'],
                batch_data['query_times'],
                batch_data['query_vars']
            )

            # Collect predictions and targets (only observed)
            query_mask = batch_data['query_mask']
            all_predictions.append(predictions[query_mask > 0].cpu())
            all_targets.append(batch_data['query_values'][query_mask > 0].cpu())

        except Exception as e:
            print(f"Error in validation batch: {e}")
            continue

    if len(all_predictions) == 0:
        return {}

    # Concatenate all
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Compute metrics
    metrics = compute_all_metrics(all_predictions, all_targets, prefix='')

    return metrics


def main():
    # Configuration
    config = {
        'n_variables': 36,  # Time-series variables (excluding 5 static)
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'dropout': 0.1,
        'batch_size': 16,
        'lr': 1e-4,
        'epochs': 50,
        'patience': 10,
        'device': 'cpu'  # Force CPU - MPS doesn't support nested tensors used by SeFT
    }

    print("=" * 80)
    print("TRAINING SeFT MODEL")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, val in config.items():
        print(f"  {key}: {val}")

    device = torch.device(config['device'])
    print(f"\nUsing device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PhysioNetDataset('data/physionet/processed', split='train')
    val_dataset = PhysioNetDataset('data/physionet/processed', split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create model
    print("\nCreating model...")
    model = SeFT(
        n_variables=config['n_variables'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dropout=config['dropout']
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_loss = float('inf')
    patience_counter = 0
    results = []

    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, config['n_variables'])
        print(f"  Train loss: {train_loss:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device, config['n_variables'])
        val_rmse = val_metrics.get('rmse', float('inf'))
        val_mae = val_metrics.get('mae', float('inf'))

        print(f"  Val RMSE: {val_rmse:.4f}")
        print(f"  Val MAE: {val_mae:.4f}")

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_rmse': val_rmse,
            'val_mae': val_mae
        })

        # Learning rate scheduling
        scheduler.step(val_rmse)

        # Save best model
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'config': config
            }
            torch.save(checkpoint, 'results/seft_best.pt')
            print(f"  ✓ Saved best model (RMSE: {val_rmse:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config['patience']}")

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save final results
    with open('results/seft_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation RMSE: {best_val_loss:.4f}")
    print(f"Results saved to: results/seft_results.json")
    print(f"Best model saved to: results/seft_best.pt")


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    main()
