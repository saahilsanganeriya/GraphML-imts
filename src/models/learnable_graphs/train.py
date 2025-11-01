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
from models.learnable_graphs.model import LearnableGraphModel
from utils.metrics import compute_all_metrics


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    return batch


def prepare_batch_for_graph(batch, device, n_variables=36):
    """
    Convert batch to Learnable Graph format.

    Args:
        batch: List of (record_id, timestamps, values, mask)
        device: torch device
        n_variables: Number of time-series variables

    Returns:
        Dictionary with batched inputs
    """
    batch_size = len(batch)

    # Prepare lists
    all_times = []
    all_vars = []
    all_vals = []
    all_masks = []
    all_targets = []
    all_target_masks = []

    max_len = 0

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

        # Convert to observation format
        hist_obs_t = []
        hist_obs_v = []
        hist_obs_val = []

        for t_idx in range(len(hist_t)):
            for v_idx in range(n_variables):
                if hist_m_ts[t_idx, v_idx] > 0:
                    hist_obs_t.append(hist_t[t_idx].item())
                    hist_obs_v.append(v_idx)
                    hist_obs_val.append(hist_v_ts[t_idx, v_idx].item())

        if len(hist_obs_t) == 0:
            continue

        # Aggregate forecast targets per variable (mean over forecast window)
        target_vals = torch.zeros(n_variables)
        target_mask = torch.zeros(n_variables)

        for v_idx in range(n_variables):
            v_mask = fore_m_ts[:, v_idx] > 0
            if v_mask.any():
                target_vals[v_idx] = fore_v_ts[v_mask, v_idx].mean()
                target_mask[v_idx] = 1.0

        all_times.append(torch.tensor(hist_obs_t))
        all_vars.append(torch.tensor(hist_obs_v))
        all_vals.append(torch.tensor(hist_obs_val))
        all_masks.append(torch.ones(len(hist_obs_t)))
        all_targets.append(target_vals)
        all_target_masks.append(target_mask)

        max_len = max(max_len, len(hist_obs_t))

    # Check if batch is empty (all patients skipped)
    if len(all_times) == 0:
        return None

    # Pad sequences
    def pad_list(tensors, max_len):
        padded = torch.zeros(len(tensors), max_len)
        for i, t in enumerate(tensors):
            padded[i, :len(t)] = t
        return padded

    batch_times = pad_list(all_times, max_len).to(device)
    batch_vars = pad_list(all_vars, max_len).long().to(device)
    batch_vals = pad_list(all_vals, max_len).to(device)
    batch_masks = pad_list(all_masks, max_len).to(device)
    batch_targets = torch.stack(all_targets).to(device)  # (B, V)
    batch_target_masks = torch.stack(all_target_masks).to(device)  # (B, V)

    return {
        'timestamps': batch_times,
        'variables': batch_vars,
        'values': batch_vals,
        'mask': batch_masks,
        'targets': batch_targets,
        'target_mask': batch_target_masks
    }


def train_epoch(model, dataloader, optimizer, device, n_variables=36):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        try:
            batch_data = prepare_batch_for_graph(batch, device, n_variables)

            # Skip empty batches
            if batch_data is None:
                continue

            # Forward pass
            predictions, adjacency = model(
                batch_data['timestamps'],
                batch_data['variables'],
                batch_data['values'],
                batch_data['mask']
            )  # predictions: (B, V)

            # Compute loss (only on observed variables)
            target_mask = batch_data['target_mask']
            loss = F.mse_loss(
                predictions[target_mask > 0],
                batch_data['targets'][target_mask > 0]
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
def evaluate(model, dataloader, device, n_variables=36, return_adjacency=False):
    """Evaluate model."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_adjacencies = []

    for batch in tqdm(dataloader, desc='Evaluating'):
        try:
            batch_data = prepare_batch_for_graph(batch, device, n_variables)

            # Skip empty batches
            if batch_data is None:
                continue

            # Forward pass
            predictions, adjacency = model(
                batch_data['timestamps'],
                batch_data['variables'],
                batch_data['values'],
                batch_data['mask']
            )

            # Collect predictions and targets (only observed)
            target_mask = batch_data['target_mask']
            all_predictions.append(predictions[target_mask > 0].cpu())
            all_targets.append(batch_data['targets'][target_mask > 0].cpu())

            if return_adjacency:
                # Average across batch dimension first (handles variable batch sizes)
                batch_avg_adj = adjacency.mean(dim=0)  # (36, 36)
                all_adjacencies.append(batch_avg_adj.cpu())

        except Exception as e:
            continue

    if len(all_predictions) == 0:
        return {}

    # Concatenate all
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Compute metrics
    metrics = compute_all_metrics(all_predictions, all_targets, prefix='')

    if return_adjacency and len(all_adjacencies) > 0:
        # Average adjacency across all batches
        avg_adjacency = torch.stack(all_adjacencies).mean(dim=0)  # (36, 36)
        metrics['adjacency'] = avg_adjacency

    return metrics


def main():
    # Configuration
    config = {
        'n_variables': 36,  # Time-series variables (excluding 5 static)
        'd_model': 128,
        'n_layers': 3,
        'n_heads': 4,
        'top_k': 12,
        'dropout': 0.1,
        'batch_size': 16,
        'lr': 1e-4,
        'epochs': 50,
        'patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    }

    print("=" * 80)
    print("TRAINING LEARNABLE GRAPH MODEL")
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
    model = LearnableGraphModel(
        n_variables=config['n_variables'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        top_k=config['top_k'],
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
        val_metrics = evaluate(model, val_loader, device, config['n_variables'], return_adjacency=(epoch % 10 == 0))
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
            if 'adjacency' in val_metrics:
                checkpoint['adjacency'] = val_metrics['adjacency']

            torch.save(checkpoint, 'results/learnable_graphs_best.pt')
            print(f"  ✓ Saved best model (RMSE: {val_rmse:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config['patience']}")

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save final results
    with open('results/learnable_graphs_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation RMSE: {best_val_loss:.4f}")
    print(f"Results saved to: results/learnable_graphs_results.json")
    print(f"Best model saved to: results/learnable_graphs_best.pt")


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    main()
