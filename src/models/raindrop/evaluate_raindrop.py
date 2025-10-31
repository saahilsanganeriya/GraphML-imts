#!/usr/bin/env python3
"""
Evaluate RainDrop on Test Set

Loads trained model and evaluates on test set with detailed metrics and visualizations.

Usage:
    python src/models/raindrop/evaluate_raindrop.py --checkpoint results/raindrop/best_model.pt
"""

import sys
import os
from pathlib import Path
import argparse
import json

# MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add paths
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / 'src'))
sys.path.insert(0, str(repo_root / 'src' / 'models' / 'raindrop' / 'Raindrop' / 'code'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models_rd import Raindrop
from utils.metrics import compute_all_metrics
from train_raindrop_forecasting import (
    load_data, convert_to_raindrop_format, compute_loss_and_metrics,
    get_device, plot_predictions_vs_actual
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate RainDrop on Test Set')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/physionet/processed',
                        help='Path to preprocessed data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: same as checkpoint dir)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use')
    
    return parser.parse_args()


def evaluate_with_details(model, data, batch_size, device):
    """
    Evaluate model with detailed per-variable and per-patient metrics.
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    per_patient_metrics = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            batch_patients = data[i:i + batch_size]
            
            # Convert to RainDrop format
            src, times, static, lengths, targets, target_masks, target_times = \
                convert_to_raindrop_format(batch_patients, device=device)
            
            # Forward pass
            predictions, _, _ = model(src, static, times, lengths)
            # predictions: (B, 36, n_forecast_steps)
            
            # Get predictions (average over forecast steps)
            pred_mean = predictions.mean(dim=-1).cpu()  # (B, 36)
            
            # Process each patient
            for j, (patient_targets, patient_mask) in enumerate(zip(targets, target_masks)):
                # Average target over time per variable
                sum_targets = (patient_targets * patient_mask).sum(dim=0)  # (36,)
                count_obs = patient_mask.sum(dim=0)  # (36,)
                valid_vars = count_obs > 0
                
                if valid_vars.sum() > 0:
                    avg_target = sum_targets[valid_vars] / count_obs[valid_vars]
                    pred_for_patient = pred_mean[j][valid_vars]
                    
                    # Compute metrics for this patient
                    patient_metrics = compute_all_metrics(
                        pred_for_patient,
                        avg_target,
                        prefix=''
                    )
                    per_patient_metrics.append(patient_metrics)
                    
                    # Collect all predictions
                    all_preds.append(pred_for_patient)
                    all_targets.append(avg_target)
    
    # Concatenate all
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Overall metrics
    overall_metrics = compute_all_metrics(all_preds, all_targets, prefix='test_')
    
    return overall_metrics, per_patient_metrics, all_preds, all_targets


def plot_metric_distribution(per_patient_metrics, metric_name, output_dir):
    """Plot distribution of a metric across patients."""
    values = [m[metric_name] for m in per_patient_metrics if not np.isnan(m[metric_name])]
    
    if len(values) == 0:
        return None
    
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(values, bins=50, alpha=0.7, edgecolor='black')
    
    # Add statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    plt.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    
    plt.xlabel(metric_name.upper(), fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title(f'Distribution of {metric_name.upper()} Across Patients', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = Path(output_dir) / f'{metric_name}_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def plot_error_analysis(predictions, targets, output_dir):
    """Plot error analysis: residuals and error distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Compute errors
    errors = (predictions - targets).numpy()
    
    # Residual plot
    ax1.scatter(targets.numpy(), errors, alpha=0.3, s=10)
    ax1.axhline(0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Actual Values', fontsize=12)
    ax1.set_ylabel('Residuals (Predicted - Actual)', fontsize=12)
    ax1.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(np.mean(errors), color='g', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors):.4f}')
    ax2.set_xlabel('Error (Predicted - Actual)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'error_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def save_results_summary(overall_metrics, per_patient_metrics, output_dir):
    """Save detailed results summary."""
    # Compute per-patient statistics
    per_patient_stats = {}
    for metric_name in ['rmse', 'mae', 'r2']:
        values = [m[metric_name] for m in per_patient_metrics if not np.isnan(m[metric_name])]
        if len(values) > 0:
            per_patient_stats[metric_name] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
    
    # Create summary
    summary = {
        'overall_metrics': {k: float(v) if not np.isnan(v) else None 
                           for k, v in overall_metrics.items()},
        'per_patient_statistics': per_patient_stats,
        'n_patients_evaluated': len(per_patient_metrics)
    }
    
    # Save JSON
    json_path = Path(output_dir) / 'test_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save text report
    txt_path = Path(output_dir) / 'test_results.txt'
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RainDrop Test Set Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-" * 80 + "\n")
        for metric_name, value in overall_metrics.items():
            if not np.isnan(value):
                f.write(f"{metric_name:25s}: {value:10.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-Patient Statistics:\n")
        f.write("=" * 80 + "\n\n")
        
        for metric_name, stats in per_patient_stats.items():
            f.write(f"{metric_name.upper()}:\n")
            f.write(f"   Mean   : {stats['mean']:.4f}\n")
            f.write(f"   Median : {stats['median']:.4f}\n")
            f.write(f"   Std    : {stats['std']:.4f}\n")
            f.write(f"   Min    : {stats['min']:.4f}\n")
            f.write(f"   Max    : {stats['max']:.4f}\n")
            f.write(f"   Q25    : {stats['q25']:.4f}\n")
            f.write(f"   Q75    : {stats['q75']:.4f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Total patients evaluated: {len(per_patient_metrics)}\n")
        f.write("=" * 80 + "\n")
    
    return str(json_path), str(txt_path)


def main():
    args = parse_args()
    
    print("=" * 80)
    print("RainDrop Test Set Evaluation")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get training args
    train_args = checkpoint.get('args', {})
    
    # Device
    device, device_name = get_device(args.device)
    print(f"Device: {device_name}")
    
    # Output directory
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent / 'test_results'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_data(args.data_dir, 'test')
    print(f"   Test: {len(test_data)} patients")
    
    # Create model
    print("\nCreating model...")
    global_structure = torch.ones(36, 36)
    
    model = Raindrop(
        d_inp=36,
        d_model=train_args.get('d_model', 64),
        nhead=train_args.get('nhead', 4),
        nhid=train_args.get('d_model', 64) * 2,
        nlayers=train_args.get('nlayers', 2),
        dropout=train_args.get('dropout', 0.3),
        max_len=215,
        d_static=9,
        n_classes=2,
        forecasting_mode=True,
        n_forecast_steps=6,
        global_structure=global_structure
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("   ✓ Model loaded")
    
    # Evaluate
    print("\nEvaluating on test set...")
    overall_metrics, per_patient_metrics, all_preds, all_targets = evaluate_with_details(
        model, test_data, args.batch_size, device
    )
    
    # Print overall metrics
    print("\n" + "=" * 80)
    print("Overall Test Metrics")
    print("=" * 80)
    for metric_name, value in overall_metrics.items():
        if not np.isnan(value):
            print(f"{metric_name:25s}: {value:10.4f}")
    print("=" * 80)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Predictions vs actual
    plot_predictions_vs_actual(all_preds, all_targets, output_dir)
    
    # Error analysis
    plot_error_analysis(all_preds, all_targets, output_dir)
    
    # Per-patient metric distributions
    for metric in ['rmse', 'mae', 'r2']:
        plot_metric_distribution(per_patient_metrics, metric, output_dir)
    
    # Save results
    print("\nSaving results...")
    json_path, txt_path = save_results_summary(overall_metrics, per_patient_metrics, output_dir)
    print(f"   ✓ Saved JSON: {json_path}")
    print(f"   ✓ Saved TXT: {txt_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()

