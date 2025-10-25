"""
Evaluation Metrics for Time Series Forecasting

Author: Saahil Sanganeria
Date: October 2025
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Binary mask for valid predictions (1 = valid, 0 = missing)
    
    Returns:
        RMSE value
    """
    if mask is not None:
        predictions = predictions[mask == 1]
        targets = targets[mask == 1]
    
    if len(predictions) == 0:
        return float('nan')
    
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse).item()


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Binary mask for valid predictions (1 = valid, 0 = missing)
    
    Returns:
        MAE value
    """
    if mask is not None:
        predictions = predictions[mask == 1]
        targets = targets[mask == 1]
    
    if len(predictions) == 0:
        return float('nan')
    
    mae = torch.mean(torch.abs(predictions - targets))
    return mae.item()


def compute_mape(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, epsilon: float = 1e-8) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Binary mask for valid predictions
        epsilon: Small value to avoid division by zero
    
    Returns:
        MAPE value (as percentage)
    """
    if mask is not None:
        predictions = predictions[mask == 1]
        targets = targets[mask == 1]
    
    if len(predictions) == 0:
        return float('nan')
    
    mape = torch.mean(torch.abs((targets - predictions) / (torch.abs(targets) + epsilon))) * 100
    return mape.item()


def compute_smape(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Binary mask for valid predictions
    
    Returns:
        SMAPE value (as percentage)
    """
    if mask is not None:
        predictions = predictions[mask == 1]
        targets = targets[mask == 1]
    
    if len(predictions) == 0:
        return float('nan')
    
    numerator = torch.abs(predictions - targets)
    denominator = (torch.abs(predictions) + torch.abs(targets)) / 2
    smape = torch.mean(numerator / (denominator + 1e-8)) * 100
    return smape.item()


def compute_r2_score(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute R² (coefficient of determination) score.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Binary mask for valid predictions
    
    Returns:
        R² score
    """
    if mask is not None:
        predictions = predictions[mask == 1]
        targets = targets[mask == 1]
    
    if len(predictions) == 0:
        return float('nan')
    
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2.item()


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Binary mask for valid predictions
        prefix: Prefix for metric names (e.g., "train_", "val_")
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        f'{prefix}rmse': compute_rmse(predictions, targets, mask),
        f'{prefix}mae': compute_mae(predictions, targets, mask),
        f'{prefix}mape': compute_mape(predictions, targets, mask),
        f'{prefix}smape': compute_smape(predictions, targets, mask),
        f'{prefix}r2': compute_r2_score(predictions, targets, mask)
    }
    
    # Add RMSE/MAE ratio
    if metrics[f'{prefix}mae'] > 0:
        metrics[f'{prefix}rmse_mae_ratio'] = metrics[f'{prefix}rmse'] / metrics[f'{prefix}mae']
    else:
        metrics[f'{prefix}rmse_mae_ratio'] = float('nan')
    
    return metrics


def compute_per_variable_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    variable_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each variable separately.
    
    Args:
        predictions: (batch_size, n_variables, horizon) predicted values
        targets: (batch_size, n_variables, horizon) ground truth values
        mask: (batch_size, n_variables, horizon) binary mask
        variable_names: List of variable names
    
    Returns:
        Dictionary mapping variable names to their metrics
    """
    n_variables = predictions.shape[1]
    per_var_metrics = {}
    
    for i, var_name in enumerate(variable_names[:n_variables]):
        var_pred = predictions[:, i, :].reshape(-1)
        var_target = targets[:, i, :].reshape(-1)
        var_mask = mask[:, i, :].reshape(-1)
        
        metrics = compute_all_metrics(var_pred, var_target, var_mask)
        per_var_metrics[var_name] = metrics
    
    return per_var_metrics


def compute_per_horizon_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    horizon_steps: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each forecast horizon separately.
    
    Args:
        predictions: (batch_size, n_variables, horizon) predicted values
        targets: (batch_size, n_variables, horizon) ground truth values
        mask: (batch_size, n_variables, horizon) binary mask
        horizon_steps: List of horizon step names (e.g., ["1h", "2h", ...])
    
    Returns:
        Dictionary mapping horizon steps to their metrics
    """
    horizon = predictions.shape[2]
    per_horizon_metrics = {}
    
    for h, step_name in enumerate(horizon_steps[:horizon]):
        h_pred = predictions[:, :, h].reshape(-1)
        h_target = targets[:, :, h].reshape(-1)
        h_mask = mask[:, :, h].reshape(-1)
        
        metrics = compute_all_metrics(h_pred, h_target, h_mask)
        per_horizon_metrics[step_name] = metrics
    
    return per_horizon_metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics table
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    for metric_name, value in metrics.items():
        if not np.isnan(value):
            print(f"{metric_name:20s}: {value:10.4f}")
        else:
            print(f"{metric_name:20s}: {'NaN':10s}")
    
    print("=" * 60)


if __name__ == '__main__':
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Create dummy data
    predictions = torch.randn(10, 5, 6)  # batch=10, vars=5, horizon=6
    targets = torch.randn(10, 5, 6)
    mask = torch.rand(10, 5, 6) > 0.2  # 80% observed
    
    # Overall metrics
    metrics = compute_all_metrics(predictions.reshape(-1), targets.reshape(-1), mask.reshape(-1), prefix="test_")
    print_metrics(metrics, "Overall Metrics")
    
    # Per-variable metrics
    variable_names = ['HR', 'Temp', 'SysABP', 'Glucose', 'pH']
    per_var = compute_per_variable_metrics(predictions, targets, mask, variable_names)
    print("\nPer-variable metrics:")
    for var_name, var_metrics in per_var.items():
        print(f"\n{var_name}:")
        for k, v in var_metrics.items():
            if not np.isnan(v):
                print(f"  {k}: {v:.4f}")
    
    print("\n✓ Metrics test passed!")

