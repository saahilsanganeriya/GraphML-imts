"""
PyTorch Dataset Classes for PhysioNet IMTS Data

Author: Saahil Sanganeria
Date: October 2025
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class PhysioNetDataset(Dataset):
    """
    PyTorch Dataset for PhysioNet Challenge 2012.
    
    Returns:
        record_id: Patient record identifier
        timestamps: Tensor of observation times (normalized to [0, 1])
        values: Tensor of observed values (normalized)
        mask: Binary mask indicating observed values
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        history_window: float = 24.0,  # hours (in normalized time)
        forecast_window: float = 6.0,  # hours (in normalized time)
        max_seq_len: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to processed data directory
            split: One of 'train', 'val', 'test'
            history_window: Hours of historical data to use (in original scale)
            forecast_window: Hours to forecast ahead (in original scale)
            max_seq_len: Maximum sequence length (for padding/truncation)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.history_window = history_window
        self.forecast_window = forecast_window
        self.max_seq_len = max_seq_len
        
        # Load data
        split_file = self.data_path / f'{split}.pt'
        if not split_file.exists():
            # Try with quantization suffix
            split_file = self.data_path / f'{split}_0.0.pt'
        
        if not split_file.exists():
            raise FileNotFoundError(f"Could not find data file: {split_file}")
        
        self.data = torch.load(split_file)
        
        # Load statistics
        stats_file = self.data_path / 'statistics.pt'
        if not stats_file.exists():
            stats_file = self.data_path / 'statistics_0.0.pt'
        
        if stats_file.exists():
            self.stats = torch.load(stats_file)
            self.time_max = self.stats['time_max']
        else:
            self.time_max = 48.0  # Default
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record_id, timestamps, values, mask = self.data[idx]
        
        # Convert to tensors if needed
        if not isinstance(timestamps, torch.Tensor):
            timestamps = torch.tensor(timestamps, dtype=torch.float32)
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        
        return {
            'record_id': record_id,
            'timestamps': timestamps,
            'values': values,
            'mask': mask
        }
    
    def get_sample_for_forecasting(self, idx):
        """
        Get a sample split into history and forecast windows.
        
        Returns:
            Dictionary with:
                - history_timestamps, history_values, history_mask
                - forecast_timestamps, forecast_values, forecast_mask
        """
        sample = self.__getitem__(idx)
        
        # Convert history window to normalized time
        # Assuming time is already normalized to [0, 1] where 1 = time_max hours
        history_threshold = self.history_window / self.time_max
        forecast_end = history_threshold + (self.forecast_window / self.time_max)
        
        timestamps = sample['timestamps']
        values = sample['values']
        mask = sample['mask']
        
        # Split by time
        history_indices = timestamps <= history_threshold
        forecast_indices = (timestamps > history_threshold) & (timestamps <= forecast_end)
        
        return {
            'record_id': sample['record_id'],
            'history_timestamps': timestamps[history_indices],
            'history_values': values[history_indices],
            'history_mask': mask[history_indices],
            'forecast_timestamps': timestamps[forecast_indices],
            'forecast_values': values[forecast_indices],
            'forecast_mask': mask[forecast_indices]
        }


class PhysioNetGraphDataset(Dataset):
    """
    PyTorch Dataset for PhysioNet with graph structure.
    
    This version constructs graphs where:
    - Each observation is a node
    - Edges connect temporally close observations or co-occurring variables
    
    Used for graph-based models like RainDrop, Hi-Patch, etc.
    
    Can use pre-generated graphs (faster) or build dynamically.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        history_window: float = 24.0,
        forecast_window: float = 6.0,
        temporal_edge_threshold: float = 2.0,  # hours
        add_variable_edges: bool = True,
        add_self_loops: bool = True,
        use_pregenerated_graphs: bool = True  # Use pre-built graphs if available
    ):
        """
        Initialize graph dataset.
        
        Args:
            data_path: Path to processed data directory
            split: One of 'train', 'val', 'test'
            history_window: Hours of historical data to use
            forecast_window: Hours to forecast ahead
            temporal_edge_threshold: Max time difference for temporal edges (hours)
            add_variable_edges: Whether to add edges between different variables
            add_self_loops: Whether to add self-loop edges
        """
        self.data_path = Path(data_path)
        self.split = split
        self.history_window = history_window
        self.forecast_window = forecast_window
        self.temporal_edge_threshold = temporal_edge_threshold
        self.add_variable_edges = add_variable_edges
        self.add_self_loops = add_self_loops
        self.use_pregenerated_graphs = use_pregenerated_graphs
        
        # Load data
        split_file = self.data_path / f'{split}.pt'
        if not split_file.exists():
            split_file = self.data_path / f'{split}_0.0.pt'
        
        if not split_file.exists():
            raise FileNotFoundError(f"Could not find data file: {split_file}")
        
        self.data = torch.load(split_file)
        
        # Load pre-generated graphs if available and requested
        self.graphs = None
        if use_pregenerated_graphs:
            graphs_file = self.data_path / f'{split}_graphs.pt'
            if not graphs_file.exists():
                graphs_file = self.data_path / f'{split}_graphs_0.0.pt'
            
            if graphs_file.exists():
                self.graphs = torch.load(graphs_file)
                print(f"Loaded {len(self.graphs)} pre-generated graphs for {split} split")
            else:
                print(f"Warning: Pre-generated graphs not found, will build dynamically")
                self.use_pregenerated_graphs = False
        
        # Load statistics
        stats_file = self.data_path / 'statistics.pt'
        if not stats_file.exists():
            stats_file = self.data_path / 'statistics_0.0.pt'
        
        if stats_file.exists():
            self.stats = torch.load(stats_file)
            self.time_max = self.stats['time_max']
            self.n_variables = len(self.stats['params'])
        else:
            self.time_max = 48.0
            self.n_variables = 41
    
    def __len__(self):
        return len(self.data)
    
    def _construct_graph(
        self,
        timestamps: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct graph from observations (VECTORIZED).
        
        Args:
            timestamps: (T,) tensor of times
            values: (T, D) tensor of values
            mask: (T, D) tensor of observation indicators
        
        Returns:
            node_features: (N, feature_dim) node feature matrix
            edge_index: (2, E) edge index
            edge_attr: (E, edge_dim) edge attributes
        """
        # Get indices of observed values (vectorized)
        obs_indices = torch.nonzero(mask, as_tuple=False)  # (N, 2)
        
        if len(obs_indices) == 0:
            return torch.zeros(0, 3), torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 1)
        
        # Extract time and variable indices
        time_idx = obs_indices[:, 0]  # (N,)
        var_idx = obs_indices[:, 1]   # (N,)
        
        # Build node features (vectorized)
        node_timestamps = timestamps[time_idx]
        node_values = values[time_idx, var_idx]
        node_features = torch.stack([node_timestamps, var_idx.float(), node_values], dim=1)  # (N, 3)
        
        N = len(node_features)
        
        # Temporal threshold in normalized time
        temporal_threshold = self.temporal_edge_threshold / self.time_max
        
        # Build edges (vectorized with broadcasting)
        time_diff = torch.abs(node_timestamps.unsqueeze(1) - node_timestamps.unsqueeze(0))  # (N, N)
        same_var = (var_idx.unsqueeze(1) == var_idx.unsqueeze(0))  # (N, N)
        same_time = (time_idx.unsqueeze(1) == time_idx.unsqueeze(0))  # (N, N)
        
        # Temporal edges: same variable, close in time, not same node
        temporal_mask = same_var & (time_diff <= temporal_threshold) & (time_diff > 0)
        
        # Variable edges: different variables, same timestamp
        if self.add_variable_edges:
            variable_mask = (~same_var) & same_time
            edge_mask = temporal_mask | variable_mask
        else:
            edge_mask = temporal_mask
        
        # Get edge indices (vectorized)
        edge_indices = torch.nonzero(edge_mask, as_tuple=False)  # (E, 2)
        
        if len(edge_indices) > 0:
            edge_index = edge_indices.t()  # (2, E)
            edge_attr = time_diff[edge_indices[:, 0], edge_indices[:, 1]].unsqueeze(1)  # (E, 1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 1)
        
        # Add self-loops (vectorized)
        if self.add_self_loops:
            self_loop_index = torch.arange(N, dtype=torch.long).repeat(2, 1)  # (2, N)
            self_loop_attr = torch.zeros(N, 1)
            edge_index = torch.cat([edge_index, self_loop_index], dim=1)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        return node_features, edge_index, edge_attr
    
    def __getitem__(self, idx):
        record_id, timestamps, values, mask = self.data[idx]
        
        # Convert to tensors if needed
        if not isinstance(timestamps, torch.Tensor):
            timestamps = torch.tensor(timestamps, dtype=torch.float32)
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        
        # Use pre-generated graph if available
        if self.use_pregenerated_graphs and self.graphs is not None:
            graph_data = self.graphs[idx]
            hist_node_feat = graph_data['node_features']
            hist_edge_index = graph_data['edge_index']
            hist_edge_attr = graph_data['edge_attr']
            
            # Still need to split for forecast targets
            history_threshold = self.history_window / self.time_max
            forecast_end = history_threshold + (self.forecast_window / self.time_max)
            forecast_indices = (timestamps > history_threshold) & (timestamps <= forecast_end)
        else:
            # Build graph dynamically
            history_threshold = self.history_window / self.time_max
            forecast_end = history_threshold + (self.forecast_window / self.time_max)
            
            history_indices = timestamps <= history_threshold
            forecast_indices = (timestamps > history_threshold) & (timestamps <= forecast_end)
            
            # Construct graph from history
            hist_node_feat, hist_edge_index, hist_edge_attr = self._construct_graph(
                timestamps[history_indices],
                values[history_indices],
                mask[history_indices]
            )
        
        return {
            'record_id': record_id,
            'node_features': hist_node_feat,
            'edge_index': hist_edge_index,
            'edge_attr': hist_edge_attr,
            'forecast_timestamps': timestamps[forecast_indices],
            'forecast_values': values[forecast_indices],
            'forecast_mask': mask[forecast_indices],
            'n_variables': self.n_variables
        }


def collate_graph_batch(batch):
    """
    Collate function for batching graph data.
    
    Args:
        batch: List of samples from PhysioNetGraphDataset
        
    Returns:
        Batched graph data
    """
    from torch_geometric.data import Data, Batch
    
    graph_list = []
    forecast_data = {
        'record_ids': [],
        'timestamps': [],
        'values': [],
        'masks': []
    }
    
    for sample in batch:
        # Create PyG Data object
        data = Data(
            x=sample['node_features'],
            edge_index=sample['edge_index'],
            edge_attr=sample['edge_attr']
        )
        graph_list.append(data)
        
        # Store forecast data
        forecast_data['record_ids'].append(sample['record_id'])
        forecast_data['timestamps'].append(sample['forecast_timestamps'])
        forecast_data['values'].append(sample['forecast_values'])
        forecast_data['masks'].append(sample['forecast_mask'])
    
    # Batch graphs
    batched_graph = Batch.from_data_list(graph_list)
    
    return {
        'graph': batched_graph,
        'forecast_data': forecast_data,
        'n_variables': batch[0]['n_variables']
    }


if __name__ == '__main__':
    # Test the dataset
    print("Testing PhysioNetDataset...")
    dataset = PhysioNetDataset('data/physionet/processed', split='train')
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Record ID: {sample['record_id']}")
    print(f"Timestamps shape: {sample['timestamps'].shape}")
    print(f"Values shape: {sample['values'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    
    print("\n\nTesting PhysioNetGraphDataset...")
    graph_dataset = PhysioNetGraphDataset('data/physionet/processed', split='train')
    print(f"Dataset size: {len(graph_dataset)}")
    
    graph_sample = graph_dataset[0]
    print(f"\nGraph sample keys: {graph_sample.keys()}")
    print(f"Node features shape: {graph_sample['node_features'].shape}")
    print(f"Edge index shape: {graph_sample['edge_index'].shape}")
    print(f"Edge attr shape: {graph_sample['edge_attr'].shape}")

