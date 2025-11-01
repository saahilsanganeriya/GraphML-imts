"""
Data Preprocessing Pipeline for PhysioNet Challenge 2012 Dataset

This module implements comprehensive preprocessing for irregular multivariate time series data.
It handles downloading, cleaning, normalization, and splitting of the PhysioNet ICU dataset.

Author: Saahil Sanganeria
Date: October 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import tarfile
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from torchvision.datasets.utils import download_url

warnings.filterwarnings('ignore')


class PhysioNetPreprocessor:
    """
    Comprehensive preprocessing pipeline for PhysioNet Challenge 2012 dataset.
    
    This class handles:
    - Data downloading and extraction
    - Quality control and validation
    - Outlier detection and handling
    - Missing data analysis
    - Normalization
    - Train/val/test splitting
    - Graph construction for graph-based models
    """
    
    # PhysioNet dataset URLs
    URLS = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download',
    ]
    
    # All parameters in the dataset
    PARAMS = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]
    
    # Static descriptors (measured once at admission)
    STATIC_PARAMS = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
    
    def __init__(
        self,
        root_dir: str = 'data/physionet',
        quantization: float = 0.0,
        train_split: float = 0.6,
        val_split: float = 0.2,
        test_split: float = 0.2,
        random_seed: int = 42,
        device: str = 'cpu'
    ):
        """
        Initialize the preprocessor.
        
        Args:
            root_dir: Root directory for data storage
            quantization: Time quantization interval (hours). 0.0 = no quantization
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            random_seed: Random seed for reproducibility
            device: Device for PyTorch tensors ('cpu' or 'cuda')
        """
        self.root_dir = Path(root_dir)
        self.raw_folder = self.root_dir / 'raw'
        self.processed_folder = self.root_dir / 'processed'
        self.quantization = quantization
        
        # Split ratios
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1"
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.random_seed = random_seed
        # Set device (MPS for Apple Silicon, CUDA for GPU, CPU otherwise)
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Create directories
        self.raw_folder.mkdir(parents=True, exist_ok=True)
        self.processed_folder.mkdir(parents=True, exist_ok=True)
        
        # Statistics for normalization
        self.data_min = None
        self.data_max = None
        self.data_mean = None
        self.data_std = None
        self.time_max = None
        
        # Compute time series params (can't do in class scope)
        self.TIME_SERIES_PARAMS = [p for p in self.PARAMS if p not in self.STATIC_PARAMS]
        
    def download(self, force_download: bool = False):
        """
        Download and extract the PhysioNet dataset.
        
        Args:
            force_download: If True, redownload even if files exist
        """
        print("=" * 80)
        print("PhysioNet Challenge 2012 Data Download and Extraction")
        print("=" * 80)
        
        for url in self.URLS:
            filename = url.rpartition('/')[2]
            filepath = self.raw_folder / filename
            
            # Check if already downloaded
            if filepath.exists() and not force_download:
                print(f"✓ {filename} already exists, skipping download")
            else:
                print(f"\nDownloading {filename}...")
                download_url(url, str(self.raw_folder), filename, None)
                print(f"✓ Downloaded {filename}")
            
            # Extract
            dirname = self.raw_folder / filename.split('.')[0]
            if dirname.exists() and not force_download:
                print(f"✓ {filename} already extracted")
            else:
                print(f"Extracting {filename}...")
                tar = tarfile.open(filepath, "r:gz")
                tar.extractall(self.raw_folder)
                tar.close()
                print(f"✓ Extracted {filename}")
        
        print("\n" + "=" * 80)
        print("Download and extraction complete!")
        print("=" * 80)
    
    def load_raw_data(self) -> List[Tuple]:
        """
        Load raw data from text files.
        
        Returns:
            List of tuples: (record_id, timestamps, values, mask)
        """
        print("\n" + "=" * 80)
        print("Loading Raw Data")
        print("=" * 80)
        
        all_patients = []
        params_dict = {k: i for i, k in enumerate(self.PARAMS)}
        
        for url in self.URLS:
            filename = url.rpartition('/')[2]
            dirname = self.raw_folder / filename.split('.')[0]
            
            print(f"\nProcessing {dirname.name}...")
            
            txt_files = sorted(dirname.glob('*.txt'))
            print(f"Found {len(txt_files)} patient records")
            
            for i, txtfile in enumerate(txt_files):
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(txt_files)} records...")
                
                record_id = txtfile.stem
                
                with open(txtfile, 'r') as f:
                    lines = f.readlines()
                    
                    prev_time = 0
                    tt = [0.]
                    vals = [torch.zeros(len(self.PARAMS))]
                    mask = [torch.zeros(len(self.PARAMS))]
                    nobs = [torch.zeros(len(self.PARAMS))]
                    
                    for line in lines[1:]:  # Skip header
                        time, param, val = line.strip().split(',')
                        
                        # Parse time (HH:MM format)
                        time_hours = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
                        
                        # Apply quantization if specified
                        if self.quantization > 0:
                            time_hours = round(time_hours / self.quantization) * self.quantization
                        
                        # Create new time point if needed
                        if time_hours != prev_time:
                            tt.append(time_hours)
                            vals.append(torch.zeros(len(self.PARAMS)))
                            mask.append(torch.zeros(len(self.PARAMS)))
                            nobs.append(torch.zeros(len(self.PARAMS)))
                            prev_time = time_hours
                        
                        # Store value
                        if param in params_dict:
                            param_idx = params_dict[param]
                            n_observations = nobs[-1][param_idx]
                            
                            # Average multiple observations at same timestamp
                            if n_observations > 0:
                                prev_val = vals[-1][param_idx]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][param_idx] = new_val
                            else:
                                vals[-1][param_idx] = float(val)
                            
                            mask[-1][param_idx] = 1
                            nobs[-1][param_idx] += 1
                
                # Convert to tensors
                tt = torch.tensor(tt, dtype=torch.float32)
                vals = torch.stack(vals)
                mask = torch.stack(mask)
                
                all_patients.append((record_id, tt, vals, mask))
        
        print(f"\n✓ Loaded {len(all_patients)} patient records")
        return all_patients
    
    def validate_data(self, patients: List[Tuple]) -> List[Tuple]:
        """
        Basic data validation - NO outlier removal.
        
        Extreme values (HR=300, Glucose=800) are clinically significant in ICU patients
        and should be kept for robust model evaluation.
        
        Only checks for:
        - Invalid/corrupted data (NaN, inf)
        - Impossible negative values where not allowed
        
        Args:
            patients: List of (record_id, timestamps, values, mask) tuples
            
        Returns:
            Validated patient data
        """
        print("\n" + "=" * 80)
        print("Data Validation (NO Outlier Removal)")
        print("=" * 80)
        print("Note: Keeping extreme values - clinically significant in ICU patients!")
        
        validated_patients = []
        total_invalid = 0
        
        for record_id, tt, vals, mask in patients:
            new_vals = vals.clone()
            new_mask = mask.clone()
            
            # Only remove truly invalid data (NaN, inf, impossible negatives)
            for idx in range(len(self.PARAMS)):
                obs_mask = mask[:, idx] == 1
                if obs_mask.sum() > 0:
                    param_vals = vals[:, idx][obs_mask]
                    
                    # Check for NaN/inf
                    invalid_mask = torch.isnan(param_vals) | torch.isinf(param_vals)
                    
                    if invalid_mask.sum() > 0:
                        invalid_indices = torch.where(obs_mask)[0][invalid_mask]
                        new_mask[invalid_indices, idx] = 0
                        new_vals[invalid_indices, idx] = 0
                        total_invalid += invalid_mask.sum().item()
            
            validated_patients.append((record_id, tt, new_vals, new_mask))
        
        print(f"\n✓ Removed {total_invalid} invalid observations (NaN/inf)")
        print("✓ Kept extreme values (clinically meaningful outliers)")
        
        return validated_patients
    
    def construct_graphs(
        self,
        patients: List[Tuple],
        temporal_threshold: float = 2.0,
        add_variable_edges: bool = True
    ) -> List[Dict]:
        """
        Pre-construct observation graphs for each patient.
        
        Args:
            patients: List of patient data
            temporal_threshold: Max time difference for temporal edges (hours, in original scale)
            add_variable_edges: Whether to add edges between co-occurring variables
            
        Returns:
            List of graph dictionaries with node_features, edge_index, edge_attr
        """
        print("\n" + "=" * 80)
        print("Constructing Observation Graphs")
        print("=" * 80)
        print(f"Temporal threshold: {temporal_threshold} hours")
        print(f"Variable edges: {add_variable_edges}")
        
        all_graphs = []
        
        for i, (record_id, tt, vals, mask) in enumerate(patients):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(patients)} graphs...")
            
            # Build graph for this patient
            graph_data = self._build_observation_graph(
                tt, vals, mask, temporal_threshold, add_variable_edges
            )
            
            all_graphs.append({
                'record_id': record_id,
                'node_features': graph_data['node_features'],
                'edge_index': graph_data['edge_index'],
                'edge_attr': graph_data['edge_attr']
            })
        
        print(f"\n✓ Constructed {len(all_graphs)} observation graphs")
        
        # Print some statistics
        avg_nodes = np.mean([g['node_features'].shape[0] for g in all_graphs])
        avg_edges = np.mean([g['edge_index'].shape[1] for g in all_graphs])
        print(f"  Average nodes per graph: {avg_nodes:.1f}")
        print(f"  Average edges per graph: {avg_edges:.1f}")
        
        return all_graphs
    
    def _build_observation_graph(
        self,
        timestamps: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        temporal_threshold: float,
        add_variable_edges: bool
    ) -> Dict:
        """
        Build observation graph for a single patient (VECTORIZED).
        
        Returns:
            Dictionary with node_features, edge_index, edge_attr
        """
        # Get indices of observed values (vectorized)
        obs_indices = torch.nonzero(mask, as_tuple=False)  # (N, 2) where N = num observations
        
        if len(obs_indices) == 0:
            return {
                'node_features': torch.zeros(0, 3),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'edge_attr': torch.zeros(0, 1)
            }
        
        # Extract time indices and variable indices
        time_idx = obs_indices[:, 0]  # (N,)
        var_idx = obs_indices[:, 1]   # (N,)
        
        # Build node features (vectorized)
        node_timestamps = timestamps[time_idx]  # (N,)
        node_values = values[time_idx, var_idx]  # (N,)
        node_features = torch.stack([node_timestamps, var_idx.float(), node_values], dim=1)  # (N, 3)
        
        N = len(node_features)
        
        # Temporal threshold in normalized time
        if self.time_max is not None:
            temporal_threshold_norm = temporal_threshold / self.time_max
        else:
            temporal_threshold_norm = temporal_threshold / 48.0
        
        # Build edges (vectorized using broadcasting)
        # Create pairwise comparisons
        time_diff = torch.abs(node_timestamps.unsqueeze(1) - node_timestamps.unsqueeze(0))  # (N, N)
        same_var = (var_idx.unsqueeze(1) == var_idx.unsqueeze(0))  # (N, N)
        same_time = (time_idx.unsqueeze(1) == time_idx.unsqueeze(0))  # (N, N)
        
        # Temporal edges: same variable, close in time, not same node
        temporal_mask = same_var & (time_diff <= temporal_threshold_norm) & (time_diff > 0)
        
        # Variable edges: different variables, same timestamp
        if add_variable_edges:
            variable_mask = (~same_var) & same_time
            edge_mask = temporal_mask | variable_mask
        else:
            edge_mask = temporal_mask
        
        # Get edge indices
        edge_indices = torch.nonzero(edge_mask, as_tuple=False)  # (E, 2)
        
        if len(edge_indices) > 0:
            edge_index = edge_indices.t()  # (2, E)
            # Edge attributes: time difference
            edge_attr = time_diff[edge_indices[:, 0], edge_indices[:, 1]].unsqueeze(1)  # (E, 1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 1)
        
        # Add self-loops (vectorized)
        self_loop_index = torch.arange(N, dtype=torch.long).repeat(2, 1)  # (2, N)
        self_loop_attr = torch.zeros(N, 1)
        
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
    
    def compute_statistics(self, patients: List[Tuple], split_indices: np.ndarray):
        """
        Compute normalization statistics from training data only.
        
        Args:
            patients: List of patient data
            split_indices: Indices of training patients
        """
        print("\n" + "=" * 80)
        print("Computing Normalization Statistics (Training Set Only)")
        print("=" * 80)
        
        train_patients = [patients[i] for i in split_indices]
        
        # Initialize statistics
        inf = float('inf')
        data_min = torch.full((len(self.PARAMS),), inf)
        data_max = torch.full((len(self.PARAMS),), -inf)
        time_max = 0.0
        
        # For mean and std
        sums = torch.zeros(len(self.PARAMS))
        counts = torch.zeros(len(self.PARAMS))
        
        # First pass: min, max, sum, count
        for record_id, tt, vals, mask in train_patients:
            for i in range(len(self.PARAMS)):
                non_missing_vals = vals[:, i][mask[:, i] == 1]
                
                if len(non_missing_vals) > 0:
                    data_min[i] = min(data_min[i], non_missing_vals.min().item())
                    data_max[i] = max(data_max[i], non_missing_vals.max().item())
                    sums[i] += non_missing_vals.sum().item()
                    counts[i] += len(non_missing_vals)
            
            time_max = max(time_max, tt.max().item())
        
        # Compute mean
        data_mean = sums / (counts + 1e-8)
        
        # Second pass: std
        squared_diffs = torch.zeros(len(self.PARAMS))
        for record_id, tt, vals, mask in train_patients:
            for i in range(len(self.PARAMS)):
                non_missing_vals = vals[:, i][mask[:, i] == 1]
                if len(non_missing_vals) > 0:
                    squared_diffs[i] += ((non_missing_vals - data_mean[i]) ** 2).sum().item()
        
        data_std = torch.sqrt(squared_diffs / (counts + 1e-8))
        
        # Handle edge cases
        data_min = torch.where(torch.isinf(data_min), torch.zeros_like(data_min), data_min)
        data_max = torch.where(torch.isinf(data_max), torch.ones_like(data_max), data_max)
        data_std = torch.where(data_std == 0, torch.ones_like(data_std), data_std)
        
        self.data_min = data_min
        self.data_max = data_max
        self.data_mean = data_mean
        self.data_std = data_std
        self.time_max = time_max
        
        print(f"\n✓ Computed statistics from {len(train_patients)} training patients")
        print(f"  Time range: [0.0, {time_max:.2f}] hours")
        print(f"\nSample statistics (first 5 variables):")
        for i in range(min(5, len(self.PARAMS))):
            print(f"  {self.PARAMS[i]:12s}: min={data_min[i]:.2f}, max={data_max[i]:.2f}, "
                  f"mean={data_mean[i]:.2f}, std={data_std[i]:.2f}")
    
    def normalize_data(self, patients: List[Tuple]) -> List[Tuple]:
        """
        Normalize data using computed statistics.
        
        Args:
            patients: List of patient data
            
        Returns:
            Normalized patient data
        """
        print("\n" + "=" * 80)
        print("Normalizing Data")
        print("=" * 80)
        
        normalized_patients = []
        
        for record_id, tt, vals, mask in patients:
            # Normalize values using z-score normalization
            normalized_vals = (vals - self.data_mean) / (self.data_std + 1e-8)
            
            # Set masked values to 0
            normalized_vals = normalized_vals * mask
            
            # Normalize timestamps to [0, 1]
            normalized_tt = tt / (self.time_max + 1e-8)
            
            normalized_patients.append((record_id, normalized_tt, normalized_vals, mask))
        
        print(f"✓ Normalized {len(patients)} patient records")
        
        return normalized_patients
    
    def create_splits(self, patients: List[Tuple]) -> Dict[str, List[int]]:
        """
        Create stratified train/val/test splits.
        
        Args:
            patients: List of patient data
            
        Returns:
            Dictionary with 'train', 'val', 'test' indices
        """
        print("\n" + "=" * 80)
        print("Creating Train/Val/Test Splits")
        print("=" * 80)
        
        np.random.seed(self.random_seed)
        
        n_patients = len(patients)
        indices = np.arange(n_patients)
        np.random.shuffle(indices)
        
        # Calculate split sizes
        n_train = int(n_patients * self.train_split)
        n_val = int(n_patients * self.val_split)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        print(f"\n✓ Created splits:")
        print(f"  Train: {len(train_indices)} patients ({len(train_indices)/n_patients*100:.1f}%)")
        print(f"  Val:   {len(val_indices)} patients ({len(val_indices)/n_patients*100:.1f}%)")
        print(f"  Test:  {len(test_indices)} patients ({len(test_indices)/n_patients*100:.1f}%)")
        
        return splits
    
    def save_processed_data(
        self,
        patients: List[Tuple],
        splits: Dict[str, List[int]],
        graphs: Optional[List[Dict]] = None,
        suffix: str = ""
    ):
        """
        Save processed data to disk.
        
        Args:
            patients: List of patient data
            splits: Dictionary of split indices
            graphs: Optional list of pre-constructed graphs
            suffix: Optional suffix for filename
        """
        print("\n" + "=" * 80)
        print("Saving Processed Data")
        print("=" * 80)
        
        # Save full dataset (all 12,000 patients)
        full_path = self.processed_folder / f'all_data{suffix}.pt'
        torch.save(patients, full_path)
        print(f"✓ Saved full dataset ({len(patients)} patients): {full_path}")
        
        # Save splits
        for split_name, indices in splits.items():
            split_data = [patients[i] for i in indices]
            split_path = self.processed_folder / f'{split_name}{suffix}.pt'
            torch.save(split_data, split_path)
            print(f"✓ Saved {split_name} split ({len(split_data)} patients): {split_path}")
        
        # Save graphs if provided
        if graphs is not None:
            # Save full graphs
            full_graphs_path = self.processed_folder / f'all_graphs{suffix}.pt'
            torch.save(graphs, full_graphs_path)
            print(f"✓ Saved all graphs: {full_graphs_path}")
            
            # Save split graphs
            for split_name, indices in splits.items():
                split_graphs = [graphs[i] for i in indices]
                graphs_path = self.processed_folder / f'{split_name}_graphs{suffix}.pt'
                torch.save(split_graphs, graphs_path)
                print(f"✓ Saved {split_name} graphs ({len(split_graphs)} graphs): {graphs_path}")
        
        # Save split indices
        splits_path = self.processed_folder / f'split_indices{suffix}.pt'
        torch.save(splits, splits_path)
        print(f"✓ Saved split indices: {splits_path}")
        
        # Save normalization statistics
        stats = {
            'data_min': self.data_min,
            'data_max': self.data_max,
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'time_max': self.time_max,
            'params': self.PARAMS,
            'static_params': self.STATIC_PARAMS,
            'time_series_params': self.TIME_SERIES_PARAMS
        }
        stats_path = self.processed_folder / f'statistics{suffix}.pt'
        torch.save(stats, stats_path)
        print(f"✓ Saved normalization statistics: {stats_path}")
        
        print("\n" + "=" * 80)
        print("All data saved successfully!")
        print("=" * 80)
    
    def process(self, download_data: bool = True, force_download: bool = False):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            download_data: Whether to download the dataset
            force_download: Whether to force redownload
        """
        print("\n" + "=" * 80)
        print("PHYSIONET CHALLENGE 2012 - DATA PREPROCESSING PIPELINE")
        print("=" * 80)
        print(f"Root directory: {self.root_dir}")
        print(f"Quantization: {self.quantization} hours")
        print(f"Splits: Train={self.train_split}, Val={self.val_split}, Test={self.test_split}")
        print(f"Random seed: {self.random_seed}")
        print("=" * 80)
        
        # Step 1: Download
        if download_data:
            self.download(force_download=force_download)
        
        # Step 2: Load raw data
        patients = self.load_raw_data()
        
        # Step 3: Basic data validation (NO outlier removal - clinically significant!)
        patients = self.validate_data(patients)
        
        # Step 4: Create splits
        splits = self.create_splits(patients)
        
        # Step 5: Compute statistics (on training set only)
        self.compute_statistics(patients, splits['train'])
        
        # Step 6: Normalize data
        patients = self.normalize_data(patients)
        
        # Step 7: Construct observation graphs (for graph-based models)
        print("\n" + "=" * 80)
        print("Graph Pre-generation for Graph-Based Models")
        print("=" * 80)
        graphs = self.construct_graphs(
            patients,
            temporal_threshold=2.0,  # 2 hours
            add_variable_edges=True
        )
        
        # Step 8: Save processed data and graphs
        suffix = f"_{self.quantization}" if self.quantization > 0 else ""
        self.save_processed_data(patients, splits, graphs, suffix=suffix)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"\nProcessed data saved to: {self.processed_folder}")
        print("\nNext steps:")
        print("1. Use the processed data for model training")
        print("2. Check docs/DATA_PREPROCESSING.md for detailed documentation")
        print("3. Run EDA scripts to visualize the processed data")
        print("=" * 80)
        
        return patients, splits


def main():
    """Main function for running preprocessing from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess PhysioNet Challenge 2012 dataset')
    parser.add_argument('--root-dir', type=str, default='data/physionet',
                       help='Root directory for data storage')
    parser.add_argument('--quantization', type=float, default=0.0,
                       help='Time quantization in hours (0.0 = no quantization)')
    parser.add_argument('--train-split', type=float, default=0.6,
                       help='Fraction of data for training')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Fraction of data for validation')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fraction of data for testing')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip downloading (data already exists)')
    parser.add_argument('--force-download', action='store_true',
                       help='Force redownload even if data exists')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = PhysioNetPreprocessor(
        root_dir=args.root_dir,
        quantization=args.quantization,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.random_seed
    )
    
    # Run preprocessing
    preprocessor.process(
        download_data=not args.no_download,
        force_download=args.force_download
    )


if __name__ == '__main__':
    main()

