#!/usr/bin/env python3
"""
Enhanced Exploratory Data Analysis (EDA) for PhysioNet Challenge 2012 Dataset
This script performs comprehensive EDA with professional visualizations for publication.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch
import warnings
warnings.filterwarnings('ignore')

# Add the Hi-Patch library paths
sys.path.append('src/models/hipatch')
sys.path.append('src/models/hipatch/lib')

from lib.physionet import PhysioNet

def setup_environment():
    """Setup directories and plotting parameters for publication-quality figures"""
    os.makedirs('eda_results', exist_ok=True)
    
    # Set publication-ready matplotlib parameters
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times'
    
    # Color scheme for professional plots
    colors = sns.color_palette("Set2", 8)
    sns.set_palette(colors)

def create_advanced_visualizations(dataset):
    """Create publication-quality visualizations with statistical insights"""
    
    print("\n=== CREATING ENHANCED VISUALIZATIONS ===")
    
    # Sample a reasonable subset for analysis
    sample_size = min(1000, len(dataset))
    
    # 1. Missingness Heatmap
    create_missingness_heatmap(dataset, sample_size)
    
    # 2. Variable Distribution Analysis
    create_distribution_analysis(dataset, sample_size)
    
    # 3. Temporal Pattern Analysis
    create_temporal_pattern_analysis(dataset, sample_size)
    
    # 4. Correlation Analysis
    create_correlation_analysis(dataset, sample_size)
    
    # 5. ICU Type Analysis
    create_icu_type_analysis(dataset, sample_size)
    
    # 6. Multi-scale Irregularity Analysis
    irregularity_df = create_irregularity_analysis(dataset, sample_size)
    
    return irregularity_df

def create_missingness_heatmap(dataset, sample_size):
    """Create a heatmap showing missingness patterns across patients and variables"""
    
    print("Creating missingness heatmap...")
    
    # Get variable names
    variables = dataset.params
    
    # Sample patients and create missingness matrix
    missingness_matrix = []
    patient_ids = []
    
    for i, (record_id, timestamps, values, mask) in enumerate(dataset.data[:sample_size]):
        # Calculate missingness percentage for each variable
        var_missingness = []
        for var_idx in range(len(variables)):
            total_possible = len(timestamps)
            observed = mask[:, var_idx].sum().item()
            missingness = 1 - (observed / total_possible)
            var_missingness.append(missingness)
        
        missingness_matrix.append(var_missingness)
        patient_ids.append(record_id)
        
        if i % 100 == 0:
            print(f"  Processed {i}/{sample_size} patients...")
    
    # Limit to 100 patients for visualization clarity
    display_size = min(100, len(missingness_matrix))
    missingness_df = pd.DataFrame(missingness_matrix[:display_size], 
                                index=patient_ids[:display_size],
                                columns=variables)
    
    # Create the heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(missingness_df, cmap='RdYlBu_r', vmin=0, vmax=1,
                cbar_kws={'label': 'Missingness Rate'}, 
                xticklabels=True, yticklabels=False)
    plt.title('Missingness Patterns Across Variables and Patients\n(Darker = More Missing)', 
              fontsize=14, pad=20)
    plt.xlabel('Physiological Variables', fontsize=12)
    plt.ylabel('Patient Records (Sample)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('eda_results/missingness_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_analysis(dataset, sample_size):
    """Create distribution plots for key physiological variables"""
    
    print("Creating distribution analysis...")
    
    variables = dataset.params
    key_variables = ['HR', 'Temp', 'SysABP', 'DiasABP', 'RespRate', 'Glucose', 'pH', 'HCT']
    
    # Collect all values for key variables
    variable_data = {var: [] for var in key_variables if var in variables}
    
    for i, (record_id, timestamps, values, mask) in enumerate(dataset.data[:sample_size]):
        for var_idx, var_name in enumerate(variables):
            if var_name in variable_data:
                var_values = values[:, var_idx][mask[:, var_idx] == 1]
                if len(var_values) > 0:
                    variable_data[var_name].extend(var_values.numpy())
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (var_name, data) in enumerate(variable_data.items()):
        if len(data) > 0 and i < len(axes):
            data = np.array(data)
            
            # Remove outliers for better visualization (keep 1st-99th percentile)
            q1, q99 = np.percentile(data, [1, 99])
            data_filtered = data[(data >= q1) & (data <= q99)]
            
            axes[i].hist(data_filtered, bins=50, alpha=0.7, density=True, 
                        color=sns.color_palette()[i % 8], edgecolor='black', linewidth=0.5)
            axes[i].set_title(f'{var_name} Distribution\n(n={len(data):,} observations)', fontsize=12)
            axes[i].set_xlabel('Value', fontsize=10)
            axes[i].set_ylabel('Density', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            
            # Add mean and median lines
            mean_val = np.mean(data_filtered)
            median_val = np.median(data_filtered)
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
            axes[i].axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.1f}')
            axes[i].legend(fontsize=8)
    
    plt.suptitle('Distribution of Key Physiological Variables', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('eda_results/variable_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_temporal_pattern_analysis(dataset, sample_size):
    """Analyze temporal patterns in measurements"""
    
    print("Creating temporal pattern analysis...")
    
    # Collect measurement times and intervals by variable
    var_measurement_times = defaultdict(list)
    variables = dataset.params
    
    for i, (record_id, timestamps, values, mask) in enumerate(dataset.data[:sample_size]):
        timestamps_hours = timestamps.numpy()
        
        for var_idx, var_name in enumerate(variables):
            var_mask = mask[:, var_idx] == 1
            if var_mask.sum() > 0:
                var_times = timestamps_hours[var_mask.numpy()]
                var_measurement_times[var_name].extend(var_times)
    
    # Focus on frequently measured variables
    frequent_vars = ['HR', 'SysABP', 'DiasABP', 'Temp', 'RespRate', 'HCT']
    available_frequent_vars = [v for v in frequent_vars if v in var_measurement_times and len(var_measurement_times[v]) > 100]
    
    # Create temporal analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, var_name in enumerate(available_frequent_vars[:6]):
        times = np.array(var_measurement_times[var_name])
        
        # Create histogram of measurement times
        axes[i].hist(times, bins=48, alpha=0.7, color=sns.color_palette()[i], 
                    edgecolor='black', linewidth=0.5, density=True)
        axes[i].set_title(f'{var_name} Measurement Times\n(n={len(times):,} measurements)', fontsize=12)
        axes[i].set_xlabel('Time (hours)', fontsize=10)
        axes[i].set_ylabel('Density', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Add vertical lines for 24h and 48h
        axes[i].axvline(24, color='red', linestyle='--', alpha=0.8, label='24h')
        axes[i].axvline(48, color='red', linestyle='--', alpha=0.8, label='48h')
        axes[i].legend(fontsize=8)
    
    plt.suptitle('Temporal Distribution of Measurements by Variable', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('eda_results/temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_analysis(dataset, sample_size):
    """Create correlation analysis between physiological variables"""
    
    print("Creating correlation analysis...")
    
    variables = dataset.params
    key_variables = ['HR', 'Temp', 'SysABP', 'DiasABP', 'RespRate', 'Glucose', 'pH', 'HCT', 'K', 'Na']
    available_vars = [v for v in key_variables if v in variables]
    
    # Collect synchronized measurements (same timestamp, same patient)
    correlation_data = {var: [] for var in available_vars}
    
    for i, (record_id, timestamps, values, mask) in enumerate(dataset.data[:sample_size]):
        for t_idx in range(len(timestamps)):
            # Check if we have measurements for multiple variables at this timestamp
            available_at_t = []
            values_at_t = []
            
            for var_idx, var_name in enumerate(variables):
                if var_name in available_vars and mask[t_idx, var_idx] == 1:
                    available_at_t.append(var_name)
                    values_at_t.append(values[t_idx, var_idx].item())
            
            # If we have measurements for at least 2 variables, record them
            if len(available_at_t) >= 2:
                for j, var_name in enumerate(available_at_t):
                    correlation_data[var_name].append(values_at_t[j])
    
    # Create correlation matrix
    correlation_matrix = np.full((len(available_vars), len(available_vars)), np.nan)
    
    for i, var1 in enumerate(available_vars):
        for j, var2 in enumerate(available_vars):
            if i == j:
                correlation_matrix[i, j] = 1.0
            elif len(correlation_data[var1]) > 10 and len(correlation_data[var2]) > 10:
                # Use only overlapping measurements
                min_len = min(len(correlation_data[var1]), len(correlation_data[var2]))
                if min_len > 10:
                    data1 = np.array(correlation_data[var1][:min_len])
                    data2 = np.array(correlation_data[var2][:min_len])
                    
                    # Remove outliers
                    valid_mask = (np.abs(data1 - np.mean(data1)) < 3 * np.std(data1)) & \
                                (np.abs(data2 - np.mean(data2)) < 3 * np.std(data2))
                    
                    if valid_mask.sum() > 10:
                        correlation_matrix[i, j] = np.corrcoef(data1[valid_mask], data2[valid_mask])[0, 1]
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.isnan(correlation_matrix)
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'},
                xticklabels=available_vars, yticklabels=available_vars)
    plt.title('Correlation Matrix of Physiological Variables\n(Based on Synchronized Measurements)', 
              fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('eda_results/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_icu_type_analysis(dataset, sample_size):
    """Analyze patterns by ICU type"""
    
    print("Creating ICU type analysis...")
    
    icu_types = defaultdict(int)
    icu_measurements = defaultdict(list)
    
    for i, (record_id, timestamps, values, mask) in enumerate(dataset.data[:sample_size]):
        # Get ICU type (index 3 in the variables list)
        icu_type_idx = dataset.params.index('ICUType')
        if mask[0, icu_type_idx] == 1:  # ICU type is recorded at admission
            icu_type = int(values[0, icu_type_idx].item())
            icu_types[icu_type] += 1
            
            # Count total measurements per ICU type
            total_measurements = mask.sum().item()
            icu_measurements[icu_type].append(total_measurements)
    
    # Create ICU type analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ICU type distribution
    icu_type_labels = {1: 'Coronary Care', 2: 'Cardiac Surgery Recovery', 
                      3: 'Medical ICU', 4: 'Surgical ICU'}
    
    types = list(icu_types.keys())
    counts = [icu_types[t] for t in types]
    labels = [icu_type_labels.get(t, f'Type {t}') for t in types]
    
    colors = sns.color_palette("Set2", len(types))
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Distribution of ICU Types\n(n={:,} patients)'.format(sum(counts)), fontsize=12)
    
    # Measurement intensity by ICU type
    for i, icu_type in enumerate(types):
        measurements = icu_measurements[icu_type]
        if measurements:
            ax2.boxplot(measurements, positions=[i], widths=0.6, 
                       patch_artist=True, boxprops=dict(facecolor=colors[i]))
    
    ax2.set_xticklabels([icu_type_labels.get(t, f'Type {t}') for t in types], rotation=45, ha='right')
    ax2.set_xlabel('ICU Type', fontsize=10)
    ax2.set_ylabel('Total Measurements per Patient', fontsize=10)
    ax2.set_title('Measurement Intensity by ICU Type', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_results/icu_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_irregularity_analysis(dataset, sample_size):
    """Analyze the irregular nature of the time series"""
    
    print("Creating irregularity analysis...")
    
    # Collect statistics on irregularity
    patient_irregularity_stats = []
    
    for i, (record_id, timestamps, values, mask) in enumerate(dataset.data[:sample_size//4]):  # Smaller sample for computation
        timestamps_hours = timestamps.numpy()
        
        # Calculate overall measurement frequency
        total_time = timestamps_hours.max() - timestamps_hours.min()
        if total_time > 0:
            avg_frequency = len(timestamps_hours) / total_time
        else:
            avg_frequency = 0
        
        # Calculate coefficient of variation of inter-measurement intervals
        if len(timestamps_hours) > 1:
            intervals = np.diff(timestamps_hours)
            cv_intervals = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        else:
            cv_intervals = 0
        
        # Calculate variable coverage (how many variables have at least one measurement)
        variable_coverage = (mask.sum(dim=0) > 0).float().mean().item()
        
        # Calculate sparsity
        total_possible = len(timestamps_hours) * len(dataset.params)
        total_observed = mask.sum().item()
        sparsity = 1 - (total_observed / total_possible)
        
        patient_irregularity_stats.append({
            'record_id': record_id,
            'avg_frequency': avg_frequency,
            'cv_intervals': cv_intervals,
            'variable_coverage': variable_coverage,
            'sparsity': sparsity,
            'sequence_length': len(timestamps_hours)
        })
    
    irregularity_df = pd.DataFrame(patient_irregularity_stats)
    
    # Create irregularity analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Frequency vs Sparsity
    scatter = axes[0, 0].scatter(irregularity_df['avg_frequency'], irregularity_df['sparsity'], 
                                alpha=0.6, c=irregularity_df['sequence_length'], cmap='viridis')
    axes[0, 0].set_xlabel('Average Measurement Frequency (measurements/hour)', fontsize=10)
    axes[0, 0].set_ylabel('Data Sparsity', fontsize=10)
    axes[0, 0].set_title('Measurement Frequency vs Data Sparsity', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Sequence Length')
    
    # CV of intervals distribution
    axes[0, 1].hist(irregularity_df['cv_intervals'], bins=30, alpha=0.7, 
                   color=sns.color_palette()[1], edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Coefficient of Variation of Intervals', fontsize=10)
    axes[0, 1].set_ylabel('Number of Patients', fontsize=10)
    axes[0, 1].set_title('Distribution of Temporal Irregularity\n(Higher CV = More Irregular)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Variable coverage distribution
    axes[1, 0].hist(irregularity_df['variable_coverage'], bins=30, alpha=0.7,
                   color=sns.color_palette()[2], edgecolor='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Variable Coverage (fraction of variables measured)', fontsize=10)
    axes[1, 0].set_ylabel('Number of Patients', fontsize=10)
    axes[1, 0].set_title('Distribution of Variable Coverage', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sparsity vs Variable Coverage
    axes[1, 1].scatter(irregularity_df['variable_coverage'], irregularity_df['sparsity'], 
                      alpha=0.6, color=sns.color_palette()[3])
    axes[1, 1].set_xlabel('Variable Coverage', fontsize=10)
    axes[1, 1].set_ylabel('Data Sparsity', fontsize=10)
    axes[1, 1].set_title('Variable Coverage vs Data Sparsity', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Multi-dimensional Analysis of Time Series Irregularity', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('eda_results/irregularity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return irregularity_df

def print_enhanced_summary(dataset, irregularity_df):
    """Print comprehensive summary with enhanced statistics"""
    
    print("\n" + "="*80)
    print("ENHANCED EDA SUMMARY FOR PHYSIONET CHALLENGE 2012")
    print("="*80)
    
    print(f"\nDATASET OVERVIEW:")
    print(f"• Dataset: PhysioNet/Computing in Cardiology Challenge 2012")
    print(f"• Total patients: {len(dataset):,}")
    print(f"• Total physiological variables: {len(dataset.params)}")
    print(f"• Observation period: 48 hours (ICU stay)")
    print(f"• Data source: ICU monitoring systems")
    
    print(f"\nVARIABLE CATEGORIES:")
    general_descriptors = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
    time_series_vars = [v for v in dataset.params if v not in general_descriptors]
    print(f"• General descriptors: {len(general_descriptors)} variables")
    print(f"• Time series variables: {len(time_series_vars)} variables")
    
    print(f"\nIRREGULARITY CHARACTERISTICS:")
    print(f"• Mean measurement frequency: {irregularity_df['avg_frequency'].mean():.2f} ± {irregularity_df['avg_frequency'].std():.2f} measurements/hour")
    print(f"• Mean temporal irregularity (CV): {irregularity_df['cv_intervals'].mean():.2f} ± {irregularity_df['cv_intervals'].std():.2f}")
    print(f"• Mean variable coverage: {irregularity_df['variable_coverage'].mean():.2f} ± {irregularity_df['variable_coverage'].std():.2f}")
    print(f"• Mean data sparsity: {irregularity_df['sparsity'].mean():.2f} ± {irregularity_df['sparsity'].std():.2f}")
    
    print(f"\nCLINICAL RELEVANCE:")
    print(f"• Focus: ICU mortality prediction")
    print(f"• Challenge: Irregular, sparse, multivariate time series")
    print(f"• Applications: Risk stratification, early warning systems")
    print(f"• Graph ML motivation: Model complex temporal-variable interactions")

def main():
    """Enhanced EDA pipeline"""
    
    print("Starting Enhanced Exploratory Data Analysis for PhysioNet Dataset")
    print("="*80)
    
    # Setup
    setup_environment()
    
    # Load dataset
    print("Loading PhysioNet dataset...")
    dataset = PhysioNet('data/physionet', download=True, quantization=0.0, device=torch.device("cpu"))
    print(f"Dataset loaded: {len(dataset)} patients")
    
    # Create enhanced visualizations
    irregularity_df = create_advanced_visualizations(dataset)
    
    # Print enhanced summary
    print_enhanced_summary(dataset, irregularity_df)
    
    print(f"\n✓ Enhanced EDA completed! Results saved to 'eda_results/' directory")
    print("Generated enhanced visualizations:")
    print("  • missingness_heatmap.png - Comprehensive missingness patterns")
    print("  • variable_distributions.png - Statistical distributions of key variables")
    print("  • temporal_patterns.png - Temporal measurement patterns by variable")
    print("  • correlation_matrix.png - Inter-variable correlations")
    print("  • icu_type_analysis.png - Analysis by ICU type")
    print("  • irregularity_analysis.png - Multi-dimensional irregularity analysis")

if __name__ == "__main__":
    main()
