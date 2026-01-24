"""Grouped plotting utilities: organize by model, then by n."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable, Optional
from collections import defaultdict


def group_data_by_model_and_n(all_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Group data by model (label) then by n (num_taxa).
    
    Args:
        all_dataframes: Dictionary mapping model labels to DataFrames
        
    Returns:
        Nested dictionary: {model_label: {n_value: DataFrame}}
    """
    grouped = {}
    
    for label, df in all_dataframes.items():
        if 'num_taxa' not in df.columns:
            # If no num_taxa column, treat entire DataFrame as single group
            grouped[label] = {0: df}
            continue
        
        # Group by num_taxa
        model_groups = {}
        for n_val in sorted(df['num_taxa'].unique()):
            n_df = df[df['num_taxa'] == n_val].copy()
            if len(n_df) > 0:
                model_groups[int(n_val)] = n_df
        
        grouped[label] = model_groups
    
    return grouped


def plot_metric_by_model_and_n(
    grouped_data: Dict[str, Dict[int, pd.DataFrame]],
    metric_extractor: Callable[[pd.DataFrame], tuple],
    runs_config: List[Dict[str, Any]],
    title: str,
    ylabel: str,
    log_x: bool = True,
    log_y: bool = False,
    figsize: Optional[tuple] = None
) -> plt.Figure:
    """Create subplot grid: one subplot per model, curves for each n.
    
    Args:
        grouped_data: {model_label: {n_value: DataFrame}}
        metric_extractor: Function(df) -> (p_values, metric_values) or None if missing
        runs_config: List of run configs for model labels/colors
        title: Overall figure title
        ylabel: Y-axis label
        log_x: Use log scale for x-axis
        log_y: Use log scale for y-axis
        figsize: Figure size (auto-calculated if None)
        
    Returns:
        Matplotlib Figure
    """
    # Get model labels from runs_config (in order)
    model_labels = [cfg['label'] for cfg in runs_config if cfg['label'] in grouped_data]
    num_models = len(model_labels)
    
    if num_models == 0:
        raise ValueError("No models found in grouped_data")
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (5 * num_models, 5)
    
    fig, axes = plt.subplots(1, num_models, figsize=figsize, sharey=True)
    if num_models == 1:
        axes = [axes]
    
    # Color palette for n values (consistent across models)
    n_colors = plt.cm.viridis(np.linspace(0.2, 0.8, 20))  # Generate many colors
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p']
    
    # Track all n values to create consistent legend
    all_n_values = set()
    for model_data in grouped_data.values():
        all_n_values.update(model_data.keys())
    all_n_values = sorted([n for n in all_n_values if n > 0])
    
    # Create color mapping for n values
    n_color_map = {n: n_colors[i % len(n_colors)] for i, n in enumerate(all_n_values)}
    n_marker_map = {n: markers[i % len(markers)] for i, n in enumerate(all_n_values)}
    
    for model_idx, label in enumerate(model_labels):
        ax = axes[model_idx]
        model_data = grouped_data[label]
        
        # Plot each n value in this model
        for n_val in sorted(model_data.keys()):
            if n_val == 0:  # Skip placeholder
                continue
            
            df = model_data[n_val]
            result = metric_extractor(df)
            
            if result is None:
                continue
            
            p_vals, metric_vals = result
            
            # Remove NaN
            mask = ~(np.isnan(p_vals) | np.isnan(metric_vals))
            if np.sum(mask) == 0:
                continue
            
            p_clean = p_vals[mask]
            metric_clean = metric_vals[mask]
            
            color = n_color_map.get(n_val, 'gray')
            marker = n_marker_map.get(n_val, 'o')
            
            ax.plot(p_clean, metric_clean, marker=marker, color=color,
                   label=f'n={n_val}', linewidth=2, markersize=6, alpha=0.8)
        
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Sampling Probability p', fontsize=10)
        if model_idx == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig
