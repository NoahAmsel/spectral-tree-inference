"""Grouped Davis-Kahan perturbation plotting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from .grouped_plotting import group_data_by_model_and_n


def plot_davis_kahan_grouped(
    all_dataframes: Dict[str, pd.DataFrame],
    runs_config: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> tuple:
    """Plot Davis-Kahan ratio grouped by model, then by n.
    
    Creates 1×N grid (N models), each subplot shows curves for each n.
    
    Args:
        all_dataframes: Dictionary mapping labels to DataFrames
        runs_config: List of run configuration dicts
        output_path: Optional path to save figure
        
    Returns:
        Tuple of (figure, dk_crossings_dict)
    """
    grouped_data = group_data_by_model_and_n(all_dataframes)
    model_labels = [cfg['label'] for cfg in runs_config if cfg['label'] in grouped_data]
    num_models = len(model_labels)
    
    if num_models == 0:
        raise ValueError("No models found")
    
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5), sharey=True)
    if num_models == 1:
        axes = [axes]
    
    # Color palette for n values
    n_colors = plt.cm.viridis(np.linspace(0.2, 0.8, 20))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p']
    
    # Get all n values
    all_n_values = set()
    for model_data in grouped_data.values():
        all_n_values.update(model_data.keys())
    all_n_values = sorted([n for n in all_n_values if n > 0])
    n_color_map = {n: n_colors[i % len(n_colors)] for i, n in enumerate(all_n_values)}
    n_marker_map = {n: markers[i % len(markers)] for i, n in enumerate(all_n_values)}
    
    dk_crossings = {}
    
    for model_idx, label in enumerate(model_labels):
        ax = axes[model_idx]
        model_data = grouped_data[label]
        dk_crossings[label] = {}
        
        for n_val in sorted(model_data.keys()):
            if n_val == 0:
                continue
            
            df = model_data[n_val]
            
            if 'mean_dk_ratio_S' not in df.columns:
                continue
            
            p_vals = df['p'].values
            dk_ratios = df['mean_dk_ratio_S'].values
            
            # Filter out inf values
            mask = ~(np.isinf(dk_ratios) | np.isnan(dk_ratios))
            if np.sum(mask) == 0:
                continue
            
            p_clean = p_vals[mask]
            dk_clean = dk_ratios[mask]
            
            color = n_color_map.get(n_val, 'gray')
            marker = n_marker_map.get(n_val, 'o')
            
            ax.plot(p_clean, dk_clean, marker=marker, color=color,
                   label=f'n={n_val}', linewidth=2, markersize=6, alpha=0.8)
        
        # Add threshold lines
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Stability (0.5)')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Critical (1.0)')
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Sampling Probability p', fontsize=10)
        if model_idx == 0:
            ax.set_ylabel('Davis-Kahan Ratio', fontsize=10)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        if model_idx == num_models - 1:
            ax.legend(loc='best', fontsize=7, framealpha=0.9)
    
    plt.suptitle('Davis-Kahan Ratio vs p', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig, dk_crossings
