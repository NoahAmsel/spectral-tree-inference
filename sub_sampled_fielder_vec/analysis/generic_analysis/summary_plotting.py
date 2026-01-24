"""Grouped summary diagnostic plotting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from .grouped_plotting import group_data_by_model_and_n


def plot_summary_diagnostics_grouped(
    all_dataframes: Dict[str, pd.DataFrame],
    all_transitions: Dict[str, pd.DataFrame],
    runs_config: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot unified summary diagnostics grouped by model, then by n.
    
    Creates 3×N grid (3 metrics × N models):
    - Row 1: Partition agreement
    - Row 2: Spectral gap
    - Row 3: Davis-Kahan ratio
    
    Args:
        all_dataframes: Dictionary mapping labels to DataFrames
        all_transitions: Dictionary mapping labels to transition DataFrames
        runs_config: List of run configuration dicts
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure
    """
    grouped_data = group_data_by_model_and_n(all_dataframes)
    model_labels = [cfg['label'] for cfg in runs_config if cfg['label'] in grouped_data]
    num_models = len(model_labels)
    
    if num_models == 0:
        raise ValueError("No models found")
    
    fig, axes = plt.subplots(3, num_models, figsize=(5 * num_models, 12), sharex='col', sharey='row')
    if num_models == 1:
        axes = axes.reshape(3, 1)
    
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
    
    # Plot each metric
    for metric_idx, metric_name in enumerate(['agreement', 'gap', 'dk_ratio']):
        for model_idx, label in enumerate(model_labels):
            ax = axes[metric_idx, model_idx]
            model_data = grouped_data[label]
            
            for n_val in sorted(model_data.keys()):
                if n_val == 0:
                    continue
                
                df = model_data[n_val]
                p_vals = df['p'].values
                
                if metric_name == 'agreement':
                    if 'partition_agreement_S' in df.columns:
                        metric_vals = df['partition_agreement_S'].values
                    else:
                        continue
                elif metric_name == 'gap':
                    if 'mean_spectral_gap_L_S' in df.columns:
                        metric_vals = df['mean_spectral_gap_L_S'].values
                    else:
                        continue
                else:  # dk_ratio
                    if 'mean_dk_ratio_S' in df.columns:
                        metric_vals = df['mean_dk_ratio_S'].values
                        # Filter out inf
                        mask = ~(np.isinf(metric_vals) | np.isnan(metric_vals))
                        if np.sum(mask) == 0:
                            continue
                        p_vals = p_vals[mask]
                        metric_vals = metric_vals[mask]
                    else:
                        continue
                
                mask = ~(np.isnan(p_vals) | np.isnan(metric_vals))
                if np.sum(mask) == 0:
                    continue
                
                color = n_color_map.get(n_val, 'gray')
                marker = n_marker_map.get(n_val, 'o')
                
                ax.plot(p_vals[mask], metric_vals[mask], marker=marker, color=color,
                       label=f'n={n_val}', linewidth=2, markersize=6, alpha=0.8)
            
            # Add threshold lines
            if metric_name == 'agreement':
                ax.axhline(y=95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='95% threshold')
            elif metric_name == 'dk_ratio':
                ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Stability (0.5)')
                ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Critical (1.0)')
            
            if metric_idx == 0:
                ax.set_title(label, fontsize=11, fontweight='bold')
            if model_idx == 0:
                if metric_name == 'agreement':
                    ax.set_ylabel('Partition Agreement (%)', fontsize=10)
                elif metric_name == 'gap':
                    ax.set_ylabel('Spectral Gap (λ₃ - λ₂)', fontsize=10)
                else:
                    ax.set_ylabel('Davis-Kahan Ratio', fontsize=10)
            
            if metric_idx == 2:  # Bottom row
                ax.set_xlabel('Sampling Probability p', fontsize=10)
            
            ax.set_xscale('log')
            if metric_name != 'agreement':
                ax.set_yscale('log')
            ax.grid(True, alpha=0.3, which='both')
            if model_idx == num_models - 1:  # Rightmost column
                ax.legend(loc='best', fontsize=7, framealpha=0.9)
    
    plt.suptitle('Phase Transition Mechanism Summary', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig
