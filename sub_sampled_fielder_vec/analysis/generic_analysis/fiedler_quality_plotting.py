"""Grouped Fiedler vector quality plotting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from .grouped_plotting import group_data_by_model_and_n
from .fiedler_diagnostics import compute_ipr, analyze_sign_stability


def plot_fiedler_quality_grouped(
    all_dataframes: Dict[str, pd.DataFrame],
    runs_config: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> tuple:
    """Plot Fiedler vector quality metrics grouped by model, then by n.
    
    Creates 2×N grid (2 metrics × N models):
    - Row 1: IPR localization ratio
    - Row 2: Sign agreement stability
    
    Args:
        all_dataframes: Dictionary mapping labels to DataFrames
        runs_config: List of run configuration dicts
        output_path: Optional path to save figure
        
    Returns:
        Tuple of (figure, sign_stability_stats_dict)
    """
    grouped_data = group_data_by_model_and_n(all_dataframes)
    model_labels = [cfg['label'] for cfg in runs_config if cfg['label'] in grouped_data]
    num_models = len(model_labels)
    
    if num_models == 0:
        raise ValueError("No models found")
    
    fig, axes = plt.subplots(2, num_models, figsize=(5 * num_models, 10), sharex='col')
    if num_models == 1:
        axes = axes.reshape(2, 1)
    
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
    
    sign_stability_stats = {}
    
    for model_idx, label in enumerate(model_labels):
        ax_ipr = axes[0, model_idx]
        ax_sign = axes[1, model_idx]
        model_data = grouped_data[label]
        sign_stability_stats[label] = {}
        
        for n_val in sorted(model_data.keys()):
            if n_val == 0:
                continue
            
            df = model_data[n_val]
            
            color = n_color_map.get(n_val, 'gray')
            marker = n_marker_map.get(n_val, 'o')
            
            # IPR analysis
            if 'mean_ipr_S' in df.columns:
                p_vals = df['p'].values
                ipr_vals = df['mean_ipr_S'].values
                localization_ratios = compute_ipr(ipr_vals, n_val)
                
                mask = ~(np.isnan(p_vals) | np.isnan(localization_ratios))
                if np.sum(mask) > 0:
                    ax_ipr.plot(p_vals[mask], localization_ratios[mask], 
                               marker=marker, color=color, label=f'n={n_val}',
                               linewidth=2, markersize=6, alpha=0.8)
            
            # Sign stability
            if 'sign_agreement' in df.columns:
                p_vals = df['p'].values
                sign_agreements = df['sign_agreement'].values
                
                mask = ~(np.isnan(p_vals) | np.isnan(sign_agreements))
                if np.sum(mask) > 0:
                    ax_sign.plot(p_vals[mask], sign_agreements[mask],
                               marker=marker, color=color, label=f'n={n_val}',
                               linewidth=2, markersize=6, alpha=0.8)
                    
                    # Compute stats
                    stats = analyze_sign_stability(sign_agreements[mask])
                    sign_stability_stats[label][n_val] = stats
        
        # Configure IPR subplot
        ax_ipr.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, 
                      alpha=0.7, label='Delocalized (good)')
        ax_ipr.set_title(label, fontsize=11, fontweight='bold')
        if model_idx == 0:
            ax_ipr.set_ylabel('Localization Ratio', fontsize=10)
        ax_ipr.set_xscale('log')
        ax_ipr.set_yscale('log')
        ax_ipr.set_ylim([0.1, None])
        ax_ipr.grid(True, alpha=0.3, which='both')
        if model_idx == num_models - 1:
            ax_ipr.legend(loc='best', fontsize=7, framealpha=0.9)
        
        # Configure sign stability subplot
        ax_sign.axhline(y=95, color='green', linestyle='--', linewidth=1.5,
                       alpha=0.7, label='95% threshold')
        if model_idx == 0:
            ax_sign.set_ylabel('Sign Agreement (%)', fontsize=10)
        ax_sign.set_xlabel('Sampling Probability p', fontsize=10)
        ax_sign.set_xscale('log')
        ax_sign.grid(True, alpha=0.3)
        if model_idx == num_models - 1:
            ax_sign.legend(loc='best', fontsize=7, framealpha=0.9)
    
    plt.suptitle('Fiedler Vector Quality Metrics', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    # Print sign stability statistics
    print("\nSign Stability Statistics:")
    print("="*60)
    for label, model_stats in sign_stability_stats.items():
        print(f"\n{label}:")
        for n_val, stats in sorted(model_stats.items()):
            print(f"  n={n_val}: Mean={stats['mean']:.2f}%, Median={stats['median']:.2f}%, Std={stats['std']:.2f}%")
    
    return fig, sign_stability_stats
