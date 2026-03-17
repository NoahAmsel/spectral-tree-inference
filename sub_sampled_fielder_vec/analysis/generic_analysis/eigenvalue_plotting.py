"""Grouped eigenvalue spectrum plotting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from .grouped_plotting import group_data_by_model_and_n, plot_metric_by_model_and_n
from .eigenvalue_analysis import extract_eigenvalues, compute_relative_gap


def plot_eigenvalue_spectrum_grouped(
    all_dataframes: Dict[str, pd.DataFrame],
    runs_config: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot eigenvalue spectrum grouped by model, then by n.
    
    Creates 3×N grid (3 metrics × N models):
    - Row 1: Fiedler eigenvalue (λ₂)
    - Row 2: Third eigenvalue (λ₃)
    - Row 3: Relative spectral gap
    
    Args:
        all_dataframes: Dictionary mapping labels to DataFrames
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
    
    # Get all n values for consistent coloring
    all_n_values = set()
    for model_data in grouped_data.values():
        all_n_values.update(model_data.keys())
    all_n_values = sorted([n for n in all_n_values if n > 0])
    n_color_map = {n: n_colors[i % len(n_colors)] for i, n in enumerate(all_n_values)}
    n_marker_map = {n: markers[i % len(markers)] for i, n in enumerate(all_n_values)}
    
    # Plot each metric
    extractors = [
        ('lambda2', _extract_lambda2),
        ('lambda3', _extract_lambda3),
        ('relative_gap', _extract_relative_gap),
    ]
    
    for metric_idx, (metric_name, extractor) in enumerate(extractors):
        for model_idx, label in enumerate(model_labels):
            ax = axes[metric_idx, model_idx]
            model_data = grouped_data[label]
            
            for n_val in sorted(model_data.keys()):
                if n_val == 0:
                    continue
                
                df = model_data[n_val]
                result = extractor(df)
                
                if result is None:
                    continue
                
                p_vals, metric_vals = result
                
                mask = ~(np.isnan(p_vals) | np.isnan(metric_vals))
                if np.sum(mask) == 0:
                    continue
                
                p_clean = p_vals[mask]
                metric_clean = metric_vals[mask]
                
                color = n_color_map.get(n_val, 'gray')
                marker = n_marker_map.get(n_val, 'o')
                
                ax.plot(p_clean, metric_clean, marker=marker, color=color,
                       label=f'n={n_val}', linewidth=2, markersize=6, alpha=0.8)
            
            if metric_idx == 0:
                ax.set_title(label, fontsize=11, fontweight='bold')
            if model_idx == 0:
                if metric_name == 'lambda2':
                    ax.set_ylabel('λ₂', fontsize=10)
                elif metric_name == 'lambda3':
                    ax.set_ylabel('λ₃', fontsize=10)
                else:
                    ax.set_ylabel('(λ₃ - λ₂) / λ₂', fontsize=10)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, which='both')
            if metric_idx == 2:  # Only show xlabel on bottom row
                ax.set_xlabel('Sampling Probability p', fontsize=10)
            if model_idx == num_models - 1:  # Only show legend on rightmost column
                ax.legend(loc='best', fontsize=7, framealpha=0.9)
    
    plt.suptitle('Eigenvalue Spectrum Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig


def _extract_lambda2(df: pd.DataFrame) -> Optional[tuple]:
    """Extract lambda2 values."""
    try:
        eig_df = extract_eigenvalues(df, matrix_name="L_S")
        return eig_df['p'].values, eig_df['lambda2'].values
    except:
        return None


def _extract_lambda3(df: pd.DataFrame) -> Optional[tuple]:
    """Extract lambda3 values."""
    try:
        eig_df = extract_eigenvalues(df, matrix_name="L_S")
        return eig_df['p'].values, eig_df['lambda3'].values
    except:
        return None


def _extract_relative_gap(df: pd.DataFrame) -> Optional[tuple]:
    """Extract relative gap values."""
    try:
        eig_df = extract_eigenvalues(df, matrix_name="L_S")
        p_vals = eig_df['p'].values
        lambda2 = eig_df['lambda2'].values
        lambda3 = eig_df['lambda3'].values
        relative_gaps = np.array([compute_relative_gap(l2, l3) for l2, l3 in zip(lambda2, lambda3)])
        return p_vals, relative_gaps
    except:
        return None
