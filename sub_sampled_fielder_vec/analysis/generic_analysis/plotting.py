"""Plotting utilities for phase transition diagnostics."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any


def plot_metric_vs_p(ax, p_values, metric_values, label, color, marker,
                     log_x=True, log_y=False, reference_line=None):
    """Generic plotter for metric vs p with optional reference line.
    
    Args:
        ax: Matplotlib axes object
        p_values: Array or list of sampling probabilities
        metric_values: Array or list of metric values
        label: Label for the data series
        color: Color for the plot
        marker: Marker style
        log_x: Use log scale for x-axis (default: True)
        log_y: Use log scale for y-axis (default: False)
        reference_line: Optional tuple (p_ref, value_ref) for reference line
    """
    # Convert to numpy arrays to handle both lists and arrays
    p_values = np.asarray(p_values)
    metric_values = np.asarray(metric_values)
    
    # Remove NaN values
    mask = ~(np.isnan(p_values) | np.isnan(metric_values))
    if np.sum(mask) == 0:
        return
    
    p_clean = p_values[mask]
    metric_clean = metric_values[mask]
    
    # Plot data
    ax.scatter(p_clean, metric_clean, marker=marker, color=color, 
               label=label, s=100, alpha=0.7, zorder=3)
    ax.plot(p_clean, metric_clean, color=color, linestyle='-', 
            linewidth=2, alpha=0.5, zorder=2)
    
    # Plot reference line if provided
    if reference_line is not None:
        p_ref, value_ref = reference_line
        if not np.isnan(p_ref) and not np.isnan(value_ref):
            ax.axhline(y=value_ref, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label='Reference', zorder=1)
    
    # Set scales
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', framealpha=0.9)


def plot_summary_diagnostics(data_dict: Dict[str, Any], transitions_dict: Dict[str, Any],
                             output_path: Optional[Path] = None):
    """Create 3-panel summary: agreement, spectral gap, DK ratio vs p.
    
    Args:
        data_dict: Dictionary mapping run labels to DataFrames with metrics
        transitions_dict: Dictionary mapping run labels to transition DataFrames
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    # Panel 1: Partition Agreement
    ax1 = axes[0]
    for label, df in data_dict.items():
        if 'partition_agreement_S' in df.columns:
            p_vals = df['p'].values
            agreements = df['partition_agreement_S'].values
            ax1.plot(p_vals, agreements, marker='o', label=label, linewidth=2)
    ax1.axhline(y=95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='95% threshold')
    ax1.set_xscale('log')
    ax1.set_xlabel('Sampling Probability p', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Partition Agreement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Partition Agreement vs p', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Spectral Gap
    ax2 = axes[1]
    for label, df in data_dict.items():
        if 'mean_spectral_gap_L_S' in df.columns:
            p_vals = df['p'].values
            gaps = df['mean_spectral_gap_L_S'].values
            ax2.plot(p_vals, gaps, marker='s', label=label, linewidth=2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sampling Probability p', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Spectral Gap (λ₃ - λ₂)', fontsize=12, fontweight='bold')
    ax2.set_title('Spectral Gap vs p', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Panel 3: Davis-Kahan Ratio
    ax3 = axes[2]
    for label, df in data_dict.items():
        if 'mean_dk_ratio_S' in df.columns:
            p_vals = df['p'].values
            dk_ratios = df['mean_dk_ratio_S'].values
            # Filter out inf values
            mask = ~(np.isinf(dk_ratios) | np.isnan(dk_ratios))
            if np.sum(mask) > 0:
                ax3.plot(p_vals[mask], dk_ratios[mask], marker='^', label=label, linewidth=2)
    ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Stability (0.5)')
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Critical (1.0)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Sampling Probability p', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Davis-Kahan Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('Davis-Kahan Ratio vs p', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary diagnostics to: {output_path}")
    
    return fig
