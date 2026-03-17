"""Phase transition scaling plot utilities."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from analysis.comparison.phase_transition_utils import fit_power_law, evaluate_power_law


def plot_phase_transition_scaling(
    runs_config: list,
    fits: Dict[str, Dict[str, Any]],
    all_transitions: Dict[str, pd.DataFrame],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Plot phase transition scaling comparison (sigmoid-based and discrete thresholds).
    
    Creates a 2-panel figure showing:
    - Left: Sigmoid-based threshold (95% agreement) with power law fits
    - Right: Discrete threshold (100% agreement) with power law fits
    
    Args:
        runs_config: List of run configuration dicts with keys: label, color, marker
        fits: Dictionary mapping labels to fit results with keys: alpha, A, equation, n, p
        all_transitions: Dictionary mapping labels to transition DataFrames with columns:
                        n_taxa, p_star_sigmoid_95, p_star_discrete_100
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # LEFT PLOT: Sigmoid-based (95% threshold)
    for run_cfg in runs_config:
        label = run_cfg["label"]
        color = run_cfg["color"]
        marker = run_cfg["marker"]
        
        if label not in fits:
            continue
        
        fit = fits[label]
        
        # Plot fitted curve first (with label)
        n_range = np.logspace(np.log10(fit['n'].min() * 0.7), 
                               np.log10(fit['n'].max() * 1.3), 100)
        p_fit = evaluate_power_law(n_range, fit['alpha'], fit['A'])
        ax1.plot(n_range, p_fit, 
                color=color, 
                linestyle='--', linewidth=2.5, alpha=0.7,
                label=f"{label}: {fit['equation']}")
        
        # Data points on top (no separate label)
        ax1.scatter(fit['n'], fit['p'], 
                   marker=marker, s=200, 
                   color=color, 
                   edgecolor='black', linewidth=1.5, zorder=3)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Taxa (n)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Critical p* (95% agreement)', fontsize=14, fontweight='bold')
    ax1.set_title('Sigmoid-Based Threshold', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=11)

    # RIGHT PLOT: Discrete (100% threshold)
    for run_cfg in runs_config:
        label = run_cfg["label"]
        color = run_cfg["color"]
        marker = run_cfg["marker"]
        
        if label not in all_transitions:
            continue
        
        trans = all_transitions[label]
        n_vals = trans['n_taxa'].values
        p_vals = trans['p_star_discrete_100'].values
        
        # Remove NaN
        mask = ~np.isnan(p_vals)
        n_clean = n_vals[mask]
        p_clean = p_vals[mask]
        
        if len(n_clean) > 0:
            # Fit power law if enough points
            if len(n_clean) >= 2:
                alpha_d, A_d, eq_d = fit_power_law(n_clean, p_clean)
                
                # Plot fitted curve first (with label)
                n_range_d = np.logspace(np.log10(n_clean.min() * 0.7),
                                         np.log10(n_clean.max() * 1.3), 100)
                p_fit_d = evaluate_power_law(n_range_d, alpha_d, A_d)
                ax2.plot(n_range_d, p_fit_d,
                        color=color, linestyle='--', linewidth=2.5, alpha=0.7,
                        label=f"{label}: {eq_d}")
            
            # Data points on top (no separate label)
            ax2.scatter(n_clean, p_clean, 
                       marker=marker, s=200, 
                       color=color, 
                       edgecolor='black', linewidth=1.5, zorder=3)
            

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Taxa (n)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('First p* (100% agreement)', fontsize=14, fontweight='bold')
    ax2.set_title('Discrete Threshold', fontsize=16, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(labelsize=11)

    plt.suptitle('Phase Transition: Sample Complexity Scaling', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.absolute()}")

    return fig
