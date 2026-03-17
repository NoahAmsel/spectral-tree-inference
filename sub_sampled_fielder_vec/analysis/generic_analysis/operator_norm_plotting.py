"""Operator norm error scaling plot utilities."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional, List
from .scaling_analysis import fit_power_law_scaling, compute_theoretical_bound
from .plotting import plot_metric_vs_p
from .grouped_plotting import group_data_by_model_and_n


def plot_operator_norm_scaling_grouped(
    all_dataframes: Dict[str, pd.DataFrame],
    runs_config: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> tuple:
    """Plot operator norm error scaling grouped by model, then by n.
    
    Creates 2×N grid (2 metrics × N models):
    - Row 1: Operator norm error ||S - M||_op vs p
    - Row 2: Theoretical bound O(√n/p) vs p
    
    Args:
        all_dataframes: Dictionary mapping labels to DataFrames
        runs_config: List of run configuration dicts
        output_path: Optional path to save figure
        
    Returns:
        Tuple of (figure, scaling_results_dict)
    """
    grouped_data = group_data_by_model_and_n(all_dataframes)
    model_labels = [cfg['label'] for cfg in runs_config if cfg['label'] in grouped_data]
    num_models = len(model_labels)
    
    if num_models == 0:
        raise ValueError("No models found")
    
    fig, axes = plt.subplots(2, num_models, figsize=(5 * num_models, 10), sharex='col', sharey='row')
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
    
    scaling_results = {}
    
    for model_idx, label in enumerate(model_labels):
        ax_error = axes[0, model_idx]
        ax_theory = axes[1, model_idx]
        model_data = grouped_data[label]
        scaling_results[label] = {}
        
        for n_val in sorted(model_data.keys()):
            if n_val == 0:
                continue
            
            df = model_data[n_val]
            
            if 'mean_operator_norm_error' not in df.columns:
                continue
            
            p_vals = df['p'].values
            errors = df['mean_operator_norm_error'].values
            
            # Remove NaN
            mask = ~(np.isnan(p_vals) | np.isnan(errors) | (errors <= 0))
            if np.sum(mask) < 3:
                continue
            
            p_clean = p_vals[mask]
            err_clean = errors[mask]
            
            color = n_color_map.get(n_val, 'gray')
            marker = n_marker_map.get(n_val, 'o')
            
            # Plot error data
            ax_error.plot(p_clean, err_clean, marker=marker, color=color,
                        label=f'n={n_val}', linewidth=2, markersize=6, alpha=0.8)
            
            # Fit power law
            alpha, A, r_squared = fit_power_law_scaling(p_clean, err_clean)
            scaling_results[label][n_val] = {'alpha': alpha, 'A': A, 'r_squared': r_squared}
            
            if not np.isnan(alpha):
                # Plot fit
                p_fit = np.logspace(np.log10(p_clean.min()), np.log10(p_clean.max()), 100)
                err_fit = A * (p_fit ** (-alpha))
                ax_error.plot(p_fit, err_fit, color=color, linestyle='--',
                            linewidth=1.5, alpha=0.6)
                
                # Plot theoretical bound
                err_theory = compute_theoretical_bound(p_fit, n_val)
                ax_theory.plot(p_fit, err_theory, color=color, linestyle=':',
                            label=f'n={n_val}', linewidth=2, alpha=0.8)
        
        # Configure error subplot
        ax_error.set_title(label, fontsize=11, fontweight='bold')
        if model_idx == 0:
            ax_error.set_ylabel('||S - M||_op', fontsize=10)
        ax_error.set_xscale('log')
        ax_error.set_yscale('log')
        ax_error.grid(True, alpha=0.3, which='both')
        if model_idx == num_models - 1:
            ax_error.legend(loc='best', fontsize=7, framealpha=0.9)
        
        # Configure theory subplot
        if model_idx == 0:
            ax_theory.set_ylabel('Theoretical Bound', fontsize=10)
        ax_theory.set_xlabel('Sampling Probability p', fontsize=10)
        ax_theory.set_xscale('log')
        ax_theory.set_yscale('log')
        ax_theory.grid(True, alpha=0.3, which='both')
        if model_idx == num_models - 1:
            ax_theory.legend(loc='best', fontsize=7, framealpha=0.9)
    
    plt.suptitle('Operator Norm Error Scaling', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    # Print scaling results
    print("\nPower Law Scaling Results (||S-M|| ~ A * p^(-α)):")
    print("="*70)
    for label, model_results in scaling_results.items():
        print(f"\n{label}:")
        for n_val, result in sorted(model_results.items()):
            if not np.isnan(result['alpha']):
                print(f"  n={n_val}: α={result['alpha']:.4f}, A={result['A']:.4e}, R²={result['r_squared']:.4f}")
    
    return fig, scaling_results


def plot_operator_norm_scaling(
    runs_config: List[Dict[str, Any]],
    all_dataframes: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None
) -> tuple:
    """Plot operator norm error scaling analysis.
    
    Creates a 2-panel figure showing:
    - Left: Operator norm error ||S - M||_op vs p with power law fits
    - Right: Theoretical bound O(√n/p) vs p
    
    Args:
        runs_config: List of run configuration dicts with keys: label, color, marker
        all_dataframes: Dictionary mapping labels to DataFrames with metrics
        output_path: Optional path to save figure
        
    Returns:
        Tuple of (figure, scaling_results_dict)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    scaling_results = {}
    
    for run_cfg in runs_config:
        label = run_cfg["label"]
        color = run_cfg["color"]
        marker = run_cfg["marker"]
        
        if label not in all_dataframes:
            continue
        
        df = all_dataframes[label]
        
        if 'mean_operator_norm_error' not in df.columns:
            print(f"⚠ {label}: Missing operator norm error data")
            continue
        
        p_vals = df['p'].values
        errors = df['mean_operator_norm_error'].values
        
        # Remove NaN
        mask = ~(np.isnan(p_vals) | np.isnan(errors) | (errors <= 0))
        if np.sum(mask) < 3:
            continue
        
        p_clean = p_vals[mask]
        err_clean = errors[mask]
        
        # Plot data
        plot_metric_vs_p(axes[0], p_clean, err_clean, label, color, marker, 
                         log_x=True, log_y=True)
        
        # Fit power law
        alpha, A, r_squared = fit_power_law_scaling(p_clean, err_clean)
        scaling_results[label] = {'alpha': alpha, 'A': A, 'r_squared': r_squared}
        
        if not np.isnan(alpha):
            # Plot fit
            p_fit = np.logspace(np.log10(p_clean.min()), np.log10(p_clean.max()), 100)
            err_fit = A * (p_fit ** (-alpha))
            axes[0].plot(p_fit, err_fit, color=color, linestyle='--', 
                       linewidth=2, alpha=0.5, label=f'{label} fit (α={alpha:.3f})')
            
            # Plot theoretical bound (if we have n)
            if 'num_taxa' in df.columns:
                n = df['num_taxa'].iloc[0]
                err_theory = compute_theoretical_bound(p_fit, n)
                axes[1].plot(p_fit, err_theory, color=color, linestyle=':', 
                            linewidth=2, alpha=0.7, label=f'{label} theory (O(√n/p))')
    
    axes[0].set_title('Operator Norm Error ||S - M||_op vs p', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sampling Probability p', fontsize=12)
    axes[0].set_ylabel('||S - M||_op', fontsize=12)
    
    axes[1].set_title('Theoretical Bound O(√n/p) vs p', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Sampling Probability p', fontsize=12)
    axes[1].set_ylabel('Theoretical Bound', fontsize=12)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    # Print scaling results
    print("\nPower Law Scaling Results (||S-M|| ~ A * p^(-α)):")
    print("="*70)
    for label, result in scaling_results.items():
        if not np.isnan(result['alpha']):
            print(f"\n{label}:")
            print(f"  Exponent α = {result['alpha']:.4f}")
            print(f"  Prefactor A = {result['A']:.4e}")
            print(f"  R² = {result['r_squared']:.4f}")
            print(f"  Expected (random sampling): α ≈ 0.5")
    
    return fig, scaling_results
