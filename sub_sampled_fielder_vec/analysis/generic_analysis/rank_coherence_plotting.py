"""Grouped numerical rank and coherence plotting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from .grouped_plotting import group_data_by_model_and_n
from .matrix_quality import extract_numerical_rank, extract_coherence


def plot_numerical_rank_grouped(
    all_dataframes: Dict[str, pd.DataFrame],
    runs_config: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot numerical rank grouped by model, then by n.
    
    Creates 1×N grid (N models), each subplot shows curves for each n.
    
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
    
    for model_idx, label in enumerate(model_labels):
        ax = axes[model_idx]
        model_data = grouped_data[label]
        
        for n_val in sorted(model_data.keys()):
            if n_val == 0:
                continue
            
            df = model_data[n_val]
            rank_df = extract_numerical_rank(df, matrix_name="L_S")
            
            p_vals = rank_df['p'].values
            ranks = rank_df['numerical_rank'].values
            
            mask = ~(np.isnan(p_vals) | np.isnan(ranks))
            if np.sum(mask) == 0:
                continue
            
            color = n_color_map.get(n_val, 'gray')
            marker = n_marker_map.get(n_val, 'o')
            
            ax.plot(p_vals[mask], ranks[mask], marker=marker, color=color,
                   label=f'n={n_val}', linewidth=2, markersize=6, alpha=0.8)
            
            # Add reference line for n
            ax.axhline(y=n_val, color=color, linestyle=':', alpha=0.3, linewidth=1)
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Sampling Probability p', fontsize=10)
        if model_idx == 0:
            ax.set_ylabel('Numerical Rank', fontsize=10)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        if model_idx == num_models - 1:
            ax.legend(loc='best', fontsize=7, framealpha=0.9)
    
    plt.suptitle('Numerical Rank of Laplacian L_S vs p', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig


def plot_coherence_grouped(
    all_dataframes: Dict[str, pd.DataFrame],
    runs_config: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> plt.Figure:
    """Plot coherence grouped by model, then by n.
    
    Creates 1×N grid (N models), each subplot shows curves for each n.
    
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
    
    for model_idx, label in enumerate(model_labels):
        ax = axes[model_idx]
        model_data = grouped_data[label]
        
        for n_val in sorted(model_data.keys()):
            if n_val == 0:
                continue
            
            df = model_data[n_val]
            coherence_df = extract_coherence(df, matrix_name="L_S")
            
            p_vals = coherence_df['p'].values
            coherences = coherence_df['coherence'].values
            
            mask = ~(np.isnan(p_vals) | np.isnan(coherences))
            if np.sum(mask) == 0:
                continue
            
            color = n_color_map.get(n_val, 'gray')
            marker = n_marker_map.get(n_val, 'o')
            
            ax.plot(p_vals[mask], coherences[mask], marker=marker, color=color,
                   label=f'n={n_val}', linewidth=2, markersize=6, alpha=0.8)
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Sampling Probability p', fontsize=10)
        if model_idx == 0:
            ax.set_ylabel('Coherence μ', fontsize=10)
        
        ax.set_xscale('log')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        if model_idx == num_models - 1:
            ax.legend(loc='best', fontsize=7, framealpha=0.9)
    
    plt.suptitle('Matrix Coherence vs p', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig
