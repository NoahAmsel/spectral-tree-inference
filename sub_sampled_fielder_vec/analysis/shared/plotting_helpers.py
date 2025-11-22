"""Reusable plotting functions for analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metric_faceted(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    log_y: bool = False,
    show_partition_agreement: bool = True
):
    """
    Create faceted plot showing metric vs p for different sequence lengths.

    Args:
        df: DataFrame with results
        metric: Column name to plot on primary y-axis
        ylabel: Label for primary y-axis
        output_path: Path to save figure
        log_y: Use log scale for primary y-axis
        show_partition_agreement: Show partition agreement on secondary y-axis
    """
    sequence_lengths = sorted(df['sequence_length'].unique())
    num_taxa_vals = sorted(df['num_taxa'].unique())

    fig, axes = plt.subplots(len(sequence_lengths), 1, figsize=(10, 3*len(sequence_lengths)))
    if len(sequence_lengths) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.4, len(num_taxa_vals)))

    for idx, L in enumerate(sequence_lengths):
        ax = axes[idx]
        ax2 = ax.twinx() if show_partition_agreement else None

        for n, color in zip(num_taxa_vals, colors):
            subset = df[(df['num_taxa'] == n) & (df['sequence_length'] == L)]
            subset_sorted = subset.sort_values('p')

            # Plot main metric
            ax.plot(subset_sorted['p'], subset_sorted[metric],
                   'o-', color=color, label=f'n={n}', markersize=6, linewidth=2)

            # Plot partition agreement on secondary axis
            if show_partition_agreement and ax2 is not None:
                agreement_col = 'partition_agreement_M' if 'partition_agreement_M' in df.columns else 'mean'
                ax2.plot(subset_sorted['p'], subset_sorted[agreement_col],
                        's--', color=color, alpha=0.3, markersize=4, linewidth=1)

        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel('Sampling probability (p)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11, color='black')
        ax.set_title(f'L = {L}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        if ax2 is not None:
            ax2.set_ylabel('Partition agreement (%)', fontsize=10, color='gray', alpha=0.7)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=9)
            ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gap_vs_p(df: pd.DataFrame, output_path: Path):
    """
    Plot quality gap (partition_agreement_M - partition_agreement_S) vs p.

    Args:
        df: DataFrame with partition_agreement_M and partition_agreement_S columns
        output_path: Path to save figure
    """
    if 'partition_agreement_M' not in df.columns or 'partition_agreement_S' not in df.columns:
        print("Warning: partition agreement columns not found. Skipping gap plot.")
        return

    df_copy = df.copy()
    df_copy['partition_gap'] = df_copy['partition_agreement_M'] - df_copy['partition_agreement_S']

    sequence_lengths = sorted(df_copy['sequence_length'].unique())
    num_taxa_vals = sorted(df_copy['num_taxa'].unique())

    fig, axes = plt.subplots(len(sequence_lengths), 1, figsize=(10, 3*len(sequence_lengths)))
    if len(sequence_lengths) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.4, len(num_taxa_vals)))

    for idx, L in enumerate(sequence_lengths):
        ax = axes[idx]

        for n, color in zip(num_taxa_vals, colors):
            subset = df_copy[(df_copy['num_taxa'] == n) & (df_copy['sequence_length'] == L)]
            subset_sorted = subset.sort_values('p')

            ax.plot(subset_sorted['p'], subset_sorted['partition_gap'],
                   'o-', color=color, label=f'n={n}', markersize=6, linewidth=2)

        # Reference lines
        ax.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='5% threshold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        ax.set_xscale('log')
        ax.set_xlabel('Sampling probability (p)', fontsize=11)
        ax.set_ylabel('Quality Gap (%)\n(M - S_avg)', fontsize=11)
        ax.set_title(f'Partition Agreement Gap: L = {L}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, max(df_copy['partition_gap'].max() + 5, 15)])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
