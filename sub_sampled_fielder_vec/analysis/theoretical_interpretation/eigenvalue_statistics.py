"""Analyze eigenvalue distribution and spectral gap behavior."""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import find_all_transitions


def compute_stats(transitions: pd.DataFrame) -> dict:
    """
    Compute eigenvalue-related statistics at transition.

    Args:
        transitions: DataFrame with transition points

    Returns:
        Dictionary of statistics
    """
    if len(transitions) == 0:
        return {}

    stats = {}

    # Spectral gap statistics
    if 'mean_spectral_gap_L_M' in transitions.columns:
        stats['mean_gap_LM'] = float(transitions['mean_spectral_gap_L_M'].mean())
        stats['median_gap_LM'] = float(transitions['mean_spectral_gap_L_M'].median())

    if 'mean_spectral_gap_L_S' in transitions.columns:
        stats['mean_gap_LS'] = float(transitions['mean_spectral_gap_L_S'].mean())
        stats['median_gap_LS'] = float(transitions['mean_spectral_gap_L_S'].median())

    # Separation statistics
    if 'mean_min_separation_L_S' in transitions.columns:
        stats['mean_separation'] = float(transitions['mean_min_separation_L_S'].mean())
        stats['min_separation'] = float(transitions['mean_min_separation_L_S'].min())

    return stats


def create_plot(df: pd.DataFrame, output_path: Path):
    """
    Plot eigenvalue gap evolution vs p.

    Args:
        df: Full results DataFrame
        output_path: Where to save the plot
    """
    if 'mean_spectral_gap_L_M' not in df.columns or 'mean_spectral_gap_L_S' not in df.columns:
        print("  Spectral gap columns not found")
        return

    # Get unique combinations
    combos = df.groupby(['num_taxa', 'sequence_length']).size().reset_index()[['num_taxa', 'sequence_length']]
    n_combos = len(combos)

    fig, axes = plt.subplots(1, n_combos, figsize=(6 * n_combos, 5))
    if n_combos == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(combos.iterrows()):
        n = row['num_taxa']
        L = row['sequence_length']
        ax = axes[idx]

        # Filter data
        df_subset = df[(df['num_taxa'] == n) & (df['sequence_length'] == L)].sort_values('p')

        # Plot both gaps
        ax.plot(df_subset['p'], df_subset['mean_spectral_gap_L_M'],
                'o-', label='Gap(L_M)', linewidth=2)
        ax.plot(df_subset['p'], df_subset['mean_spectral_gap_L_S'],
                's-', label='Gap(L_S)', linewidth=2)

        ax.set_xlabel('Sampling Probability (p)', fontsize=11)
        ax.set_ylabel('Spectral Gap', fontsize=11)
        ax.set_title(f'n={n}, L={L}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Spectral Gap Evolution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run eigenvalue statistics analysis.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Eigenvalue statistics dictionary
    """
    print("\n=== Eigenvalue Statistics ===")

    # Find transitions
    transitions = find_all_transitions(df, threshold=90.0)

    # Compute stats
    stats = compute_stats(transitions)

    if stats:
        if 'mean_gap_LM' in stats:
            print(f"  Mean spectral gap (L_M) at transition: {stats['mean_gap_LM']:.4f}")
        if 'mean_gap_LS' in stats:
            print(f"  Mean spectral gap (L_S) at transition: {stats['mean_gap_LS']:.4f}")

    # Create plot
    create_plot(df, output_dir / "eigenvalue_gaps_vs_p.png")

    return stats
