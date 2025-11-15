"""
Shared utilities for analyzing spectral tree reconstruction results.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set consistent plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_results(results_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load results_grid.json and convert to pandas DataFrame.

    Args:
        results_path: Path to results_grid.json. If None, uses default location.

    Returns:
        DataFrame with all results
    """
    if results_path is None:
        results_path = Path(__file__).parent.parent / "results" / "combined_grid_search_results" / "results_grid.json"

    with open(results_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data['rows'])

    # Add computed columns
    df['spectral_gap_ratio'] = df['mean_spectral_gap_L_S'] / df['mean_spectral_gap_L_M']
    df['rank_ratio_L_S'] = df['mean_empirical_rank_L_S'] / df['num_taxa']
    df['rank_ratio_L_M'] = df['mean_empirical_rank_L_M'] / df['num_taxa']
    df['matrix_size'] = df['num_taxa'] * df['sequence_length']
    df['effective_samples'] = df['p'] * df['num_taxa'] ** 2

    return df


def get_transition_point(df: pd.DataFrame, num_taxa: int, sequence_length: int,
                         threshold: float = 90.0, metric: str = 'mean') -> Optional[pd.Series]:
    """
    Find the transition point where sign agreement crosses the threshold.

    Args:
        df: DataFrame with results
        num_taxa: Number of taxa (n)
        sequence_length: Sequence length (L)
        threshold: Sign agreement threshold (default 90%)
        metric: Column name for sign agreement (default 'mean')

    Returns:
        Row at transition point, or None if no transition found
    """
    subset = df[(df['num_taxa'] == num_taxa) & (df['sequence_length'] == sequence_length)]
    subset_sorted = subset.sort_values('p')

    # Find first point where metric >= threshold
    transition = subset_sorted[subset_sorted[metric] >= threshold]

    if len(transition) > 0:
        return transition.iloc[0]
    return None


def get_all_transitions(df: pd.DataFrame, threshold: float = 90.0) -> pd.DataFrame:
    """
    Get transition points for all (num_taxa, sequence_length) combinations.

    Args:
        df: DataFrame with results
        threshold: Sign agreement threshold (default 90%)

    Returns:
        DataFrame with transition points
    """
    transitions = []

    for num_taxa in df['num_taxa'].unique():
        for seq_len in df['sequence_length'].unique():
            trans_point = get_transition_point(df, num_taxa, seq_len, threshold)
            if trans_point is not None:
                transitions.append(trans_point)

    if transitions:
        return pd.DataFrame(transitions)
    return pd.DataFrame()


def plot_metric_vs_p_faceted(df: pd.DataFrame, metric: str, ylabel: str,
                             output_path: str, log_y: bool = False,
                             show_sign_agreement: bool = True):
    """
    Create faceted plot showing metric vs p for different sequence lengths.

    Args:
        df: DataFrame with results
        metric: Column name to plot
        ylabel: Label for y-axis
        output_path: Path to save figure
        log_y: Use log scale for y-axis
        show_sign_agreement: Overlay sign agreement on secondary axis
    """
    sequence_lengths = sorted(df['sequence_length'].unique())
    num_taxa_vals = sorted(df['num_taxa'].unique())

    fig, axes = plt.subplots(len(sequence_lengths), 1, figsize=(10, 3*len(sequence_lengths)))
    if len(sequence_lengths) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.4, len(num_taxa_vals)))

    for idx, L in enumerate(sequence_lengths):
        ax = axes[idx]
        ax2 = ax.twinx() if show_sign_agreement else None

        for n, color in zip(num_taxa_vals, colors):
            subset = df[(df['num_taxa'] == n) & (df['sequence_length'] == L)]
            subset_sorted = subset.sort_values('p')

            # Plot main metric
            ax.plot(subset_sorted['p'], subset_sorted[metric],
                   'o-', color=color, label=f'n={n}', markersize=6, linewidth=2)

            # Plot sign agreement on secondary axis
            if show_sign_agreement and ax2 is not None:
                ax2.plot(subset_sorted['p'], subset_sorted['mean'],
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
            ax2.set_ylabel('Sign agreement (%)', fontsize=10, color='gray', alpha=0.7)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=9)
            ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def save_markdown_report(content: str, output_path: str):
    """
    Save a markdown report.

    Args:
        content: Markdown content
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Saved report: {output_path}")


def get_output_dir(phase: int) -> Path:
    """
    Get output directory for a specific phase.

    Args:
        phase: Phase number (1, 2, or 3)

    Returns:
        Path to output directory
    """
    base_dir = Path(__file__).parent.parent / "results" / "combined_grid_search_results" / "analysis_outputs"
    phase_dir = base_dir / f"phase{phase}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    return phase_dir


if __name__ == "__main__":
    # Test the utilities
    print("Testing utilities...")
    df = load_results()
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nUnique num_taxa: {sorted(df['num_taxa'].unique())}")
    print(f"Unique sequence_length: {sorted(df['sequence_length'].unique())}")
    print(f"Unique p values: {len(df['p'].unique())}")

    # Test transition finding
    trans = get_transition_point(df, 8192, 10000)
    if trans is not None:
        print(f"\nTransition for n=8192, L=10000:")
        print(f"  p = {trans['p']:.4f}")
        print(f"  sign agreement = {trans['mean']:.1f}%")
        print(f"  spectral gap ratio = {trans['spectral_gap_ratio']:.2f}")
