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
    Plot eigenvalue gap evolution vs p in grid layout.

    Grid layout: rows = num_taxa, columns = sequence_length
    Each subplot shows Gap(L_M) and Gap(L_S) evolution with partition agreement overlay.

    Args:
        df: Full results DataFrame
        output_path: Where to save the plot
    """
    if 'mean_spectral_gap_L_M' not in df.columns or 'mean_spectral_gap_L_S' not in df.columns:
        print("  Spectral gap columns not found")
        return

    # Get unique values and sort
    num_taxa_vals = sorted(df['num_taxa'].unique())
    sequence_lengths = sorted(df['sequence_length'].unique())

    n_rows = len(num_taxa_vals)
    n_cols = len(sequence_lengths)

    # Create grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Handle single row/col cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Get partition agreement column
    agreement_col = 'partition_agreement_M' if 'partition_agreement_M' in df.columns else 'mean'

    for i, n in enumerate(num_taxa_vals):
        for j, L in enumerate(sequence_lengths):
            ax = axes[i, j]
            ax2 = ax.twinx()  # Secondary axis for partition agreement

            # Filter data
            df_subset = df[(df['num_taxa'] == n) & (df['sequence_length'] == L)].sort_values('p')

            if len(df_subset) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax2.set_yticks([])
                continue

            # Detect guardrails trigger point (after 2nd consecutive 100% agreement)
            # Only plot computed data, not filled data
            consecutive_100 = 0
            last_computed_idx = len(df_subset) - 1  # Default: all data

            for pos, (idx, row) in enumerate(df_subset.iterrows()):
                if row[agreement_col] >= 100.0:
                    consecutive_100 += 1
                    if consecutive_100 == 2:
                        # Guardrails triggered after this point
                        last_computed_idx = pos
                        break
                else:
                    consecutive_100 = 0

            # Use only computed data (up to and including 2nd 100%)
            df_plot = df_subset.iloc[:last_computed_idx + 1]

            # Plot spectral gaps on primary axis
            ax.plot(df_plot['p'], df_plot['mean_spectral_gap_L_M'],
                    'o-', color='#2E86AB', label='Gap(L_M)', linewidth=2.5, markersize=6)
            ax.plot(df_plot['p'], df_plot['mean_spectral_gap_L_S'],
                    's-', color='#A23B72', label='Gap(L_S)', linewidth=2.5, markersize=6)

            # Plot partition agreement on secondary axis (use full df_subset to show until p=1)
            if agreement_col in df_subset.columns:
                ax2.plot(df_subset['p'], df_subset[agreement_col],
                        'd--', color='#F18F01', alpha=0.5, linewidth=1.5, markersize=4,
                        label='Agreement')

            # Styling
            ax.set_xscale('log')
            ax.set_xlim([df['p'].min(), 1.0])  # Extend x-axis to p=1
            ax.set_xlabel('p', fontsize=10)
            ax.set_ylabel('Spectral Gap', fontsize=10, color='black')
            ax.set_title(f'n={n}, L={L}', fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='black')
            ax.grid(False)  # Remove grid

            # Secondary axis styling
            ax2.set_ylabel('Agreement (%)', fontsize=9, color='#F18F01', alpha=0.7)
            ax2.tick_params(axis='y', labelcolor='#F18F01', labelsize=8)
            ax2.set_ylim([0, 105])

            # Add row/column labels
            if j == 0:
                ax.text(-0.3, 0.5, f'n={n}', transform=ax.transAxes,
                       fontsize=12, fontweight='bold', rotation=90,
                       verticalalignment='center')
            if i == 0:
                ax.text(0.5, 1.15, f'L={L}', transform=ax.transAxes,
                       fontsize=12, fontweight='bold',
                       horizontalalignment='center')

    # Create unified legend below all plots
    # Get handles and labels from first subplot
    ax_first = axes[0, 0]
    ax2_first = ax_first.get_shared_y_axes().get_siblings(ax_first)[0] if hasattr(ax_first, 'get_shared_y_axes') else axes[0, 0].twinx()
    lines1, labels1 = axes[0, 0].get_legend_handles_labels()

    # Get second axis handles
    for ax_row in axes:
        for ax in ax_row:
            if hasattr(ax, 'twin_axes'):
                ax2_temp = ax.twin_axes
                break

    # Manually create legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2E86AB', marker='o', linestyle='-', linewidth=2.5, markersize=6, label='Gap(L_M)'),
        Line2D([0], [0], color='#A23B72', marker='s', linestyle='-', linewidth=2.5, markersize=6, label='Gap(L_S)'),
        Line2D([0], [0], color='#F18F01', marker='d', linestyle='--', linewidth=1.5, markersize=4, alpha=0.5, label='Agreement')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)

    # Main title and subtitle
    fig.suptitle('Spectral Gap Evolution (Grid: rows=n, columns=L)',
                 fontsize=14, fontweight='bold', y=0.985)
    fig.text(0.5, 0.955, 'Data after two consecutive 100% agreements is not shown (filled, not computed)',
             ha='center', fontsize=10, color='gray', style='italic')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
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
