"""Compare empirical transitions to Davis-Kahan theoretical bounds."""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import find_all_transitions


def compute_davis_kahan_bound(transitions: pd.DataFrame) -> dict:
    """
    Compute theoretical bounds from Davis-Kahan theorem.

    The Davis-Kahan theorem states:
    ||v - v_perturbed|| ≤ ||Perturbation||_2 / spectral_gap

    Args:
        transitions: DataFrame with transition points

    Returns:
        Dictionary with bound comparisons
    """
    if len(transitions) == 0:
        return {}

    stats = {}

    # Check for error norm column (try both names for backward compatibility)
    error_col = None
    if 'mean_operator_norm_error' in transitions.columns:
        error_col = 'mean_operator_norm_error'
    elif 'mean_frobenius_error' in transitions.columns:
        error_col = 'mean_frobenius_error'  # Legacy name (actually spectral norm)

    # Check if we have the required columns
    if error_col is None or 'mean_spectral_gap_L_M' not in transitions.columns:
        return stats

    # Davis-Kahan bound
    perturbation = transitions[error_col]
    gap = transitions['mean_spectral_gap_L_M']

    # Avoid division by zero
    gap_safe = gap.replace(0, np.nan)
    dk_bound = perturbation / gap_safe

    stats['mean_dk_bound'] = float(dk_bound.mean())
    stats['median_dk_bound'] = float(dk_bound.median())

    return stats


def create_plot(transitions: pd.DataFrame, output_path: Path):
    """
    Plot theoretical bound at transition points only (legacy plot).

    Args:
        transitions: DataFrame with transition points
        output_path: Where to save the plot
    """
    if len(transitions) == 0:
        print("  No transitions to plot")
        return

    # Check for error norm column (try both names)
    error_col = None
    if 'mean_operator_norm_error' in transitions.columns:
        error_col = 'mean_operator_norm_error'
    elif 'mean_frobenius_error' in transitions.columns:
        error_col = 'mean_frobenius_error'  # Legacy name

    if error_col is None or 'mean_spectral_gap_L_M' not in transitions.columns or 'sequence_length' not in transitions.columns:
        print("  Required columns not found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute Davis-Kahan bound (using spectral norm)
    perturbation = transitions[error_col]
    gap = transitions['mean_spectral_gap_L_M']
    gap_safe = gap.replace(0, np.nan)
    dk_bound = perturbation / gap_safe

    # Plot by sequence length
    for L in sorted(transitions['sequence_length'].unique()):
        trans_L = transitions[transitions['sequence_length'] == L]
        dk_L = dk_bound[transitions['sequence_length'] == L]
        ax.scatter(trans_L['p'], dk_L, s=100, alpha=0.7, label=f'L={L}')

    # Add reference lines
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Bound = 1 (theory)')
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Bound = 10')

    ax.set_xlabel('Critical Sampling Probability (p_crit)', fontsize=12)
    ax.set_ylabel('Davis-Kahan Bound (||E||_2 / δ)', fontsize=12)  # Corrected label
    ax.set_title('Theoretical Error Bound at Transition', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_evolution_plot(df: pd.DataFrame, output_path: Path):
    """
    Plot Davis-Kahan bound evolution across ALL p values in grid layout.

    Grid layout: rows = num_taxa, columns = sequence_length
    Shows how ||E||_2 / δ changes from failure to success regime.

    Args:
        df: Full results DataFrame
        output_path: Where to save the plot
    """
    # Check for error norm column (try both names)
    error_col = None
    if 'mean_operator_norm_error' in df.columns:
        error_col = 'mean_operator_norm_error'
    elif 'mean_frobenius_error' in df.columns:
        error_col = 'mean_frobenius_error'  # Legacy name

    if error_col is None or 'mean_spectral_gap_L_M' not in df.columns:
        print("  Required columns not found for evolution plot")
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

            # Compute DK bound (using spectral norm)
            gap = df_plot['mean_spectral_gap_L_M'].replace(0, np.nan)
            dk_bound = df_plot[error_col] / gap

            # For log scale plotting: replace computed zeros with small value
            # This makes them visible on log scale while being honest about what's computed
            dk_bound_plot = dk_bound.copy()
            min_nonzero = dk_bound[dk_bound > 0].min() if (dk_bound > 0).any() else 0.01
            epsilon = min_nonzero / 100  # Place zeros well below smallest real value
            dk_bound_plot = dk_bound_plot.replace(0, epsilon)

            # Plot DK bound on primary axis
            ax.plot(df_plot['p'], dk_bound_plot,
                    'o-', color='#6A4C93', label='||E||₂ / δ', linewidth=2.5, markersize=6)

            # Add reference lines
            ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                      label='Bound = 1' if i == 0 and j == 0 else '')
            ax.axhline(y=10, color='orange', linestyle='--', linewidth=1, alpha=0.5,
                      label='Bound = 10' if i == 0 and j == 0 else '')

            # Plot partition agreement on secondary axis (use full df_subset to show until p=1)
            if agreement_col in df_subset.columns:
                ax2.plot(df_subset['p'], df_subset[agreement_col],
                        's--', color='#06A77D', alpha=0.5, linewidth=1.5, markersize=4,
                        label='Agreement' if i == 0 and j == 0 else '')

            # Styling
            ax.set_xscale('log')
            ax.set_yscale('log')  # Back to regular log scale
            ax.set_xlim([df['p'].min(), 1.0])  # Extend x-axis to p=1
            ax.set_xlabel('p', fontsize=10)
            ax.set_ylabel('DK Bound', fontsize=10, color='black')
            ax.set_title(f'n={n}, L={L}', fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='black')
            ax.grid(False)  # Remove grid

            # Secondary axis styling
            ax2.set_ylabel('Agreement (%)', fontsize=9, color='#06A77D', alpha=0.7)
            ax2.tick_params(axis='y', labelcolor='#06A77D', labelsize=8)
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
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#6A4C93', marker='o', linestyle='-', linewidth=2.5, markersize=6, label='||E||₂ / δ'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Bound = 1'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Bound = 10'),
        Line2D([0], [0], color='#06A77D', marker='s', linestyle='--', linewidth=1.5, markersize=4, alpha=0.5, label='Agreement')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)

    # Main title and subtitle
    fig.suptitle('Davis-Kahan Bound Evolution (Grid: rows=n, columns=L)',
                 fontsize=14, fontweight='bold', y=0.985)
    fig.text(0.5, 0.935, 'Data after two consecutive 100% agreements is not shown (filled, not computed)',
             ha='center', fontsize=10, color='gray', style='italic')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run theoretical bounds analysis.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Theoretical bounds statistics
    """
    print("\n=== Theoretical Bounds (Davis-Kahan) ===")

    # Find transitions
    transitions = find_all_transitions(df, threshold=90.0)

    # Compute bounds at transitions
    stats = compute_davis_kahan_bound(transitions)

    if stats:
        print(f"  Mean Davis-Kahan bound at transition: {stats['mean_dk_bound']:.4f}")

    # Create transition plot (legacy, single points)
    create_plot(transitions, output_dir / "davis_kahan_bound.png")

    # Create evolution plot (new, full p range)
    create_evolution_plot(df, output_dir / "davis_kahan_bound_evolution.png")

    return stats
