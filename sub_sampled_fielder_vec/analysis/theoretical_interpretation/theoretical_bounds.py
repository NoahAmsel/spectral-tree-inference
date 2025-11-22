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
    ||v - v_perturbed|| d ||Perturbation||_F / spectral_gap

    Args:
        transitions: DataFrame with transition points

    Returns:
        Dictionary with bound comparisons
    """
    if len(transitions) == 0:
        return {}

    stats = {}

    # Check if we have the required columns
    required_cols = ['mean_frobenius_error', 'mean_spectral_gap_L_M']
    if not all(col in transitions.columns for col in required_cols):
        return stats

    # Davis-Kahan bound
    perturbation = transitions['mean_frobenius_error']
    gap = transitions['mean_spectral_gap_L_M']

    # Avoid division by zero
    gap_safe = gap.replace(0, np.nan)
    dk_bound = perturbation / gap_safe

    stats['mean_dk_bound'] = float(dk_bound.mean())
    stats['median_dk_bound'] = float(dk_bound.median())

    return stats


def create_plot(transitions: pd.DataFrame, output_path: Path):
    """
    Plot theoretical bound vs empirical error.

    Args:
        transitions: DataFrame with transition points
        output_path: Where to save the plot
    """
    if len(transitions) == 0:
        print("  No transitions to plot")
        return

    required_cols = ['mean_frobenius_error', 'mean_spectral_gap_L_M', 'sequence_length']
    if not all(col in transitions.columns for col in required_cols):
        print("  Required columns not found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute Davis-Kahan bound
    perturbation = transitions['mean_frobenius_error']
    gap = transitions['mean_spectral_gap_L_M']
    gap_safe = gap.replace(0, np.nan)
    dk_bound = perturbation / gap_safe

    # Plot by sequence length
    for L in sorted(transitions['sequence_length'].unique()):
        trans_L = transitions[transitions['sequence_length'] == L]
        dk_L = dk_bound[transitions['sequence_length'] == L]
        ax.scatter(trans_L['p'], dk_L, s=100, alpha=0.7, label=f'L={L}')

    ax.set_xlabel('Critical Sampling Probability (p_crit)', fontsize=12)
    ax.set_ylabel('Davis-Kahan Bound (||E||_F / gap)', fontsize=12)
    ax.set_title('Theoretical Error Bound at Transition', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
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

    # Compute bounds
    stats = compute_davis_kahan_bound(transitions)

    if stats:
        print(f"  Mean Davis-Kahan bound at transition: {stats['mean_dk_bound']:.4f}")

    # Create plot
    create_plot(transitions, output_dir / "davis_kahan_bound.png")

    return stats
