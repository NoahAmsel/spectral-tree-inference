"""Fit power law relationships for transition_p vs matrix size."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import find_all_transitions
import matplotlib.pyplot as plt


def power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * np.power(x, b)


def fit_power_law(transitions: pd.DataFrame) -> dict:
    """
    Fit power law to transition_p vs sequence_length.

    Args:
        transitions: DataFrame with transition points

    Returns:
        Dictionary with fit parameters and R^2
    """
    if len(transitions) == 0:
        return {}

    # Group by sequence_length, average transition p
    grouped = transitions.groupby('sequence_length').agg({
        'p': 'mean',
        'num_taxa': 'first'
    }).reset_index()

    x = grouped['sequence_length'].values
    y = grouped['p'].values

    try:
        # Fit power law: p_crit = a * L^b
        params, _ = curve_fit(power_law, x, y, p0=[1.0, -0.5])
        a, b = params

        # Compute R^2
        y_pred = power_law(x, a, b)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            'a': float(a),
            'b': float(b),
            'r_squared': float(r_squared),
            'formula': f'p_crit = {a:.4f} * L^{b:.4f}'
        }
    except:
        return {}


def create_plot(transitions: pd.DataFrame, output_path: Path, fit_params: dict):
    """
    Create power law fit visualization.

    Args:
        transitions: DataFrame with transition points
        output_path: Where to save the plot
        fit_params: Dictionary from fit_power_law()
    """
    if len(transitions) == 0:
        print("  No transitions to plot")
        return

    grouped = transitions.groupby('sequence_length').agg({
        'p': 'mean',
        'num_taxa': 'first'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot of actual transitions
    ax.scatter(grouped['sequence_length'], grouped['p'],
               s=100, alpha=0.7, label='Observed transitions')

    # Plot power law fit if available
    if fit_params:
        x_fit = np.linspace(grouped['sequence_length'].min(),
                           grouped['sequence_length'].max(), 100)
        y_fit = power_law(x_fit, fit_params['a'], fit_params['b'])
        ax.plot(x_fit, y_fit, 'r--', linewidth=2,
                label=f"{fit_params['formula']} (R^2={fit_params['r_squared']:.3f})")

    ax.set_xlabel('Sequence Length (L)', fontsize=12)
    ax.set_ylabel('Critical Sampling Probability (p_crit)', fontsize=12)
    ax.set_title('Power Law Scaling: p_crit vs L', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run power law fitting analysis.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Fit parameters dictionary
    """
    print("\n=== Power Law Fitting (p_crit vs L) ===")

    # Find all transitions
    transitions = find_all_transitions(df, threshold=90.0)

    if len(transitions) == 0:
        print("  No transitions found!")
        return {}

    # Fit power law
    fit_params = fit_power_law(transitions)

    if fit_params:
        print(f"  Power law fit: {fit_params['formula']}")
        print(f"  R^2 = {fit_params['r_squared']:.4f}")
    else:
        print("  Power law fit failed")

    # Create plot
    create_plot(transitions, output_dir / "power_law_fit.png", fit_params)

    return fit_params
