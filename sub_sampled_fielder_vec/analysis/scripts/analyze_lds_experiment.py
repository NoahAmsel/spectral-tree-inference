#!/usr/bin/env python3
"""Analyze LDS experiment results across multiple tree sizes.

This script provides a comprehensive analysis of an LDS/HLDT experiment,
showing phase transition curves, critical p* values, and power law scaling.

Usage:
    python analyze_lds_experiment.py <experiment_dir>

Example:
    python analyze_lds_experiment.py ../../results/20260214-171901-kingman_mean_n512-8192_mu_0p1_hldt
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from comparison.phase_transition_utils import (
    sigmoid,
    fit_sigmoid_with_interpolation_fallback,
    fit_power_law,
    evaluate_power_law
)


def load_experiment_data(exp_dir: Path) -> Dict[int, List[Tuple[float, float]]]:
    """Load results from all n-value subdirectories.

    Args:
        exp_dir: Experiment directory containing n{X}_L{Y} subdirectories

    Returns:
        Dict mapping n_taxa -> [(p, partition_agreement_M), ...]
    """
    exp_dir = Path(exp_dir)
    data = {}

    # Find all n{X}_L{Y} subdirectories
    for subdir in sorted(exp_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("n"):
            continue

        # Extract n from directory name (e.g., n512_L10000 -> 512)
        try:
            n_taxa = int(subdir.name.split("_")[0][1:])
        except (IndexError, ValueError):
            continue

        # Load results.json
        results_file = subdir / "results.json"
        if not results_file.exists():
            print(f"⚠ Skipping {subdir.name}: no results.json")
            continue

        with results_file.open() as f:
            results = json.load(f)

        # Extract (p, partition_agreement_M) pairs
        p_agreement_pairs = []
        for row in results["rows"]:
            p = row["p"]
            agreement = row["partition_agreement_M"]
            p_agreement_pairs.append((p, agreement))

        # Sort by p
        p_agreement_pairs.sort(key=lambda x: x[0])
        data[n_taxa] = p_agreement_pairs

        print(f"✓ Loaded n={n_taxa}: {len(p_agreement_pairs)} p-values")

    return data


def compute_phase_transition_metrics(data: Dict[int, List[Tuple[float, float]]]) -> pd.DataFrame:
    """Compute p* and sigmoid parameters for each n.

    Args:
        data: Dict from load_experiment_data()

    Returns:
        DataFrame with columns: [n_taxa, p_star_95, sigmoid_L, sigmoid_k, sigmoid_p0,
                                  max_agreement, min_p, max_p, n_points]
    """
    rows = []

    for n_taxa in sorted(data.keys()):
        pairs = data[n_taxa]
        p_vals = np.array([p for p, _ in pairs])
        agreements = np.array([a for _, a in pairs])

        # Fit sigmoid with interpolation fallback
        sigmoid_params, p_star_95 = fit_sigmoid_with_interpolation_fallback(
            p_vals, agreements, threshold=95.0
        )
        L, k, p0 = sigmoid_params

        # Additional metrics
        max_agreement = agreements.max()
        min_p = p_vals.min()
        max_p = p_vals.max()
        n_points = len(p_vals)

        rows.append({
            "n_taxa": n_taxa,
            "p_star_95": p_star_95,
            "sigmoid_L": L,
            "sigmoid_k": k,
            "sigmoid_p0": p0,
            "max_agreement": max_agreement,
            "min_p": min_p,
            "max_p": max_p,
            "n_points": n_points
        })

    return pd.DataFrame(rows)


def plot_phase_transitions(data: Dict[int, List[Tuple[float, float]]],
                           metrics_df: pd.DataFrame,
                           save_path: Path = None):
    """Plot phase transition curves for all n values.

    Args:
        data: Dict from load_experiment_data()
        metrics_df: DataFrame from compute_phase_transition_metrics()
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors for different n values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(data)))

    # Plot 1: Phase transition curves (log scale)
    for i, n_taxa in enumerate(sorted(data.keys())):
        pairs = data[n_taxa]
        p_vals = np.array([p for p, _ in pairs])
        agreements = np.array([a for _, a in pairs])

        # Plot data points
        ax1.semilogx(p_vals, agreements, 'o', color=colors[i],
                     markersize=6, alpha=0.7, label=f'n={n_taxa}')

        # Plot fitted sigmoid if available
        row = metrics_df[metrics_df['n_taxa'] == n_taxa].iloc[0]
        if not np.isnan(row['sigmoid_L']):
            p_dense = np.logspace(np.log10(p_vals.min()), np.log10(p_vals.max()), 100)
            agreement_fit = sigmoid(p_dense, row['sigmoid_L'], row['sigmoid_k'], row['sigmoid_p0'])
            ax1.semilogx(p_dense, agreement_fit, '-', color=colors[i],
                        linewidth=2, alpha=0.5)

        # Mark p* (95% threshold)
        if not np.isnan(row['p_star_95']):
            ax1.plot(row['p_star_95'], 95, 'D', color=colors[i],
                    markersize=10, markeredgecolor='black', markeredgewidth=1.5)

    ax1.axhline(95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='95% threshold')
    ax1.axhline(50, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='Random baseline')
    ax1.set_xlabel('Sampling Rate p (log scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Partition Agreement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Phase Transition Curves: Agreement vs p', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(40, 105)

    # Plot 2: Critical p* scaling
    n_vals = metrics_df['n_taxa'].values
    p_star_vals = metrics_df['p_star_95'].values

    # Filter out NaN values for fitting
    mask = ~np.isnan(p_star_vals)
    if mask.sum() >= 2:
        # Fit power law
        alpha, A, equation = fit_power_law(n_vals[mask], p_star_vals[mask])

        # Plot data points
        ax2.loglog(n_vals, p_star_vals, 'o', markersize=10, color='steelblue',
                  markeredgecolor='black', markeredgewidth=1.5, label='Measured p*')

        # Plot fitted power law
        if not np.isnan(alpha):
            n_dense = np.logspace(np.log10(n_vals.min()), np.log10(n_vals.max()), 100)
            p_fit = evaluate_power_law(n_dense, alpha, A)
            ax2.loglog(n_dense, p_fit, '--', linewidth=2, color='red',
                      label=f'Fit: {equation}')

            # Add equation to plot
            ax2.text(0.05, 0.95, equation, transform=ax2.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        # Just plot points without fit
        ax2.loglog(n_vals, p_star_vals, 'o', markersize=10, color='steelblue',
                  markeredgecolor='black', markeredgewidth=1.5, label='Measured p*')

    ax2.set_xlabel('Number of Taxa n (log scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Critical p* (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Power Law Scaling: p* vs n', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")

    plt.show()


def print_summary_table(metrics_df: pd.DataFrame):
    """Print formatted summary table of metrics.

    Args:
        metrics_df: DataFrame from compute_phase_transition_metrics()
    """
    print("\n" + "="*90)
    print("PHASE TRANSITION SUMMARY")
    print("="*90)
    print(f"{'n_taxa':<10} {'p*_95%':<12} {'Max Agr%':<12} {'Sigmoid L':<12} {'Sigmoid k':<12} {'Points':<8}")
    print("-"*90)

    for _, row in metrics_df.iterrows():
        n_taxa = int(row['n_taxa'])
        p_star = row['p_star_95']
        max_agr = row['max_agreement']
        sigmoid_L = row['sigmoid_L']
        sigmoid_k = row['sigmoid_k']
        n_points = int(row['n_points'])

        p_star_str = f"{p_star:.6f}" if not np.isnan(p_star) else "N/A"
        sigmoid_L_str = f"{sigmoid_L:.2f}" if not np.isnan(sigmoid_L) else "N/A"
        sigmoid_k_str = f"{sigmoid_k:.2f}" if not np.isnan(sigmoid_k) else "N/A"

        print(f"{n_taxa:<10} {p_star_str:<12} {max_agr:<12.2f} {sigmoid_L_str:<12} {sigmoid_k_str:<12} {n_points:<8}")

    print("="*90)

    # Power law analysis
    n_vals = metrics_df['n_taxa'].values
    p_star_vals = metrics_df['p_star_95'].values
    mask = ~np.isnan(p_star_vals)

    if mask.sum() >= 2:
        alpha, A, equation = fit_power_law(n_vals[mask], p_star_vals[mask])

        print(f"\nPOWER LAW SCALING:")
        print(f"  {equation}")
        print(f"  Exponent α = {alpha:.4f}")
        print(f"  Coefficient A = {A:.4e}")

        # Interpretation
        if alpha < -0.5:
            interpretation = "✓ EXCELLENT: Strong inverse scaling (p* decreases rapidly with n)"
        elif alpha < -0.2:
            interpretation = "✓ GOOD: Moderate inverse scaling"
        elif alpha < 0:
            interpretation = "⚠ WEAK: Slight inverse scaling"
        else:
            interpretation = "✗ POOR: No inverse scaling (p* increases or stays constant with n)"

        print(f"  Interpretation: {interpretation}")
    else:
        print(f"\n⚠ Insufficient data for power law fitting (only {mask.sum()} valid points)")

    print("="*90)


def analyze_per_n_quality(data: Dict[int, List[Tuple[float, float]]]):
    """Analyze quality metrics for each n value.

    Args:
        data: Dict from load_experiment_data()
    """
    print("\n" + "="*90)
    print("PER-N QUALITY ANALYSIS")
    print("="*90)

    for n_taxa in sorted(data.keys()):
        pairs = data[n_taxa]
        p_vals = np.array([p for p, _ in pairs])
        agreements = np.array([a for _, a in pairs])

        # Compute quality metrics
        max_agr = agreements.max()
        min_agr = agreements.min()
        range_agr = max_agr - min_agr

        # Phase transition sharpness (how quickly it goes from 50% to 95%)
        idx_50 = np.argmin(np.abs(agreements - 50))
        idx_95 = np.argmin(np.abs(agreements - 95))
        p_50 = p_vals[idx_50] if agreements[idx_50] <= 60 else np.nan
        p_95 = p_vals[idx_95] if agreements[idx_95] >= 90 else np.nan

        transition_width = np.log10(p_95 / p_50) if not (np.isnan(p_50) or np.isnan(p_95)) else np.nan

        print(f"\nn = {n_taxa}")
        print(f"  P-value range:        [{p_vals.min():.6f}, {p_vals.max():.6f}]")
        print(f"  Agreement range:      [{min_agr:.1f}%, {max_agr:.1f}%]")
        print(f"  Transition width:     {transition_width:.2f} log10 units" if not np.isnan(transition_width) else "  Transition width:     N/A")

        # Quality assessment
        if max_agr >= 99:
            quality = "✓ EXCELLENT (reaches near-perfect agreement)"
        elif max_agr >= 95:
            quality = "✓ GOOD (reaches 95% threshold)"
        elif max_agr >= 90:
            quality = "⚠ MODERATE (reaches 90%, misses 95% threshold)"
        else:
            quality = "✗ POOR (never reaches 90% agreement)"

        print(f"  Quality:              {quality}")

    print("="*90)


def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp_dir = Path(sys.argv[1])

    if not exp_dir.exists():
        print(f"✗ Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)

    print(f"Analyzing experiment: {exp_dir.name}")
    print("="*90)

    # Load data
    print("\n1. Loading experiment data...")
    data = load_experiment_data(exp_dir)

    if not data:
        print("✗ Error: No data loaded. Check experiment directory structure.")
        sys.exit(1)

    print(f"✓ Loaded {len(data)} n-values: {sorted(data.keys())}")

    # Compute metrics
    print("\n2. Computing phase transition metrics...")
    metrics_df = compute_phase_transition_metrics(data)

    # Print summary
    print_summary_table(metrics_df)

    # Per-n quality analysis
    analyze_per_n_quality(data)

    # Plot phase transitions
    print("\n3. Generating phase transition plots...")
    save_path = exp_dir / "phase_transition_analysis.png"
    plot_phase_transitions(data, metrics_df, save_path)

    # Save metrics to CSV
    csv_path = exp_dir / "phase_transition_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Saved metrics to {csv_path}")

    print("\n" + "="*90)
    print("ANALYSIS COMPLETE")
    print("="*90)


if __name__ == "__main__":
    main()
