#!/usr/bin/env python3
"""Generate phase transition plots for a single experiment run.

Usage:
    python analysis/plot_phase_transition.py results/SOME_RUN_DIR

Analyzes p* vs n scaling for a single sampling method across multiple n values.
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from analysis.comparison.phase_transition_utils import (
    fit_sigmoid_with_interpolation_fallback,
    find_discrete_threshold,
    fit_power_law,
    evaluate_power_law,
)


def load_single_run_data(run_dir: Path) -> Dict[int, List[Tuple[float, float]]]:
    """Load results from single experiment run with multiple n values.

    Args:
        run_dir: Directory containing n{X}_L{Y} subdirectories

    Returns:
        Dict mapping n_taxa -> [(p, agreement), ...]
    """
    run_dir = Path(run_dir)
    data = {}

    # Find all n{X}_L{Y} subdirectories
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("n"):
            continue

        # Extract n from directory name
        n_taxa = int(subdir.name.split("_")[0][1:])

        # Load results.json
        results_file = subdir / "results.json"
        if not results_file.exists():
            continue

        with results_file.open() as f:
            results = json.load(f)

        # Extract (p, partition_agreement_M) pairs
        pairs = [(row["p"], row["partition_agreement_M"]) for row in results["rows"]]
        pairs.sort(key=lambda x: x[0])
        data[n_taxa] = pairs

    if not data:
        raise ValueError(f"No n*_L* subdirectories found in {run_dir}")

    return data


def compute_transitions(data: Dict[int, List[Tuple[float, float]]]) -> pd.DataFrame:
    """Compute p* thresholds for all n values.

    Args:
        data: Dict from load_single_run_data()

    Returns:
        DataFrame with [n_taxa, p_star_sigmoid_95, p_star_discrete_100, ...]
    """
    rows = []

    for n_taxa in sorted(data.keys()):
        pairs = data[n_taxa]
        p_vals = np.array([p for p, _ in pairs])
        agreements = np.array([a for _, a in pairs])

        print(f"Processing n={n_taxa}...")

        # Sigmoid fit
        sigmoid_params, p_star_95 = fit_sigmoid_with_interpolation_fallback(
            p_vals, agreements, threshold=95.0)
        L, k, p0 = sigmoid_params

        # Discrete threshold
        p_star_100 = find_discrete_threshold(p_vals, agreements, threshold=100.0)

        rows.append({
            "n_taxa": n_taxa,
            "p_star_sigmoid_95": p_star_95,
            "p_star_discrete_100": p_star_100 if p_star_100 is not None else np.nan,
            "sigmoid_L": L,
            "sigmoid_k": k,
            "sigmoid_p0": p0,
        })

    return pd.DataFrame(rows)


def plot_phase_transition(transitions: pd.DataFrame, output_dir: Path, method_name: str = "Method"):
    """Generate side-by-side phase transition plots.

    Args:
        transitions: DataFrame from compute_transitions()
        output_dir: Directory to save plots
        method_name: Name for plot labels (default: "Method")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    n_vals = transitions['n_taxa'].values
    p_sigmoid = transitions['p_star_sigmoid_95'].values
    p_discrete = transitions['p_star_discrete_100'].values

    color = '#2E86AB'
    marker = 'o'

    # LEFT: Sigmoid-based (95% threshold) with power law fit
    mask_sigmoid = ~np.isnan(p_sigmoid)
    if np.sum(mask_sigmoid) >= 2:
        n_clean = n_vals[mask_sigmoid]
        p_clean = p_sigmoid[mask_sigmoid]

        # Fit power law
        alpha, A, equation = fit_power_law(n_clean, p_clean)

        # Plot fit first (background)
        n_range = np.logspace(np.log10(n_clean.min() * 0.7),
                               np.log10(n_clean.max() * 1.3), 100)
        p_fit = evaluate_power_law(n_range, alpha, A)
        ax1.plot(n_range, p_fit, color=color, linestyle='--',
                linewidth=3, alpha=0.7, label=f"{method_name}: {equation}")

        # Data points on top (no separate label)
        ax1.scatter(n_clean, p_clean, marker=marker, s=250,
                   color=color, edgecolor='black', linewidth=2, zorder=3)

        # Add alpha annotation
        ax1.text(0.05, 0.95, f"Exponent α = {alpha:.3f}",
                transform=ax1.transAxes, fontsize=13,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Taxa (n)', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Critical p* (95% agreement)', fontsize=15, fontweight='bold')
    ax1.set_title('Sigmoid-Based Threshold', fontsize=17, fontweight='bold', pad=15)
    ax1.legend(fontsize=12, loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=12)

    # RIGHT: Discrete (100% threshold)
    mask_discrete = ~np.isnan(p_discrete)
    if np.sum(mask_discrete) > 0:
        n_clean = n_vals[mask_discrete]
        p_clean = p_discrete[mask_discrete]

        # Fit power law if enough points
        if len(n_clean) >= 2:
            alpha_d, A_d, eq_d = fit_power_law(n_clean, p_clean)

            # Plot fitted curve first (background)
            n_range_d = np.logspace(np.log10(n_clean.min() * 0.7),
                                     np.log10(n_clean.max() * 1.3), 100)
            p_fit_d = evaluate_power_law(n_range_d, alpha_d, A_d)
            ax2.plot(n_range_d, p_fit_d, color=color, linestyle='--',
                    linewidth=2.5, alpha=0.7, label=f"{method_name}: {eq_d}")

            # Show exponent in text box
            ax2.text(0.05, 0.95, f"Exponent α = {alpha_d:.3f}",
                    transform=ax2.transAxes, fontsize=13,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            # Just label the data if no fit
            label_text = f"{method_name}"

        # Data points on top (no separate label)
        ax2.scatter(n_clean, p_clean, marker=marker, s=250,
                   color=color, edgecolor='black', linewidth=2, zorder=3)

        # Connect with lines if multiple points (no label)
        if len(n_clean) > 1:
            ax2.plot(n_clean, p_clean, color=color, linestyle='-',
                    linewidth=2, alpha=0.6)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Taxa (n)', fontsize=15, fontweight='bold')
    ax2.set_ylabel('First p* (100% agreement)', fontsize=15, fontweight='bold')
    ax2.set_title('Discrete Threshold', fontsize=17, fontweight='bold', pad=15)
    ax2.legend(fontsize=12, loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(labelsize=12)

    plt.suptitle(f'Phase Transition: {method_name}',
                 fontsize=19, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = output_dir / "phase_transition.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate phase transition plots for a single experiment run."
    )
    parser.add_argument("run_dir", type=Path,
                       help="Experiment directory containing n*_L*/results.json")
    parser.add_argument("--name", type=str, default="Method",
                       help="Method name for plot labels (default: 'Method')")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)

    print(f"Loading data from: {run_dir}")

    # Load and process
    data = load_single_run_data(run_dir)
    print(f"Found {len(data)} different n values: {sorted(data.keys())}")

    transitions = compute_transitions(data)

    # Display table
    print("\n" + "="*80)
    print("Phase Transition Critical Points (p*)")
    print("="*80)
    print(transitions[['n_taxa', 'p_star_sigmoid_95', 'p_star_discrete_100']].to_string())
    print("="*80)

    # Generate plot
    plot_phase_transition(transitions, run_dir, method_name=args.name)


if __name__ == "__main__":
    main()
