"""Plot comparison between uniform and leveraged sampling results.

Usage:
    python analysis/comparison/plot_comparison.py {comparison_dir}

Example:
    python analysis/comparison/plot_comparison.py results/20260117-114348-method_comparison
"""
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_merged_results(base_dir: Path, method: str):
    """Load merged results JSON for a method."""
    merged_file = base_dir / method / "results_grid_merged.json"
    if not merged_file.exists():
        raise FileNotFoundError(
            f"Merged results not found: {merged_file}\n"
            f"Run: python scripts/merge_results.py {base_dir / method}"
        )
    with merged_file.open() as f:
        return json.load(f)


def group_by_n_taxa(rows):
    """Group flat rows by num_taxa."""
    grouped = {}
    for row in rows:
        n = row["num_taxa"]
        if n not in grouped:
            grouped[n] = []
        grouped[n].append(row)
    return grouped


def plot_partition_agreement_vs_p(uniform_data, leveraged_data, output_dir: Path):
    """Plot partition agreement vs p for each n, comparing methods."""
    # Group rows by n_taxa
    uniform_by_n = group_by_n_taxa(uniform_data["rows"])
    leveraged_by_n = group_by_n_taxa(leveraged_data["rows"])

    # Get sorted n_taxa values
    n_taxa_values = sorted(uniform_by_n.keys())
    n_plots = len(n_taxa_values)

    # Arrange in grid
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for idx, n in enumerate(n_taxa_values):
        ax = axes[idx]

        # Extract p_values and agreements for this n
        rows_u = uniform_by_n[n]
        rows_l = leveraged_by_n[n]

        p_u = [row["p"] for row in rows_u]
        agreement_u = [row["partition_agreement_M"] for row in rows_u]

        p_l = [row["p"] for row in rows_l]
        agreement_l = [row["partition_agreement_M"] for row in rows_l]

        # Plot
        ax.plot(p_u, agreement_u, label="Uniform", marker='o', linewidth=2, markersize=6)
        ax.plot(p_l, agreement_l, label="Leveraged", marker='s', linewidth=2, markersize=6)

        ax.set_xlabel("Sampling probability p", fontsize=12)
        ax.set_ylabel("Partition agreement (%)", fontsize=12)
        ax.set_xscale("log")
        ax.set_title(f"n = {n} taxa", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        # Add 95% reference line
        ax.axhline(95, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = output_dir / "partition_agreement_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_p_min_vs_n(uniform_data, leveraged_data, output_dir: Path, threshold=95.0):
    """Plot minimum p needed to achieve threshold agreement vs n."""
    # Group by n_taxa
    uniform_by_n = group_by_n_taxa(uniform_data["rows"])
    leveraged_by_n = group_by_n_taxa(leveraged_data["rows"])

    def find_p_min(rows, threshold):
        """Find minimum p where agreement >= threshold."""
        for row in sorted(rows, key=lambda r: r["p"]):
            if row["partition_agreement_M"] >= threshold:
                return row["p"]
        return None  # Never reached threshold

    n_taxa_values = []
    p_min_uniform = []
    p_min_leveraged = []

    for n in sorted(uniform_by_n.keys()):
        p_u = find_p_min(uniform_by_n[n], threshold)
        p_l = find_p_min(leveraged_by_n[n], threshold)

        if p_u is not None and p_l is not None:
            n_taxa_values.append(n)
            p_min_uniform.append(p_u)
            p_min_leveraged.append(p_l)

    if not n_taxa_values:
        print(f"⚠ No data reached {threshold}% threshold")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_taxa_values, p_min_uniform, label="Uniform", marker='o',
            linewidth=3, markersize=10)
    ax.plot(n_taxa_values, p_min_leveraged, label="Leveraged", marker='s',
            linewidth=3, markersize=10)

    ax.set_xlabel("Number of taxa (n)", fontsize=14)
    ax.set_ylabel(f"Minimum p for {threshold}% agreement", fontsize=14)
    ax.set_title(f"Sampling Efficiency: p_min vs n (threshold={threshold}%)",
                 fontsize=16, fontweight='bold')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"p_min_vs_n_threshold{int(threshold)}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    comparison_dir = Path(sys.argv[1])
    if not comparison_dir.exists():
        print(f"Error: Directory not found: {comparison_dir}")
        sys.exit(1)

    print(f"Loading comparison results from: {comparison_dir}")

    # Load merged results
    try:
        uniform_data = load_merged_results(comparison_dir, "uniform")
        leveraged_data = load_merged_results(comparison_dir, "leveraged")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create output directory for plots
    output_dir = comparison_dir / "comparison_plots"
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_partition_agreement_vs_p(uniform_data, leveraged_data, output_dir)
    plot_p_min_vs_n(uniform_data, leveraged_data, output_dir, threshold=95.0)

    print(f"\n{'='*80}")
    print(f"Plots saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
