"""Generate formatted tables for stability analysis results.

Tables display mean ± std for numeric metrics, success rates for
partition validity, and typical partition splits with variability.
"""
from typing import List, Dict
from pathlib import Path
import numpy as np


def write_stability_table(results: List[Dict], output_path: Path) -> None:
    """
    Write stability diagnostics to a formatted .txt table.

    Each row shows aggregated statistics from K trials of the same configuration.

    Args:
        results: List of aggregated stability result dictionaries
        output_path: Path to output .txt file
    """
    # Column headers and widths (wider for mean±std format)
    headers = [
        ("Model", 20),
        ("n", 6),
        ("L", 6),
        ("mu", 6),
        ("Trials", 7),
        ("Coherence", 15),
        ("NumRank", 12),
        ("Sigma2", 13),
        ("Partition", 15),
        ("Valid", 8),
        ("Gap", 13),
        ("RelGap", 13),
        ("Lambda2", 13),
        ("Lambda3", 13)
    ]

    # Create separator line
    separator = "─" * sum(width for _, width in headers)

    # Open file for writing
    with open(output_path, 'w') as f:
        # Write header
        f.write("Stability Analysis - Statistical Diagnostics Across K Trials\n")
        f.write("=" * len(separator) + "\n\n")
        f.write("Values shown as: mean ± std (for numeric metrics)\n")
        f.write("Partition validity shown as: success/total\n")
        f.write("\n")

        # Write column headers
        header_line = ""
        for name, width in headers:
            header_line += f"{name:<{width}}"
        f.write(header_line + "\n")
        f.write(separator + "\n")

        # Write data rows
        for result in results:
            # Extract metadata
            model = result['tree_model']
            n = result['n']
            L = result['L']
            mu = result['mu']
            num_trials = result['num_trials']

            # Extract mean ± std for numeric metrics
            coherence_mean = result['coherence']['mean']
            coherence_std = result['coherence']['std']

            numrank_mean = result['numerical_rank']['mean']
            numrank_std = result['numerical_rank']['std']

            sigma2_mean = result['sigma2']['mean']
            sigma2_std = result['sigma2']['std']

            gap_mean = result['spectral_gap']['mean']
            gap_std = result['spectral_gap']['std']

            relgap_mean = result['relative_spectral_gap']['mean']
            relgap_std = result['relative_spectral_gap']['std']

            lambda2_mean = result['lambda2']['mean']
            lambda2_std = result['lambda2']['std']

            lambda3_mean = result['lambda3']['mean']
            lambda3_std = result['lambda3']['std']

            # Extract partition split statistics
            split_info = result['partition_split']
            small_mean = split_info['mean_small']
            large_mean = split_info['mean_large']
            small_std = split_info['std_small']
            large_std = split_info['std_large']

            # Format partition as "small|large ± std"
            # We'll show the typical split with combined variability
            partition_str = f"{small_mean:.0f}|{large_mean:.0f}±{max(small_std, large_std):.1f}"

            # Extract partition validity
            valid_info = result['is_valid_partition']
            success_count = valid_info['success_count']
            total_trials = valid_info['num_trials']
            valid_str = f"{success_count}/{total_trials}"

            # Format row with proper widths
            row = ""
            row += f"{model:<20}"
            row += f"{n:<6}"
            row += f"{L:<6}"
            row += f"{mu:<6.2f}"
            row += f"{num_trials:<7}"
            row += f"{coherence_mean:>6.4f}±{coherence_std:<6.4f}"
            row += f"{numrank_mean:>5.2f}±{numrank_std:<4.2f}"
            row += f"{sigma2_mean:>5.3f}±{sigma2_std:<5.3f}"
            row += f"{partition_str:<15}"
            row += f"{valid_str:<8}"
            row += f"{gap_mean:>5.3f}±{gap_std:<5.3f}"
            row += f"{relgap_mean:>5.3f}±{relgap_std:<5.3f}"
            row += f"{lambda2_mean:>5.3f}±{lambda2_std:<5.3f}"
            row += f"{lambda3_mean:>5.3f}±{lambda3_std:<5.3f}"

            f.write(row + "\n")

        # Footer
        f.write(separator + "\n")
        f.write(f"\nTotal configurations analyzed: {len(results)}\n")

    print(f"✓ Stability table written to: {output_path}")


def write_stability_summary(results: List[Dict], output_path: Path) -> None:
    """
    Write summary statistics across all configurations.

    This shows how metrics vary across different (model, n, L, mu) combinations.

    Args:
        results: List of aggregated stability result dictionaries
        output_path: Path to output .txt file
    """
    # Collect mean values across all configs (to see config-to-config variation)
    coherence_means = [r['coherence']['mean'] for r in results]
    numrank_means = [r['numerical_rank']['mean'] for r in results]
    sigma2_means = [r['sigma2']['mean'] for r in results]
    gap_means = [r['spectral_gap']['mean'] for r in results]
    relgap_means = [r['relative_spectral_gap']['mean'] for r in results]

    # Collect success rates
    success_rates = [r['is_valid_partition']['success_rate'] for r in results]

    with open(output_path, 'w') as f:
        f.write("Stability Summary - Variation Across Configurations\n")
        f.write("=" * 60 + "\n\n")
        f.write("This shows how trial-averaged metrics vary across different\n")
        f.write("(model, n, L, μ) configurations.\n\n")

        metrics = [
            ("Coherence (mean)", coherence_means),
            ("Numerical Rank (mean)", numrank_means),
            ("Sigma2 (mean)", sigma2_means),
            ("Spectral Gap (mean)", gap_means),
            ("Relative Spectral Gap (mean)", relgap_means),
            ("Partition Validity Success Rate", success_rates)
        ]

        for name, values in metrics:
            f.write(f"{name}:\n")
            f.write(f"  Mean:   {np.mean(values):10.6f}\n")
            f.write(f"  Median: {np.median(values):10.6f}\n")
            f.write(f"  Min:    {np.min(values):10.6f}\n")
            f.write(f"  Max:    {np.max(values):10.6f}\n")
            f.write(f"  Std:    {np.std(values):10.6f}\n")
            f.write("\n")

        # Report on partition validity across all configs
        f.write("Partition Validity Analysis:\n")
        perfect_count = sum(1 for r in results if r['is_valid_partition']['success_count'] == r['num_trials'])
        partial_count = sum(1 for r in results if 0 < r['is_valid_partition']['success_count'] < r['num_trials'])
        fail_count = sum(1 for r in results if r['is_valid_partition']['success_count'] == 0)

        f.write(f"  Configs with 100% success: {perfect_count}/{len(results)}\n")
        f.write(f"  Configs with partial success: {partial_count}/{len(results)}\n")
        f.write(f"  Configs with 0% success: {fail_count}/{len(results)}\n")

    print(f"✓ Stability summary written to: {output_path}")


def write_detailed_metrics_table(results: List[Dict], output_path: Path) -> None:
    """
    Write detailed table showing mean, std, min, max for ALL metrics.

    This is a more verbose table for deeper analysis.

    Args:
        results: List of aggregated stability result dictionaries
        output_path: Path to output .txt file
    """
    with open(output_path, 'w') as f:
        f.write("Detailed Stability Metrics - Full Statistics\n")
        f.write("=" * 120 + "\n\n")

        for idx, result in enumerate(results):
            model = result['tree_model']
            n = result['n']
            L = result['L']
            mu = result['mu']
            num_trials = result['num_trials']

            f.write(f"\n[{idx+1}] {model}, n={n}, L={L}, μ={mu} ({num_trials} trials)\n")
            f.write("-" * 80 + "\n")

            # Define metrics to display
            metrics = [
                ("Coherence", result['coherence']),
                ("Numerical Rank", result['numerical_rank']),
                ("Sigma2", result['sigma2']),
                ("Spectral Gap", result['spectral_gap']),
                ("Relative Spectral Gap", result['relative_spectral_gap']),
                ("Lambda2", result['lambda2']),
                ("Lambda3", result['lambda3'])
            ]

            for metric_name, stats in metrics:
                f.write(f"  {metric_name:25s}: ")
                f.write(f"mean={stats['mean']:8.4f}  ")
                f.write(f"std={stats['std']:8.4f}  ")
                f.write(f"min={stats['min']:8.4f}  ")
                f.write(f"max={stats['max']:8.4f}\n")

            # Partition split
            split = result['partition_split']
            f.write(f"  {'Partition Split':25s}: ")
            f.write(f"{split['mean_small']:.1f}|{split['mean_large']:.1f} ")
            f.write(f"(std: {split['std_small']:.1f}|{split['std_large']:.1f}, ")
            f.write(f"range: [{split['min_small']}-{split['max_small']}]|[{split['min_large']}-{split['max_large']}])\n")

            # Partition validity
            valid = result['is_valid_partition']
            f.write(f"  {'Partition Validity':25s}: ")
            f.write(f"{valid['success_count']}/{valid['num_trials']} ")
            f.write(f"({valid['success_rate']*100:.1f}%)\n")

        f.write("\n" + "=" * 120 + "\n")

    print(f"✓ Detailed metrics table written to: {output_path}")
