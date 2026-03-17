"""Generate formatted text tables for diagnostic results.

Simple text formatting with fixed-width columns for easy readability.
"""
from typing import List, Dict
from pathlib import Path


def write_diagnostics_table(results: List[Dict], output_path: Path) -> None:
    """
    Write all diagnostics to a formatted .txt table.

    Args:
        results: List of diagnostic result dictionaries
        output_path: Path to output .txt file
    """
    # Column headers and widths
    headers = [
        ("Model", 15),
        ("n", 6),
        ("L", 6),
        ("mu", 6),
        ("Coherence", 11),
        ("NumRank", 10),
        ("Sigma2", 10),
        ("Partition", 12),
        ("Valid", 6),
        ("Gap", 10),
        ("RelGap", 10),
        ("Lambda2", 10),
        ("Lambda3", 10)
    ]

    # Create separator line
    separator = "─" * sum(width for _, width in headers)

    # Open file for writing
    with open(output_path, 'w') as f:
        # Write header
        f.write("Target Quality Analysis - Full Similarity Matrix Diagnostics\n")
        f.write("=" * len(separator) + "\n\n")

        # Write column headers
        header_line = ""
        for name, width in headers:
            header_line += f"{name:<{width}}"
        f.write(header_line + "\n")
        f.write(separator + "\n")

        # Write data rows
        for result in results:
            # Extract values
            model = result['tree_model']
            n = result['n']
            L = result['L']
            mu = result['mu']
            coherence = result['coherence']
            num_rank = result['numerical_rank']
            sigma2 = result['sigma2']
            partition_split = result['partition_split']
            is_valid = result.get('is_valid_partition', False)
            gap = result['spectral_gap']
            rel_gap = result['relative_spectral_gap']
            lambda2 = result['lambda2']
            lambda3 = result['lambda3']

            # Format partition split
            partition_str = f"{partition_split[0]}|{partition_split[1]}"
            valid_str = "Yes" if is_valid else "No"

            # Format row with proper widths
            row = ""
            row += f"{model:<15}"
            row += f"{n:<6}"
            row += f"{L:<6}"
            row += f"{mu:<6.2f}"
            row += f"{coherence:<11.6f}"
            row += f"{num_rank:<10.2f}"
            row += f"{sigma2:<10.4f}"
            row += f"{partition_str:<12}"
            row += f"{valid_str:<6}"
            row += f"{gap:<10.6f}"
            row += f"{rel_gap:<10.6f}"
            row += f"{lambda2:<10.6f}"
            row += f"{lambda3:<10.6f}"

            f.write(row + "\n")

        # Footer
        f.write(separator + "\n")
        f.write(f"\nTotal configurations analyzed: {len(results)}\n")

    print(f"✓ Diagnostics table written to: {output_path}")


def write_summary_stats(results: List[Dict], output_path: Path) -> None:
    """
    Write summary statistics across all configurations.

    Args:
        results: List of diagnostic result dictionaries
        output_path: Path to output .txt file
    """
    import numpy as np

    # Collect metrics
    coherences = [r['coherence'] for r in results]
    num_ranks = [r['numerical_rank'] for r in results]
    sigma2s = [r['sigma2'] for r in results]
    gaps = [r['spectral_gap'] for r in results]
    rel_gaps = [r['relative_spectral_gap'] for r in results]

    with open(output_path, 'w') as f:
        f.write("Summary Statistics Across All Configurations\n")
        f.write("=" * 60 + "\n\n")

        metrics = [
            ("Coherence", coherences),
            ("Numerical Rank", num_ranks),
            ("Sigma2", sigma2s),
            ("Spectral Gap", gaps),
            ("Relative Spectral Gap", rel_gaps)
        ]

        for name, values in metrics:
            f.write(f"{name}:\n")
            f.write(f"  Mean:   {np.mean(values):10.6f}\n")
            f.write(f"  Median: {np.median(values):10.6f}\n")
            f.write(f"  Min:    {np.min(values):10.6f}\n")
            f.write(f"  Max:    {np.max(values):10.6f}\n")
            f.write(f"  Std:    {np.std(values):10.6f}\n")
            f.write("\n")

    print(f"✓ Summary statistics written to: {output_path}")

