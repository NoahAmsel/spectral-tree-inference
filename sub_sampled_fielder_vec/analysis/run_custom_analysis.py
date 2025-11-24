#!/usr/bin/env python3
"""Run theoretical interpretation analysis on a custom results directory."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from shared import load_results, get_output_dir
from theoretical_interpretation import eigenvalue_statistics, coherence_analysis, theoretical_bounds, tables, write_report


def run_custom_analysis(results_path: str):
    """
    Run theoretical interpretation analysis on custom results.

    Args:
        results_path: Path to results_grid.json file or directory containing it
    """
    # Convert to Path
    results_path = Path(results_path)

    # If it's a directory, look for results_grid.json
    if results_path.is_dir():
        results_file = results_path / "results_grid.json"
    else:
        results_file = results_path

    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return

    print("=" * 80)
    print(f"THEORETICAL INTERPRETATION: {results_file.parent.name}")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from: {results_file}")
    df = load_results(str(results_file))
    print(f"Loaded {len(df)} data points")

    # Create output directory in the same location as the results
    output_dir = results_file.parent / "analysis_outputs" / "theoretical_interpretation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Run analyses
    print("\n" + "=" * 80)
    tables_stats = tables.analyze(df, output_dir)
    eigenvalue_stats = eigenvalue_statistics.analyze(df, output_dir)
    coherence_stats = coherence_analysis.analyze(df, output_dir)
    bounds_stats = theoretical_bounds.analyze(df, output_dir)

    # Generate report
    write_report.generate_report(eigenvalue_stats, coherence_stats, bounds_stats, tables_stats, output_dir)

    print("\n" + "=" * 80)
    print("THEORETICAL INTERPRETATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_custom_analysis.py <results_path>")
        print("  results_path: Path to results_grid.json or directory containing it")
        sys.exit(1)

    run_custom_analysis(sys.argv[1])
