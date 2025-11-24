"""Main runner for theoretical interpretation analysis."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import load_results, get_output_dir
from . import eigenvalue_statistics, coherence_analysis, theoretical_bounds, tables, write_report


def main():
    """Run complete theoretical interpretation analysis."""
    print("=" * 80)
    print("THEORETICAL INTERPRETATION: Why Does the Transition Occur?")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_results()
    print(f"Loaded {len(df)} data points")

    # Get output directory
    output_dir = get_output_dir("theoretical_interpretation")
    print(f"Output directory: {output_dir}")

    # Run analyses
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
    main()
