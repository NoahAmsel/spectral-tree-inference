"""Main runner for metrics analysis."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import load_results, get_output_dir
from . import spectral_gap, rank_recovery, frobenius_error, quality_gap, transition_table, write_report


def main():
    """Run complete metrics analysis."""
    print("=" * 80)
    print("METRICS ANALYSIS: What Predicts the Transition?")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_results()
    print(f"Loaded {len(df)} data points")

    # Get output directory
    output_dir = get_output_dir("metrics_analysis")
    print(f"Output directory: {output_dir}")

    # Run analyses (each is simple and focused)
    gap_stats = spectral_gap.analyze(df, output_dir)
    rank_stats = rank_recovery.analyze(df, output_dir)
    frobenius_error.analyze(df, output_dir)
    quality_gap_stats = quality_gap.analyze(df, output_dir)
    transition_table.create_table(df, output_dir)

    # Generate report
    write_report.generate_report(gap_stats, rank_stats, quality_gap_stats, output_dir)

    print("\n" + "=" * 80)
    print("METRICS ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
