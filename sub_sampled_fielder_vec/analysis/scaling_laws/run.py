"""Main runner for scaling laws analysis."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import load_results, get_output_dir
from . import power_law_fitting, sample_requirements, phase_diagrams, write_report


def main():
    """Run complete scaling laws analysis."""
    print("=" * 80)
    print("SCALING LAWS: How Do Requirements Scale with Problem Size?")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_results()
    print(f"Loaded {len(df)} data points")

    # Get output directory
    output_dir = get_output_dir("scaling_laws")
    print(f"Output directory: {output_dir}")

    # Run analyses
    power_law_params = power_law_fitting.analyze(df, output_dir)
    sample_stats = sample_requirements.analyze(df, output_dir)
    phase_diagrams.analyze(df, output_dir)

    # Generate report
    write_report.generate_report(power_law_params, sample_stats, output_dir)

    print("\n" + "=" * 80)
    print("SCALING LAWS ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
