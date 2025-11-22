"""Run all analyses in sequence.

This is the main entry point for the complete analysis pipeline.

Usage:
    python run_all.py
"""

import sys
from pathlib import Path

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent))

from metrics_analysis.run import main as run_metrics
from scaling_laws.run import main as run_scaling
from theoretical_interpretation.run import main as run_theory


def main():
    """Run all three analysis phases in sequence."""
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE ANALYSIS PIPELINE")
    print("=" * 80)
    print("\nThis will run three analysis phases:")
    print("  1. Metrics Analysis - What predicts the transition?")
    print("  2. Scaling Laws - How do requirements scale?")
    print("  3. Theoretical Interpretation - Why does it work?")
    print("\n" + "=" * 80 + "\n")

    try:
        # Phase 1: Metrics Analysis
        print("\n" + ">" * 40)
        print("PHASE 1/3: METRICS ANALYSIS")
        print(">" * 40)
        run_metrics()

        # Phase 2: Scaling Laws
        print("\n" + ">" * 40)
        print("PHASE 2/3: SCALING LAWS")
        print(">" * 40)
        run_scaling()

        # Phase 3: Theoretical Interpretation
        print("\n" + ">" * 40)
        print("PHASE 3/3: THEORETICAL INTERPRETATION")
        print(">" * 40)
        run_theory()

        # Final summary
        print("\n" + "=" * 80)
        print("[DONE] ALL ANALYSES COMPLETE")
        print("=" * 80)
        print("\nResults saved in:")
        print("  - analysis_results/metrics_analysis/")
        print("  - analysis_results/scaling_laws/")
        print("  - analysis_results/theoretical_interpretation/")
        print("\nSee the *_summary.md files in each directory for reports.")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
