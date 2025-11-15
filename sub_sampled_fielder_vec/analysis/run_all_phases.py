"""
Main runner script to execute all analysis phases sequentially.

Usage:
    python run_all_phases.py [--phase 1|2|3|all]
"""

import sys
import argparse
from pathlib import Path

# Import phase modules
import phase1_diagnostic_metrics
import phase2_scaling_laws
import phase3_theoretical_connection


def run_phase1():
    """Run Phase 1: Diagnostic Metrics Analysis"""
    print("\n" + "="*80)
    print("STARTING PHASE 1")
    print("="*80 + "\n")
    phase1_diagnostic_metrics.main()


def run_phase2():
    """Run Phase 2: Scaling Laws Analysis"""
    print("\n" + "="*80)
    print("STARTING PHASE 2")
    print("="*80 + "\n")
    phase2_scaling_laws.main()


def run_phase3():
    """Run Phase 3: Theoretical Interpretation"""
    print("\n" + "="*80)
    print("STARTING PHASE 3")
    print("="*80 + "\n")
    phase3_theoretical_connection.main()


def run_all():
    """Run all phases sequentially"""
    print("\n" + "="*80)
    print("RUNNING COMPLETE ANALYSIS PIPELINE")
    print("="*80 + "\n")

    run_phase1()
    run_phase2()
    run_phase3()

    print("\n" + "="*80)
    print("ALL PHASES COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    base_dir = Path(__file__).parent.parent / "results" / "combined_grid_search_results" / "analysis_outputs"
    print(f"  Phase 1: {base_dir / 'phase1'}")
    print(f"  Phase 2: {base_dir / 'phase2'}")
    print(f"  Phase 3: {base_dir / 'phase3'}")
    print("\nSee the phase*_summary.md files for detailed reports.")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run spectral tree reconstruction analysis phases"
    )
    parser.add_argument(
        '--phase',
        type=str,
        default='all',
        choices=['1', '2', '3', 'all'],
        help='Which phase to run (default: all)'
    )

    args = parser.parse_args()

    if args.phase == '1':
        run_phase1()
    elif args.phase == '2':
        run_phase2()
    elif args.phase == '3':
        run_phase3()
    elif args.phase == 'all':
        run_all()
    else:
        print(f"Unknown phase: {args.phase}")
        sys.exit(1)


if __name__ == "__main__":
    main()
