#!/usr/bin/env python3
"""Main CLI entry point for stability analysis (K-trial experiments).

Usage:
    python -m spectral_analysis.target_quality_anlysis.cli.stability_analysis_main <config.json> [--num-trials K]

Example:
    # Run with default K=10 trials
    python -m spectral_analysis.target_quality_anlysis.cli.stability_analysis_main config_template.json

    # Run with 30 trials for better statistical confidence
    python -m spectral_analysis.target_quality_anlysis.cli.stability_analysis_main config_template.json --num-trials 30
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directories to path for imports
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from spectral_analysis.target_quality_anlysis.computation.stability_runner import run_stability_diagnostics
from spectral_analysis.target_quality_anlysis.output.stability_tables import (
    write_stability_table,
    write_stability_summary,
    write_detailed_metrics_table
)
from .config import load_config
from .directory import create_output_directory


def run_stability_analysis(config_path: Path, num_trials: int = 10) -> None:
    """
    Run stability analysis with K trials per configuration.

    Args:
        config_path: Path to configuration JSON file
        num_trials: Number of independent trials to run per config (K)
    """
    print("=" * 80)
    print("STABILITY ANALYSIS - Statistical Diagnostics Across K Trials")
    print("=" * 80)
    print()

    # Load configuration
    print(f"📄 Loading configuration from: {config_path}")
    config = load_config(config_path)
    experiment_name = config.get('experiment_name', 'stability_analysis')
    print(f"✓ Experiment: {experiment_name}")
    print(f"✓ Number of trials per configuration: {num_trials}")
    print()

    # Create output directory
    output_dir = create_output_directory(f"{experiment_name}_stability_K{num_trials}")
    print(f"📁 Output directory: {output_dir}")
    print()

    # Save config to output directory for reference
    config_copy_path = output_dir / "config.json"
    config_with_trials = {**config, 'num_trials': num_trials}
    with open(config_copy_path, 'w') as f:
        json.dump(config_with_trials, f, indent=2)
    print(f"✓ Configuration saved to: {config_copy_path}")
    print()

    # Extract analysis parameters
    analysis_params = config.get('analysis_params', {})
    num_gaps = analysis_params.get('num_gaps', 1)
    min_split = analysis_params.get('min_split', 2)
    k_scree = analysis_params.get('k_scree', 20)  # Not used for plots, but passed through

    # Run stability diagnostics for all model × config combinations
    print("🔬 Running stability diagnostics...")
    print()

    results = []
    total_combinations = len(config['models']) * len(config['configs'])
    current = 0

    for model_spec in config['models']:
        model_name = model_spec.get('name', model_spec['tree']['model'])

        for exp_config in config['configs']:
            current += 1
            n = exp_config['n']
            L = exp_config['L']
            mu = exp_config['mu']

            print(f"[{current}/{total_combinations}] Processing: {model_name}, n={n}, L={L}, μ={mu}")

            # Prepare tree config
            tree_config = {
                'model': model_spec['tree']['model'],
                'params': {
                    'num_taxa': n,
                    **model_spec['tree']['params']
                }
            }

            # Prepare sequence config
            seq_config = {
                'model': model_spec['sequence']['model'],
                'len': L,
                'params': {
                    'mutation_rate': mu,
                    **model_spec['sequence']['params']
                }
            }

            # Run stability diagnostics (K trials)
            try:
                result = run_stability_diagnostics(
                    tree_config=tree_config,
                    seq_config=seq_config,
                    num_trials=num_trials,
                    num_gaps=num_gaps,
                    min_split=min_split,
                    k_scree=k_scree,
                    verbose=True
                )
                # Override tree_model with the user-defined model name
                result['tree_model'] = model_name
                results.append(result)

            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        print()

    if not results:
        print("❌ No results generated. Exiting.")
        return

    print(f"✓ Completed {len(results)} stability analyses")
    print()

    # Generate tables
    print("📊 Generating stability tables...")
    table_path = output_dir / "stability_table.txt"
    write_stability_table(results, table_path)

    summary_path = output_dir / "stability_summary.txt"
    write_stability_summary(results, summary_path)

    detailed_path = output_dir / "detailed_metrics.txt"
    write_detailed_metrics_table(results, detailed_path)
    print()

    # Final summary
    print("=" * 80)
    print("✅ STABILITY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total configurations analyzed: {len(results)}")
    print(f"Trials per configuration: {num_trials}")
    print(f"Results saved to: {output_dir}")
    print()
    print("Generated files:")
    print(f"  - {table_path.name}       (main results table with mean±std)")
    print(f"  - {summary_path.name}     (cross-config summary statistics)")
    print(f"  - {detailed_path.name}    (detailed metrics with min/max)")
    print()

    # Highlight any concerning patterns
    print("🔍 Quick Insights:")

    # Check for configs with low partition validity success rates
    low_validity_configs = [
        r for r in results
        if r['is_valid_partition']['success_rate'] < 0.5
    ]

    if low_validity_configs:
        print(f"  ⚠️  {len(low_validity_configs)} config(s) have <50% partition validity success rate:")
        for r in low_validity_configs:
            success_rate = r['is_valid_partition']['success_rate'] * 100
            print(f"      - {r['tree_model']}, n={r['n']}, L={r['L']}, μ={r['mu']}: {success_rate:.1f}% valid")
    else:
        print(f"  ✓ All configs have ≥50% partition validity success rate")

    # Check for high variability in sigma2
    high_var_sigma2 = [
        r for r in results
        if r['sigma2']['std'] / (r['sigma2']['mean'] + 1e-10) > 0.2  # CV > 20%
    ]

    if high_var_sigma2:
        print(f"  ⚠️  {len(high_var_sigma2)} config(s) show high variability in σ₂ (CV > 20%):")
        for r in high_var_sigma2:
            cv = r['sigma2']['std'] / (r['sigma2']['mean'] + 1e-10)
            print(f"      - {r['tree_model']}, n={r['n']}, L={r['L']}, μ={r['mu']}: CV={cv:.2f}")
    else:
        print(f"  ✓ All configs have stable σ₂ values (CV ≤ 20%)")

    print()


def main():
    """Parse arguments and run stability analysis."""
    parser = argparse.ArgumentParser(
        description="Stability Analysis - Run K trials to assess statistical variability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default K=10 trials
    python -m spectral_analysis.target_quality_anlysis.cli.stability_analysis_main config_template.json

    # Run with 30 trials for better statistical confidence
    python -m spectral_analysis.target_quality_anlysis.cli.stability_analysis_main config_template.json --num-trials 30

    # Run with 100 trials for publication-quality results
    python -m spectral_analysis.target_quality_anlysis.cli.stability_analysis_main config.json --num-trials 100
        """
    )

    parser.add_argument(
        'config',
        type=Path,
        help='Path to configuration JSON file'
    )

    parser.add_argument(
        '--num-trials',
        type=int,
        default=10,
        help='Number of independent trials to run per configuration (default: 10)'
    )

    args = parser.parse_args()

    # Validate num_trials
    if args.num_trials < 1:
        print(f"❌ Error: --num-trials must be ≥ 1 (got {args.num_trials})")
        sys.exit(1)

    if args.num_trials > 1000:
        print(f"⚠️  Warning: {args.num_trials} trials is a lot. This will take a while...")

    # Resolve config path (handle relative paths)
    if args.config.is_absolute():
        config_path = args.config
    else:
        # Try multiple locations: current directory, script directory, and script's parent
        script_dir = Path(__file__).parent
        script_parent = script_dir.parent
        
        potential_paths = [
            (Path.cwd() / args.config).resolve(),
            (script_dir / args.config).resolve(),
            (script_parent / args.config).resolve(),
        ]
        
        config_path = None
        for path in potential_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            # Use the first path for error message
            config_path = potential_paths[0]

    # Validate config file exists
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {args.config}")
        print(f"   Current working directory: {Path.cwd()}")
        print(f"   Searched in:")
        print(f"     - {Path.cwd() / args.config}")
        print(f"     - {Path(__file__).parent / args.config}")
        print(f"     - {Path(__file__).parent.parent / args.config}")
        sys.exit(1)

    # Use resolved path
    args.config = config_path

    # Run analysis
    try:
        run_stability_analysis(args.config, args.num_trials)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
