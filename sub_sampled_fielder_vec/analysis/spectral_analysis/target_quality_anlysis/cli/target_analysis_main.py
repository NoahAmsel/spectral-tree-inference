#!/usr/bin/env python3
"""Main CLI entry point for target quality analysis.

Usage:
    python -m spectral_analysis.target_quality_anlysis.cli.target_analysis_main <config.json>

Example:
    python -m spectral_analysis.target_quality_anlysis.cli.target_analysis_main config_template.json
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directories to path for imports
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from spectral_analysis.target_quality_anlysis.computation.computation_orchestrator import run_full_diagnostics
from spectral_analysis.target_quality_anlysis.output.tables import (
    write_diagnostics_table,
    write_summary_stats
)
from spectral_analysis.target_quality_anlysis.visualization.scree_plots import generate_all_scree_plots
from spectral_analysis.target_quality_anlysis.visualization.coherence_plots import plot_coherence_comparison
from spectral_analysis.target_quality_anlysis.visualization.heatmaps import plot_metric_heatmaps
from spectral_analysis.target_quality_anlysis.visualization.tree_plots import (
    plot_tree_with_partition,
    plot_combined_tree_and_fiedler
)
from .config import load_config
from .directory import create_output_directory


def run_analysis(config_path: Path) -> None:
    """
    Run complete target quality analysis pipeline.
    
    Args:
        config_path: Path to configuration JSON file
    """
    print("=" * 80)
    print("TARGET QUALITY ANALYSIS - Full Similarity Matrix Diagnostics")
    print("=" * 80)
    print()

    # Load configuration
    print(f"📄 Loading configuration from: {config_path}")
    config = load_config(config_path)
    experiment_name = config.get('experiment_name', 'target_analysis')
    print(f"✓ Experiment: {experiment_name}")
    print()

    # Create output directories
    output_dir = create_output_directory(experiment_name)
    tree_plots_dir = output_dir / "tree_plots"
    tree_plots_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print(f"🌳 Tree plots will be saved in: {tree_plots_dir}")
    print()

    # Save config to output directory for reference
    config_copy_path = output_dir / "config.json"
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved to: {config_copy_path}")
    print()

    # Extract analysis parameters
    analysis_params = config.get('analysis_params', {})
    num_gaps = analysis_params.get('num_gaps', 1)
    min_split = analysis_params.get('min_split', 2)
    k_scree = analysis_params.get('k_scree', 20)

    # Run diagnostics for all model × config combinations
    print("🔬 Running diagnostics...")
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

            # Run full diagnostics
            try:
                result = run_full_diagnostics(
                    tree_config=tree_config,
                    seq_config=seq_config,
                    num_gaps=num_gaps,
                    min_split=min_split,
                    k_scree=k_scree
                )
                # Override tree_model with the user-defined model name for proper subdirectory creation
                result['tree_model'] = model_name
                results.append(result)
                print(f"  ✓ Coherence: {result['coherence']:.6f}, NumRank: {result['numerical_rank']:.2f}, σ₂: {result['sigma2']:.4f}")
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        print()

    if not results:
        print("❌ No results generated. Exiting.")
        return

    print(f"✓ Completed {len(results)} diagnostic runs")
    print()

    # Generate tables
    print("📊 Generating tables...")
    table_path = output_dir / "diagnostics_table.txt"
    write_diagnostics_table(results, table_path)

    summary_path = output_dir / "summary_statistics.txt"
    write_summary_stats(results, summary_path)
    print()

    # Generate plots
    print("📈 Generating scree plots...")
    generate_all_scree_plots(results, output_dir)
    print()

    print("📊 Generating coherence plot...")
    plot_coherence_comparison(results, output_dir)
    print()
    
    print("🌳 Generating tree partition plots...")
    for result in results:
        if result.get('tree') and result.get('partition_mask') is not None:
            model_name = result['tree_model']
            model_plot_dir = tree_plots_dir / model_name
            model_plot_dir.mkdir(exist_ok=True)

            # Format mu without decimal point to avoid .with_suffix() issues
            mu_str = f"{result['mu']:.2f}".replace('.', '_')
            base_path = model_plot_dir / f"H_partition_n{result['n']}_L{result['L']}_mu{mu_str}"
            title_str = f"Partition for {model_name} (n={result['n']}, L={result['L']}, μ={result['mu']:.2f})"

            # Generate circular tree + standalone Fiedler plot
            plot_tree_with_partition(
                tree=result['tree'],
                partition_mask=result['partition_mask'],
                fiedler_vector=result['fiedler_vector'],
                output_path=base_path,
                title=title_str,
                stats_dict=result
            )

            # Generate combined side-by-side plot
            plot_combined_tree_and_fiedler(
                tree=result['tree'],
                partition_mask=result['partition_mask'],
                fiedler_vector=result['fiedler_vector'],
                output_path=base_path,
                title=title_str,
                stats_dict=result
            )
    print()


    print("🗺  Generating spectral gap heatmaps...")
    plot_metric_heatmaps(
        results=results,
        output_dir=output_dir,
        metric_key='spectral_gap',
        metric_label='Spectral Gap (|λ₃ - λ₂|)',
        output_prefix='F_gap_heatmaps',
        cmap='magma',
        value_format=".1f"
    )
    print()

    print("🗺  Generating numerical rank heatmaps...")
    plot_metric_heatmaps(
        results=results,
        output_dir=output_dir,
        metric_key='numerical_rank',
        metric_label='Numerical Rank',
        output_prefix='G_numrank_heatmaps',
        cmap='YlGnBu',
        value_format=".2f"
    )
    print()

    # Final summary
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total configurations analyzed: {len(results)}")
    print(f"Results saved to: {output_dir}")
    print()
    print("Generated files:")
    print(f"  - {table_path.name}")
    print(f"  - {summary_path.name}")
    print(f"  - A_similarity_eigenvalues.png (scree plot)")
    print(f"  - B_laplacian_eigenvalues.png (scree plot)")
    print(f"  - C_coherence.png (bar chart)")
    print(f"  - F_gap_heatmaps.png (spectral gap heatmaps grid)")
    print(f"  - G_numrank_heatmaps.png (numerical rank heatmaps grid)")
    print(f"  - tree_plots/<model_name>/H_partition_...tree.pdf (circular phylogenetic trees)")
    print(f"  - tree_plots/<model_name>/H_partition_...fiedler.pdf (Fiedler vector bar plots)")
    print(f"  - tree_plots/<model_name>/H_partition_...combined.pdf (combined tree + Fiedler plots)")
    print(f"  - tree_plots/<model_name>/H_partition_...nwk/txt (raw data files)")
    print()



def main():
    """Parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Target Quality Analysis - Analyze full similarity matrices before subsampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with template config
    python -m spectral_analysis.target_quality_anlysis.cli.target_analysis_main config_template.json

    # Run with custom config
    python -m spectral_analysis.target_quality_anlysis.cli.target_analysis_main my_custom_config.json
        """
    )

    parser.add_argument(
        'config',
        type=Path,
        help='Path to configuration JSON file'
    )

    args = parser.parse_args()

    # Resolve config path (handle relative paths)
    if args.config.is_absolute():
        config_path = args.config
    else:
        config_path = (Path.cwd() / args.config).resolve()
    
    # Validate config file exists
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {config_path}")
        print(f"   Current working directory: {Path.cwd()}")
        print(f"   Tried path: {config_path}")
        sys.exit(1)
    
    # Use resolved path
    args.config = config_path

    # Run analysis
    try:
        run_analysis(args.config)
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
