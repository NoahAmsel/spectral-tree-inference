#!/usr/bin/env python3
"""Main CLI entry point for spectral phase transition analysis.

Usage:
    python -m spectral_analysis.sweep_params_analysis.run_analysis <results_dir>

Example:
    python -m spectral_analysis.sweep_params_analysis.run_analysis \\
        results/20251130-193600-balanced_tree_mu_01
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any

# Allow running the script directly via `python .../run_analysis.py`
if __package__ in {None, ""}:
    import sys

    PACKAGE_PARENT = Path(__file__).resolve().parents[2]
    if str(PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PARENT))

from spectral_analysis.sweep_params_analysis.utils import (
    load_results,
    group_by_config,
    save_outputs,
)
from spectral_analysis.sweep_params_analysis.phase_transition import (
    detect_critical_p,
    classify_regime,
    detect_eigenvalue_crossing,
)
from spectral_analysis.sweep_params_analysis.davis_kahan import verify_stability
from spectral_analysis.sweep_params_analysis.scaling_laws import fit_p_critical
from spectral_analysis.sweep_params_analysis.plots import (
    plot_phase_boundary,
    plot_eigenvalue_pop,
    plot_eigenvalue_pop_zoom,
    plot_spectral_gap,
    plot_eigenvalue_ratio,
    plot_stability_curve,
    plot_ipr_delocalization,
    plot_scaling_law_E1,
    plot_scaling_law_E2,
    plot_empirical_rank,
    plot_coherence,
)


def analyze_results(results_dir: Path) -> None:
    """Run complete spectral analysis pipeline.

    Args:
        results_dir: Path to experimental results directory
    """
    print(f"📊 Analyzing results from: {results_dir}")

    # 1. Load data
    rows = load_results(results_dir)
    print(f"✓ Loaded {len(rows)} data rows")

    # 2. Create output directory
    output_dir = save_outputs(results_dir)
    print(f"✓ Output directory: {output_dir}")

    # 3. Group by configuration
    grouped = group_by_config(rows)
    print(f"✓ Found {len(grouped)} unique configurations")

    # 4. Detect critical points and classify regimes
    critical_points = []
    regime_data = []
    p_crit_map = {}  # For Graph B
    eigenvalue_crossing_map = {}  # For Graph B

    for (N, L), config_rows in grouped.items():
        p_crit = detect_critical_p(config_rows)
        p_cross = detect_eigenvalue_crossing(config_rows)

        # Store for plotting
        if p_crit is not None:
            p_crit_map[(N, L)] = p_crit
        if p_cross is not None:
            eigenvalue_crossing_map[(N, L)] = p_cross

        for row in config_rows:
            regime = classify_regime(row["partition_agreement_M"])
            regime_data.append(
                {
                    "N": N,
                    "L": L,
                    "p": row["p"],
                    "partition_agreement_M": row["partition_agreement_M"],
                    "regime": regime,
                }
            )

        if p_crit is not None:
            critical_points.append({"N": N, "L": L, "p_crit": p_crit})

    print(f"✓ Detected {len(critical_points)} critical transitions")
    print(f"✓ Detected {len(eigenvalue_crossing_map)} eigenvalue crossings")

    # 5. Save regime classification
    regime_csv = output_dir / "regime_classification.csv"
    with open(regime_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["N", "L", "p", "partition_agreement_M", "regime"]
        )
        writer.writeheader()
        writer.writerows(regime_data)
    print(f"✓ Saved regime classification: {regime_csv}")

    # 6. Fit scaling laws
    if critical_points:
        L_fit, N_fit = fit_p_critical(critical_points)
        scaling_txt = output_dir / "scaling_law_fit.txt"
        with open(scaling_txt, "w") as f:
            f.write("Scaling Law Analysis: p_crit ~ f(N, L)\n")
            f.write("=" * 50 + "\n\n")

            if L_fit:
                f.write(f"p_crit vs L: p_crit = {L_fit['a']:.4e} * L^{L_fit['b']:.4f}\n")
                f.write(f"R² = {L_fit['r_squared']:.4f}\n\n")

            if N_fit:
                f.write(f"p_crit vs N: p_crit = {N_fit['a']:.4e} * N^{N_fit['b']:.4f}\n")
                f.write(f"R² = {N_fit['r_squared']:.4f}\n")

        print(f"✓ Saved scaling law fit: {scaling_txt}")

    # 7. Generate plots
    print("📈 Generating visualizations...")

    # Graph A: Phase boundary (single combined view)
    plot_phase_boundary(rows, output_dir / "A_phase_boundary.png")
    print("  ✓ Graph A: Phase boundary")

    # Graphs B, C, D: Grid layouts (all configurations)
    plot_eigenvalue_pop(
        grouped,
        output_dir / "B_eigenvalue_pop.png",
        p_crit_map,
        eigenvalue_crossing_map,
    )
    print("  ✓ Graph B: Eigenvalue trajectories (5×4 grid)")

    plot_eigenvalue_pop_zoom(
        grouped,
        output_dir / "B1_eigenvalue_pop_zoom.png",
        p_crit_map,
        eigenvalue_crossing_map,
    )
    print("  ✓ Graph B1: Eigenvalues near p_crit (zoom grid)")

    plot_spectral_gap(grouped, output_dir / "B2_spectral_gap.png", p_crit_map)
    print("  ✓ Graph B2: Spectral gap |λ₂ - λ₃| (5×4 grid)")

    plot_eigenvalue_ratio(grouped, output_dir / "B3_eigenvalue_ratio.png", p_crit_map)
    print("  ✓ Graph B3: Eigenvalue ratio λ₂/λ₃ (5×4 grid)")

    plot_stability_curve(grouped, output_dir / "C_stability_curve.png")
    print("  ✓ Graph C: Davis-Kahan stability (5×4 grid)")

    # Compute global x-axis limits for synchronization
    all_p_vals = [row["p"] for row in rows]
    xlim = (min(all_p_vals), max(all_p_vals))

    plot_ipr_delocalization(grouped, output_dir / "D_ipr_delocalization.png", xlim=xlim)
    print("  ✓ Graph D: IPR delocalization (5×4 grid)")

    # Graph E1: Scaling law - p_crit vs L (curves per N)
    plot_scaling_law_E1(critical_points, output_dir / "E1_scaling_law_vs_L.png")
    print("  ✓ Graph E1: Scaling law p_crit vs L")

    # Graph E2: Scaling law - p_crit vs n (curves per L)
    plot_scaling_law_E2(critical_points, output_dir / "E2_scaling_law_vs_n.png")
    print("  ✓ Graph E2: Scaling law p_crit vs n")

    # Graph F_1: Empirical rank recovery
    plot_empirical_rank(grouped, output_dir / "F1_empirical_rank.png", p_crit_map, xlim=xlim)
    print("  ✓ Graph F_1: Empirical rank recovery (5×4 grid)")

    # Graph F_2: Coherence analysis
    plot_coherence(grouped, output_dir / "F2_coherence.png", p_crit_map, xlim=xlim)
    print("  ✓ Graph F_2: Coherence (eigenvector localization, 5×4 grid)")

    print(f"\n✅ Analysis complete! Results saved to:\n   {output_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spectral phase transition analysis for subsampled Laplacians"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to results directory (containing results_grid_merged.json)",
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"❌ Error: Directory not found: {args.results_dir}")
        return 1

    try:
        analyze_results(args.results_dir)
        return 0
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
