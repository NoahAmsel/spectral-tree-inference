"""Leveraged sampling analysis tools.

This module provides utilities for analyzing leveraged matrix completion sampling
experiments, including data loading, metrics computation, and visualization.
"""

from .data_loader import (
    load_experiment_results,
    load_comparison_results,
    load_comparison_dataframe,
    get_available_runs,
    extract_config,
    to_dataframe,
)

from .metrics import (
    compute_critical_p,
    fit_scaling_law,
    compute_phase_transitions_from_dataframe,
    compute_leverage_concentration,
    compute_phase1_quality,
    compute_leverage_efficiency,
    compute_recovery_success_rate,
    compute_sample_efficiency,
    compute_convergence_rate,
)

from .plotting import (
    plot_agreement_vs_p,
    plot_phase_transition_scaling,
    plot_leverage_diagnostics,
    plot_phase1_svd,
    plot_method_comparison_heatmap,
    plot_ialm_convergence,
)

__all__ = [
    # Data loading
    "load_experiment_results",
    "load_comparison_results",
    "load_comparison_dataframe",
    "get_available_runs",
    "extract_config",
    "to_dataframe",
    # Metrics
    "compute_critical_p",
    "fit_scaling_law",
    "compute_phase_transitions_from_dataframe",
    "compute_leverage_concentration",
    "compute_phase1_quality",
    "compute_leverage_efficiency",
    "compute_recovery_success_rate",
    "compute_sample_efficiency",
    "compute_convergence_rate",
    # Plotting
    "plot_agreement_vs_p",
    "plot_phase_transition_scaling",
    "plot_leverage_diagnostics",
    "plot_phase1_svd",
    "plot_method_comparison_heatmap",
    "plot_ialm_convergence",
]
