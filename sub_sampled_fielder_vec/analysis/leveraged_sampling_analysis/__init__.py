"""Leveraged sampling analysis tools.

This module provides utilities for analyzing leveraged matrix completion sampling
experiments, including data loading, metrics computation, and visualization.
"""

from .io.data_loader import (
    load_experiment_results,
    load_comparison_results,
    load_comparison_dataframe,
    load_single_run_dataframe,
    get_available_runs,
    extract_config,
    to_dataframe,
)

from .metrics.phase_transition import (
    compute_critical_p,
    fit_scaling_law,
    compute_phase_transitions_from_dataframe,
)

from .metrics.leverage import (
    compute_leverage_concentration,
    compute_phase1_quality,
    compute_leverage_efficiency,
)

from .metrics.recovery import (
    compute_recovery_success_rate,
    compute_sample_efficiency,
    compute_convergence_rate,
)

from .visualization.agreement import (
    plot_agreement_vs_p,
    plot_merged_agreement_vs_p,
)

from .visualization.scaling import (
    plot_phase_transition_scaling,
)

from .visualization.leverage import (
    plot_leverage_diagnostics,
    plot_phase1_svd,
)

from .visualization.comparison import (
    plot_method_comparison_heatmap,
)

from .visualization.convergence import (
    plot_ialm_convergence,
)

# Diagnostic utilities (optional - import explicitly if needed)
# from .metrics.diagnostics import *
# from .visualization.diagnostics import *

__all__ = [
    # Data loading
    "load_experiment_results",
    "load_comparison_results",
    "load_comparison_dataframe",
    "load_single_run_dataframe",
    "get_available_runs",
    "extract_config",
    "to_dataframe",
    # Metrics - Phase transition
    "compute_critical_p",
    "fit_scaling_law",
    "compute_phase_transitions_from_dataframe",
    # Metrics - Leverage
    "compute_leverage_concentration",
    "compute_phase1_quality",
    "compute_leverage_efficiency",
    # Metrics - Recovery
    "compute_recovery_success_rate",
    "compute_sample_efficiency",
    "compute_convergence_rate",
    # Plotting - Agreement
    "plot_agreement_vs_p",
    "plot_merged_agreement_vs_p",
    # Plotting - Scaling
    "plot_phase_transition_scaling",
    # Plotting - Leverage
    "plot_leverage_diagnostics",
    "plot_phase1_svd",
    # Plotting - Comparison
    "plot_method_comparison_heatmap",
    # Plotting - Convergence
    "plot_ialm_convergence",
    # Note: Diagnostic functions are available via:
    # from analysis.leveraged_sampling_analysis.metrics.diagnostics import *
    # from analysis.leveraged_sampling_analysis.visualization.diagnostics import *
]
