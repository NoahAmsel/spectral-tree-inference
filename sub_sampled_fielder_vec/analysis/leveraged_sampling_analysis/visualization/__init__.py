"""Visualization utilities for leveraged sampling analysis.

This module provides plotting functions for visualizing experiment results,
phase transitions, and leveraged-specific diagnostics.
"""

from .agreement import (
    plot_agreement_vs_p,
    plot_merged_agreement_vs_p,
)

from .scaling import (
    plot_phase_transition_scaling,
)

from .leverage import (
    plot_leverage_diagnostics,
    plot_phase1_svd,
)

from .comparison import (
    plot_method_comparison_heatmap,
)

from .convergence import (
    plot_ialm_convergence,
)

# Diagnostic plots (optional - import explicitly if needed)
# from .diagnostics import *

__all__ = [
    # Agreement plots
    "plot_agreement_vs_p",
    "plot_merged_agreement_vs_p",
    # Scaling plots
    "plot_phase_transition_scaling",
    # Leverage plots
    "plot_leverage_diagnostics",
    "plot_phase1_svd",
    # Comparison plots
    "plot_method_comparison_heatmap",
    # Convergence plots
    "plot_ialm_convergence",
]
