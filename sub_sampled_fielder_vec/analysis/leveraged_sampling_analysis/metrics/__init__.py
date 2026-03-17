"""Metrics computation for leveraged sampling analysis.

This module provides functions to compute various metrics related to leveraged
matrix completion, including phase transitions, leverage score quality, and
matrix recovery quality.
"""

from .phase_transition import (
    compute_critical_p,
    fit_scaling_law,
    compute_phase_transitions_from_dataframe,
)

from .leverage import (
    compute_leverage_concentration,
    compute_phase1_quality,
    compute_leverage_efficiency,
)

from .recovery import (
    compute_recovery_success_rate,
    compute_sample_efficiency,
    compute_convergence_rate,
)

# Diagnostic metrics (optional - import explicitly if needed)
# from .diagnostics import *

__all__ = [
    # Phase transition
    "compute_critical_p",
    "fit_scaling_law",
    "compute_phase_transitions_from_dataframe",
    # Leverage
    "compute_leverage_concentration",
    "compute_phase1_quality",
    "compute_leverage_efficiency",
    # Recovery
    "compute_recovery_success_rate",
    "compute_sample_efficiency",
    "compute_convergence_rate",
]
