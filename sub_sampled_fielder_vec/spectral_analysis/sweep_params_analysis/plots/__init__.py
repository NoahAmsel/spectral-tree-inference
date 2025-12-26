"""Visualization functions for spectral phase transition analysis."""

from .phase_boundary import plot_phase_boundary
from .eigenvalue_pop import plot_eigenvalue_pop, plot_eigenvalue_pop_zoom
from .spectral_gap import plot_spectral_gap
from .eigenvalue_ratio import plot_eigenvalue_ratio
from .stability_curve import plot_stability_curve
from .ipr_delocalization import plot_ipr_delocalization
from .scaling_law import plot_scaling_law, plot_scaling_law_E1, plot_scaling_law_E2
from .empirical_rank import plot_empirical_rank
from .coherence import plot_coherence

__all__ = [
    "plot_phase_boundary",
    "plot_eigenvalue_pop",
    "plot_eigenvalue_pop_zoom",
    "plot_spectral_gap",
    "plot_eigenvalue_ratio",
    "plot_stability_curve",
    "plot_ipr_delocalization",
    "plot_scaling_law",
    "plot_scaling_law_E1",
    "plot_scaling_law_E2",
    "plot_empirical_rank",
    "plot_coherence",
]
