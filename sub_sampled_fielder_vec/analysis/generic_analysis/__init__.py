"""Numerical linear algebra diagnostics for phase transition analysis."""

from .eigenvalue_analysis import extract_eigenvalues, compute_relative_gap
from .davis_kahan import find_dk_crossing, check_stability
from .scaling_analysis import fit_power_law_scaling, compute_theoretical_bound
from .fiedler_diagnostics import compute_ipr, analyze_sign_stability
from .matrix_quality import extract_numerical_rank, extract_coherence
from .plotting import plot_metric_vs_p, plot_summary_diagnostics
from .phase_transition_plotting import plot_phase_transition_scaling
from .operator_norm_plotting import plot_operator_norm_scaling, plot_operator_norm_scaling_grouped
from .grouped_plotting import group_data_by_model_and_n, plot_metric_by_model_and_n
from .eigenvalue_plotting import plot_eigenvalue_spectrum_grouped
from .davis_kahan_plotting import plot_davis_kahan_grouped
from .fiedler_quality_plotting import plot_fiedler_quality_grouped
from .rank_coherence_plotting import plot_numerical_rank_grouped, plot_coherence_grouped
from .summary_plotting import plot_summary_diagnostics_grouped

__all__ = [
    'extract_eigenvalues',
    'compute_relative_gap',
    'find_dk_crossing',
    'check_stability',
    'fit_power_law_scaling',
    'compute_theoretical_bound',
    'compute_ipr',
    'analyze_sign_stability',
    'extract_numerical_rank',
    'extract_coherence',
    'plot_metric_vs_p',
    'plot_summary_diagnostics',
    'plot_phase_transition_scaling',
    'plot_operator_norm_scaling',
    'plot_operator_norm_scaling_grouped',
    'group_data_by_model_and_n',
    'plot_metric_by_model_and_n',
    'plot_eigenvalue_spectrum_grouped',
    'plot_davis_kahan_grouped',
    'plot_fiedler_quality_grouped',
    'plot_numerical_rank_grouped',
    'plot_coherence_grouped',
    'plot_summary_diagnostics_grouped',
]
