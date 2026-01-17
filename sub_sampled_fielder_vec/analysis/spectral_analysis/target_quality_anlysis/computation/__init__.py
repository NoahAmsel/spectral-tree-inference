"""Computation modules for individual diagnostic metrics."""

from .coherence import compute_coherence
from .numerical_rank import compute_numerical_rank
from .partition import compute_partition_diagnostics
from .spectral_gaps import compute_spectral_gaps
from .eigenvalues import compute_eigenvalues_for_scree
from .computation_orchestrator import run_full_diagnostics
from .stability_runner import run_stability_diagnostics

__all__ = [
    'compute_coherence',
    'compute_numerical_rank',
    'compute_partition_diagnostics',
    'compute_spectral_gaps',
    'compute_eigenvalues_for_scree',
    'run_full_diagnostics',
    'run_stability_diagnostics'
]

