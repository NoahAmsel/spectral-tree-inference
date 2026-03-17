"""Compute matrix coherence metric."""
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path for imports
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.core.metric_computer import MetricComputer


def compute_coherence(M: np.ndarray, k: int = 2) -> float:
    """
    Compute matrix coherence: max_i ||u_i||_∞^2.

    Args:
        M: Similarity matrix (n × n)
        k: Number of top singular vectors to consider

    Returns:
        Maximum coherence value across top-k singular vectors
    """
    metric_computer = MetricComputer(coherence_k=k)

    # Compute top k eigenvalues/eigenvectors
    eigvecs, _ = metric_computer._compute_largest_eigenvalues(M, k=k)

    # Compute coherence from eigenvectors
    coherence = metric_computer._compute_coherence(eigvecs)

    return float(coherence)

