"""Compute numerical rank metric."""
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path for imports
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.core.metric_computer import MetricComputer


def compute_numerical_rank(M: np.ndarray) -> float:
    """
    Compute numerical rank: NumRank(M) = ||M||_F^2 / ||M||_2^2.

    Args:
        M: Similarity matrix (n × n)

    Returns:
        Numerical rank (continuous value, not integer)
    """
    metric_computer = MetricComputer()

    # Get largest eigenvalue (spectral norm)
    _, singular_values = metric_computer._compute_largest_eigenvalues(M, k=1)
    largest_singular_value = singular_values[0]

    # Compute numerical rank
    num_rank = metric_computer._compute_numerical_rank(M, largest_singular_value)

    return float(num_rank)

