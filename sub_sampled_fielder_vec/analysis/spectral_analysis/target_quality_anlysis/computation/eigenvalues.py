"""Compute eigenvalues for scree plots."""
import numpy as np
from typing import Dict
from pathlib import Path
import sys

# Add parent directories to path for imports
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.core.utils import compute_laplacian
import scipy.linalg


def compute_eigenvalues_for_scree(M: np.ndarray, k: int = 20) -> Dict[str, np.ndarray]:
    """
    Compute eigenvalues for scree plots.

    For similarity matrix M: largest k eigenvalues (descending)
    For Laplacian L_M: smallest k eigenvalues (ascending)

    Args:
        M: Similarity matrix (n × n)
        k: Number of eigenvalues to compute

    Returns:
        Dictionary with keys:
        - 'M_eigenvalues': Largest k eigenvalues of M (descending)
        - 'L_M_eigenvalues': Smallest k eigenvalues of L_M (ascending)
    """
    n = M.shape[0]
    k = min(k, n - 1)

    # Compute largest k eigenvalues of M
    eigvals_M, _ = scipy.linalg.eigh(M, subset_by_index=(n - k, n - 1))
    eigvals_M = eigvals_M[::-1]  # Reverse to get descending order

    # Compute Laplacian and its smallest k eigenvalues
    L_M = compute_laplacian(M)
    eigvals_L_M, _ = scipy.linalg.eigh(L_M, subset_by_index=(0, k - 1))

    return {
        'M_eigenvalues': eigvals_M,
        'L_M_eigenvalues': eigvals_L_M
    }

