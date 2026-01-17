"""Compute partition diagnostics (Fiedler vector, σ₂, partition split)."""
import numpy as np
from typing import Dict
from pathlib import Path
import sys

# Add parent directories to path for imports
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.core.utils import compute_laplacian
from src.utils.metrics import compute_reference_partition_and_quality
import scipy.linalg


def compute_partition_diagnostics(
    M: np.ndarray,
    num_gaps: int = 1,
    min_split: int = 2
) -> Dict[str, any]:
    """
    Compute Fiedler partition, σ₂ quality score, and partition split sizes.

    Args:
        M: Similarity matrix (n × n)
        num_gaps: Number of gap-based thresholds to evaluate
        min_split: Minimum partition size

    Returns:
        Dictionary with keys:
        - 'fiedler_vector': Second eigenvector of Laplacian
        - 'partition': Boolean array indicating partition membership
        - 'sigma2': Second singular value of cross-partition submatrix
        - 'partition_split': Tuple (n_small, n_large)
    """
    # Compute Laplacian
    L_M = compute_laplacian(M)

    # Get smallest eigenvalues/eigenvectors (Fiedler vector is 2nd)
    eigvals, eigvecs = scipy.linalg.eigh(L_M, subset_by_index=(0, 2))
    fiedler_vector = eigvecs[:, 1]  # Second eigenvector

    # Compute partition and quality
    partition, sigma2, partition_split = compute_reference_partition_and_quality(
        fiedler_vector, M, num_gaps=num_gaps, min_split=min_split
    )

    return {
        'fiedler_vector': fiedler_vector,
        'partition': partition,
        'sigma2': float(sigma2),
        'partition_split': partition_split
    }

