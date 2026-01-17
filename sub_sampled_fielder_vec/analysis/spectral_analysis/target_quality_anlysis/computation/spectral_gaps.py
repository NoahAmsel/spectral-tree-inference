"""Compute spectral gaps of the Laplacian."""
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


def compute_spectral_gaps(M: np.ndarray) -> Dict[str, float]:
    """
    Compute spectral gaps of the Laplacian.

    Returns both absolute gap |λ₃ - λ₂| and relative gap |λ₃ - λ₂| / λ₂.

    Args:
        M: Similarity matrix (n × n)

    Returns:
        Dictionary with keys:
        - 'gap': |λ₃ - λ₂| (absolute spectral gap)
        - 'relative_gap': |λ₃ - λ₂| / λ₂
        - 'lambda2': Second smallest eigenvalue (Fiedler eigenvalue)
        - 'lambda3': Third smallest eigenvalue
    """
    # Compute Laplacian
    L_M = compute_laplacian(M)

    # Get smallest 3 eigenvalues
    eigvals, _ = scipy.linalg.eigh(L_M, subset_by_index=(0, 2))

    lambda2 = float(eigvals[1])
    lambda3 = float(eigvals[2])

    gap = abs(lambda3 - lambda2)
    relative_gap = gap / lambda2 if abs(lambda2) > 1e-12 else float('inf')

    return {
        'gap': gap,
        'relative_gap': relative_gap,
        'lambda2': lambda2,
        'lambda3': lambda3
    }

