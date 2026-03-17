"""Soft thresholding operator for L1-norm proximal operator."""
import numpy as np


def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """
    Element-wise soft thresholding operator with symmetry preservation.

    S_tau(x) = sign(x) * max(|x| - tau, 0)

    Used for L1-norm proximal operator: prox_{tau||·||_1}(X)

    Args:
        X: Input matrix
        tau: Threshold parameter

    Returns:
        Soft-thresholded symmetric matrix
    """
    result = np.sign(X) * np.maximum(np.abs(X) - tau, 0)
    # Enforce symmetry for phylogenetic similarity matrices
    return (result + result.T) / 2
