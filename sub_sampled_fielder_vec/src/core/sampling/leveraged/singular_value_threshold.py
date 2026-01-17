"""Singular value thresholding (SVT) operator for nuclear norm proximal operator."""
import numpy as np
import warnings
from typing import Tuple
from scipy.linalg import svd


def singular_value_threshold(X: np.ndarray, tau: float) -> Tuple[np.ndarray, bool]:
    """
    Singular value thresholding (SVT) operator.
    
    Proximal operator for nuclear norm: prox_{tau||·||_*}(X)
    
    Algorithm:
        1. Compute SVD: X = U Σ V^T
        2. Soft-threshold singular values: Σ_tau = max(Σ - tau, 0)
        3. Reconstruct: X_tau = U Σ_tau V^T
    
    Args:
        X: Input matrix (n x n)
        tau: Threshold parameter
        
    Returns:
        Tuple of (thresholded matrix, numerical_issue_flag)
    """
    # Suppress all numpy warnings during SVT - we handle issues explicitly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Check for numerical issues in input
        if np.any(~np.isfinite(X)):
            # Replace NaN/Inf with zeros to allow graceful degradation
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return X, True

        # Enforce symmetry before SVD (critical for phylogenetic matrices)
        X_sym = (X + X.T) / 2

        try:
            U, s, Vt = svd(X_sym, full_matrices=False)
        except (np.linalg.LinAlgError, ValueError):
            # SVD failed - return input with flag
            return X, True
        
        # Check for numerical issues in SVD output
        if np.any(~np.isfinite(s)) or np.any(~np.isfinite(U)) or np.any(~np.isfinite(Vt)):
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), True
        
        # Soft-threshold singular values
        s_thresh = np.maximum(s - tau, 0)
        
        # Reconstruct
        result = U @ np.diag(s_thresh) @ Vt
        
        # Final check
        if np.any(~np.isfinite(result)):
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), True
        
        return result, False
