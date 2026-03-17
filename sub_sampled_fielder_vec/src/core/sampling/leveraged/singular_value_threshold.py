"""Singular value thresholding (SVT) operator for nuclear norm proximal operator."""
import numpy as np
import warnings
from typing import Tuple
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.sparse import issparse, csr_matrix


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

        # Performance optimization: Use sparse/truncated SVD for large sparse matrices
        # Heuristic: If matrix is >1000x1000 and <10% dense, use truncated SVD
        # This reduces O(n³) to O(n²k) where k is number of singular values kept
        n = X_sym.shape[0]
        sparsity = np.sum(X_sym != 0) / X_sym.size if X_sym.size > 0 else 0
        use_sparse_svd = (n > 1000) and (sparsity < 0.1)

        # Estimate target rank: For low-rank matrices, we expect rank r << n
        # Keep 2x expected rank to ensure we capture all significant singular values
        # For phylogenetic matrices, r=2 typically, so k=20 is conservative
        target_k = min(20, n - 2) if use_sparse_svd else n

        try:
            if use_sparse_svd and target_k > 0:
                # Use sparse SVD (scipy.sparse.linalg.svds)
                # Note: svds requires k < min(n,m), and returns singular values in ascending order
                X_sparse = csr_matrix(X_sym)
                U, s, Vt = svds(X_sparse, k=target_k)
                # svds returns singular values in ascending order - reverse them
                U = U[:, ::-1]
                s = s[::-1]
                Vt = Vt[::-1, :]
            else:
                # Use full dense SVD for small or dense matrices
                U, s, Vt = svd(X_sym, full_matrices=False)
        except (np.linalg.LinAlgError, ValueError, ArithmeticError):
            # SVD failed - return input with flag
            return X, True
        
        # Check for numerical issues in SVD output
        if np.any(~np.isfinite(s)) or np.any(~np.isfinite(U)) or np.any(~np.isfinite(Vt)):
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), True
        
        # Soft-threshold singular values
        s_thresh = np.maximum(s - tau, 0)

        # Effective rank truncation: Keep only significant singular values
        # Theory: Low-rank matrix should have rank r << n
        # Problem: Without truncation, numerical noise creates ~n small non-zero values
        # Solution: Truncate values below epsilon * max(s_thresh) or keep only top k
        #
        # Use aggressive threshold: 1% of maximum thresholded singular value
        # This ensures we keep r-10 truly significant components, not numerical noise
        epsilon = 0.01 * np.max(s_thresh) if np.max(s_thresh) > 0 else tau
        significant = s_thresh > epsilon
        effective_rank = np.sum(significant)

        if effective_rank > 0:
            # Reconstruct with only significant singular values
            s_trunc = s_thresh[significant]
            U_trunc = U[:, significant]
            Vt_trunc = Vt[significant, :]
            result = U_trunc @ np.diag(s_trunc) @ Vt_trunc
        else:
            # All singular values eliminated - return zero matrix
            result = np.zeros_like(X)
        
        # Final check
        if np.any(~np.isfinite(result)):
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), True
        
        return result, False
