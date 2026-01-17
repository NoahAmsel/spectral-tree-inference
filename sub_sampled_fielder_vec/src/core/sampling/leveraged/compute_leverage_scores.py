"""Compute leverage scores from rank-r SVD of observed entries."""
import numpy as np
from typing import Tuple
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def compute_leverage_scores(X: np.ndarray, Omega: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute row and column leverage scores from rank-r SVD of observed entries.
    
    Phase 1 of leveraged sampling: Estimate importance of each row/column
    by computing leverage scores from a low-rank approximation.
    
    Args:
        X: Full matrix (n x n) - only entries in Omega are used
        Omega: Boolean mask (n x n) indicating observed entries
        r: Target rank for SVD (typically 2 for Fiedler vector)
        
    Returns:
        Tuple of (row_leverage_scores, column_leverage_scores, singular_values)
        - row_leverage_scores: (n,) array with μ_i = (n/r) * ||U[i, :]||²
        - column_leverage_scores: (n,) array with ν_j = (n/r) * ||V[j, :]||²
        - singular_values: (r,) array with top r singular values from Phase 1 SVD
    """
    n = X.shape[0]

    # Extract observed entries: P_Omega(X)
    X_observed = np.zeros_like(X)
    X_observed[Omega] = X[Omega]

    # Convert to sparse matrix for efficient SVD (at low p, matrix is 99%+ zeros)
    X_sparse = csr_matrix(X_observed)
    
    # Compute rank-r SVD
    # For large matrices, use TruncatedSVD for efficiency
    if n > 1000:
        # Use sklearn's TruncatedSVD (supports sparse matrices efficiently)
        svd_model = TruncatedSVD(n_components=r, random_state=42)
        # Fit on sparse matrix (much faster for low-density matrices)
        U = svd_model.fit_transform(X_sparse)
        s = svd_model.singular_values_
        Vt = svd_model.components_
        V = Vt.T
    else:
        # For small matrices, use dense SVD (scipy.linalg.svd doesn't support sparse)
        U, s, Vt = svd(X_observed, full_matrices=False)
        # Truncate to rank r
        U = U[:, :r]
        s = s[:r]
        Vt = Vt[:r, :]
        V = Vt.T
    
    # Compute row leverage scores: μ_i = (n/r) * ||U[i, :]||²
    # U is (n x r), so we compute row-wise L2 norms
    row_norms_sq = np.sum(U ** 2, axis=1)  # (n,)
    row_leverage = (n / r) * row_norms_sq
    
    # Compute column leverage scores: ν_j = (n/r) * ||V[j, :]||²
    # V is (n x r), so we compute row-wise L2 norms
    col_norms_sq = np.sum(V ** 2, axis=1)  # (n,)
    col_leverage = (n / r) * col_norms_sq

    # Also return singular values for diagnostic tracking
    return row_leverage, col_leverage, s[:r]
