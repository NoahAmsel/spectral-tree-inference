"""Leverage score computation from rank-r SVD."""
import numpy as np
from typing import Tuple
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD


def compute_leverage_scores(X: np.ndarray, Omega: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute row and column leverage scores from rank-r SVD of observed entries.
    
    Phase 1 of leveraged sampling: Estimate importance of each row/column
    by computing leverage scores from a low-rank approximation.
    
    Args:
        X: Full matrix (n x n) - only entries in Omega are used
        Omega: Boolean mask (n x n) indicating observed entries
        r: Target rank for SVD (typically 2 for Fiedler vector)
        
    Returns:
        Tuple of (row_leverage_scores, column_leverage_scores)
        - row_leverage_scores: (n,) array with μ_i = (n/r) * ||U[i, :]||²
        - column_leverage_scores: (n,) array with ν_j = (n/r) * ||V[j, :]||²
    """
    n = X.shape[0]
    
    # Extract observed entries: P_Omega(X)
    X_observed = np.zeros_like(X)
    X_observed[Omega] = X[Omega]
    
    # Compute rank-r SVD
    # For large matrices, use TruncatedSVD for efficiency
    if n > 1000:
        # Use sklearn's TruncatedSVD (more memory efficient)
        svd_model = TruncatedSVD(n_components=r, random_state=42)
        # Fit on observed matrix (treat as dense for SVD)
        U = svd_model.fit_transform(X_observed)
        s = svd_model.singular_values_
        Vt = svd_model.components_
        V = Vt.T
    else:
        # Use full SVD for smaller matrices
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
    
    return row_leverage, col_leverage


def compute_sampling_probabilities(
    row_leverage: np.ndarray,
    col_leverage: np.ndarray,
    r: int,
    n: int
) -> np.ndarray:
    """
    Compute sampling probabilities p_ij from leverage scores.
    
    According to the paper: p_ij ∝ (μ_i + ν_j) * r * log²(n) / n
    
    Args:
        row_leverage: Row leverage scores μ_i (n,)
        col_leverage: Column leverage scores ν_j (n,)
        r: Target rank
        n: Matrix dimension
        
    Returns:
        Sampling probability matrix (n x n) with p_ij proportional to importance
    """
    # Compute unnormalized probabilities: p_ij ∝ (μ_i + ν_j) * r * log²(n) / n
    # Use broadcasting: (n, 1) + (1, n) = (n, n)
    log_n_sq = (np.log(n) ** 2) if n > 1 else 1.0
    scale_factor = (r * log_n_sq) / n
    
    # Broadcast: row_leverage[:, None] is (n, 1), col_leverage[None, :] is (1, n)
    p_unnormalized = (row_leverage[:, None] + col_leverage[None, :]) * scale_factor
    
    # For symmetric matrices, we only need upper triangle
    # But we'll compute full matrix and then extract upper triangle for sampling
    # Normalize to get a proper probability distribution
    # Note: We exclude diagonal (self-similarity is always 1.0)
    upper_triangle_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    p_upper = p_unnormalized * upper_triangle_mask
    
    # Normalize upper triangle probabilities to sum to 1
    p_sum = np.sum(p_upper)
    if p_sum > 0:
        p_upper = p_upper / p_sum
    else:
        # Fallback to uniform if all probabilities are zero
        n_upper = n * (n - 1) // 2
        p_upper = upper_triangle_mask.astype(float) / n_upper
    
    # Make symmetric (for consistency, though we'll sample from upper triangle)
    p_matrix = p_upper + p_upper.T
    
    return p_matrix

