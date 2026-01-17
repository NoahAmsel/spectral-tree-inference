"""Compute sampling probabilities from leverage scores."""
import numpy as np


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
