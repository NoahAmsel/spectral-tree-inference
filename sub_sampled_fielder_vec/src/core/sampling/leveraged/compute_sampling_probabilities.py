"""Compute sampling probabilities from leverage scores."""
import numpy as np


def compute_sampling_probabilities(
    row_leverage: np.ndarray,
    col_leverage: np.ndarray,
    r: int,
    n: int,
    prob_formula: str = "additive"
) -> np.ndarray:
    """
    Compute sampling probabilities p_ij from leverage scores.

    Args:
        row_leverage: Row leverage scores μ_i (n,)
        col_leverage: Column leverage scores ν_j (n,)
        r: Target rank
        n: Matrix dimension
        prob_formula: Formula for combining row/col leverage scores:
            'additive': p_ij ∝ (μ_i + ν_j) — HLDT paper default
            'multiplicative': p_ij ∝ μ_i × ν_j — concentrates on high-leverage pairs
            'max': p_ij ∝ max(μ_i, ν_j)

    Returns:
        Sampling probability matrix (n x n) with p_ij proportional to importance
    """
    log_n_sq = (np.log(n) ** 2) if n > 1 else 1.0
    scale_factor = (r * log_n_sq) / n

    if prob_formula == "multiplicative":
        p_unnormalized = (row_leverage[:, None] * col_leverage[None, :]) * scale_factor
    elif prob_formula == "max":
        p_unnormalized = np.maximum(row_leverage[:, None], col_leverage[None, :]) * scale_factor
    else:  # "additive" (default, HLDT paper formula)
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
