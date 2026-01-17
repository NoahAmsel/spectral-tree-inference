"""Non-uniform sampling from upper triangle based on probability distribution."""
import numpy as np


def nonuniform_sample_upper_triangle(
    n: int,
    budget: int,
    p_matrix: np.ndarray,
    rng,
    Omega_1: np.ndarray = None
) -> np.ndarray:
    """
    Sample entries from upper triangle according to probability distribution.
    
    Args:
        n: Matrix dimension
        budget: Number of entries to sample
        p_matrix: Probability matrix (n x n) - only upper triangle is used
        rng: Random number generator
        Omega_1: Phase 1 observation set (n x n boolean mask). If provided,
                sampling is restricted to Omega_1^c (complement of Phase 1).
        
    Returns:
        Boolean mask (n x n) with True for sampled entries
    """
    # Extract upper triangle probabilities
    upper_triangle_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    p_upper = p_matrix * upper_triangle_mask
    
    # Restrict to Omega_1^c (complement of Phase 1 observations)
    if Omega_1 is not None:
        p_upper[Omega_1] = 0.0
    
    # Flatten to 1D for sampling
    p_flat = p_upper[upper_triangle_mask]
    indices_flat = np.where(upper_triangle_mask)
    
    # Normalize probabilities
    p_flat = p_flat / np.sum(p_flat) if np.sum(p_flat) > 0 else p_flat
    
    # Sample according to probabilities (with replacement if budget > available)
    n_upper = len(p_flat)
    if budget >= n_upper:
        # Sample all entries
        sampled_idx = np.arange(n_upper)
    else:
        if hasattr(rng, 'choice'):
            sampled_idx = rng.choice(n_upper, size=budget, replace=False, p=p_flat)
        else:
            sampled_idx = np.random.choice(n_upper, size=budget, replace=False, p=p_flat)
    
    # Create boolean mask
    Omega = np.zeros((n, n), dtype=bool)
    for idx in sampled_idx:
        i, j = indices_flat[0][idx], indices_flat[1][idx]
        Omega[i, j] = True
        Omega[j, i] = True  # Symmetric
    
    return Omega
