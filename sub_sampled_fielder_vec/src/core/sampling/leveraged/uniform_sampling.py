"""Uniform sampling from upper triangle of symmetric matrix."""
import numpy as np


def uniform_sample_upper_triangle(
    n: int,
    budget: int,
    rng
) -> np.ndarray:
    """
    Uniformly sample entries from upper triangle.
    
    Args:
        n: Matrix dimension
        budget: Number of entries to sample
        rng: Random number generator
        
    Returns:
        Boolean mask (n x n) with True for sampled entries
    """
    # Get all upper triangle indices (excluding diagonal)
    upper_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_indices.append((i, j))
    
    # Sample without replacement
    if budget >= len(upper_indices):
        # Sample all entries
        sampled_indices = upper_indices
    else:
        if hasattr(rng, 'choice'):
            sampled_idx = rng.choice(len(upper_indices), size=budget, replace=False)
        else:
            sampled_idx = np.random.choice(len(upper_indices), size=budget, replace=False)
        sampled_indices = [upper_indices[i] for i in sampled_idx]
    
    # Create boolean mask
    Omega = np.zeros((n, n), dtype=bool)
    for i, j in sampled_indices:
        Omega[i, j] = True
        Omega[j, i] = True  # Symmetric
    
    return Omega
