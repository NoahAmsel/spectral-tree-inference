"""Utilities for logging detailed sampling diagnostics."""
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from .logging import log_info, log_warning


def create_sparse_sampled_probs(
    sampled_indices: List[Tuple[int, int]],
    prob_matrix: np.ndarray,
    n: int
) -> np.ndarray:
    """
    Create sparse n×n matrix with probabilities only at sampled entries.

    Args:
        sampled_indices: List of (i,j) tuples that were sampled in Phase 2
        prob_matrix: Full (n,n) probability matrix used for sampling
        n: Matrix size

    Returns:
        Sparse (n,n) array with values only at sampled entries, zeros elsewhere
    """
    sparse_probs = np.zeros((n, n), dtype=np.float32)

    for i, j in sampled_indices:
        # Store probability at upper triangle position
        sparse_probs[i, j] = prob_matrix[i, j]
        # Mirror to lower triangle for symmetry
        sparse_probs[j, i] = prob_matrix[i, j]

    return sparse_probs


def save_sampling_diagnostics(
    run_dir: str,
    p_value: float,
    leverage_scores: np.ndarray,
    sampled_indices: List[Tuple[int, int]],
    prob_matrix: Optional[np.ndarray]
) -> None:
    """
    Save sampling diagnostics for a single p-value.

    Saves to: {run_dir}/sampling_data/p_{p_value:.4f}.npz

    Args:
        run_dir: Experiment output directory
        p_value: Sampling budget fraction
        leverage_scores: (n,) array of estimated leverage scores from Phase 1
        sampled_indices: List of (i,j) tuples sampled in Phase 2
        prob_matrix: (n,n) sampling probability matrix, or None if no Phase 2
    """
    # Create sampling_data subdirectory
    sampling_dir = Path(run_dir) / "sampling_data"
    sampling_dir.mkdir(exist_ok=True)

    # Create output filename
    filename = sampling_dir / f"p_{p_value:.4f}.npz"

    # Create sparse probability matrix (only non-zero at sampled entries)
    n = len(leverage_scores)
    if prob_matrix is not None and len(sampled_indices) > 0:
        phase2_probs_sampled = create_sparse_sampled_probs(
            sampled_indices, prob_matrix, n
        )
        nnz = len(sampled_indices) * 2  # Count symmetric pairs
    else:
        # No Phase 2 sampling occurred
        phase2_probs_sampled = np.zeros((n, n), dtype=np.float32)
        nnz = 0

    # Save compressed
    np.savez_compressed(
        filename,
        leverage_scores=leverage_scores.astype(np.float32),
        phase2_probs_sampled=phase2_probs_sampled
    )

    # Log success
    log_info('bootstrap',
        f"Saved sampling diagnostics: leverage_scores ({n},), "
        f"phase2_probs_sampled ({n}×{n}, {nnz} non-zero)"
    )
