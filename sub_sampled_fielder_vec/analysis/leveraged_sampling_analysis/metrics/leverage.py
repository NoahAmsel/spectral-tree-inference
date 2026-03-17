"""Leverage score metrics for leveraged sampling analysis.

This module provides functions to compute metrics related to leverage scores,
including concentration, Phase 1 quality, and efficiency.
"""
import numpy as np
from typing import Tuple


def compute_leverage_concentration(
    leverage_max: float, leverage_sum: float, n: int
) -> float:
    """Measure uniformity of leverage scores.

    Lower values indicate more uniform leverage scores.
    Theoretical minimum is 1/n (perfectly uniform).

    Args:
        leverage_max: Maximum leverage score
        leverage_sum: Sum of all leverage scores
        n: Number of taxa (matrix dimension)

    Returns:
        Concentration ratio: leverage_max / (leverage_sum / n)
        Values close to 1 indicate uniform distribution
    """
    if leverage_sum == 0 or n == 0:
        return np.nan

    expected_uniform = leverage_sum / n
    if expected_uniform == 0:
        return np.nan

    return leverage_max / expected_uniform


def compute_phase1_quality(
    s1: float, s2: float, s3: float, target_rank: int = 2
) -> Tuple[float, float]:
    """Assess SVD quality from Phase 1 singular values.

    Args:
        s1, s2, s3: First three singular values from Phase 1
        target_rank: Expected rank (typically 2 for Fiedler vector)

    Returns:
        (rank_quality, gap_quality)
        rank_quality: Ratio s2/s1 (should be significant for rank-2 structure)
        gap_quality: Ratio s2/s3 (larger gap indicates clearer rank-2 structure)
    """
    if s1 == 0:
        return np.nan, np.nan

    rank_quality = s2 / s1 if s1 > 0 else np.nan
    gap_quality = s2 / s3 if s3 > 0 else np.nan

    return rank_quality, gap_quality


def compute_leverage_efficiency(
    leverage_sum: float, expected_rank: int, n: int
) -> float:
    """Compare leverage sum to theoretical bounds.

    For a rank-r matrix, sum of leverage scores should be approximately r.
    This function computes the ratio to assess efficiency.

    Args:
        leverage_sum: Sum of all leverage scores
        expected_rank: Expected rank of the matrix (typically 2)
        n: Number of taxa

    Returns:
        Efficiency ratio: leverage_sum / expected_rank
        Values close to 1 indicate good leverage score estimation
    """
    if expected_rank == 0:
        return np.nan

    return leverage_sum / expected_rank
