"""Diagnostic metrics for investigating leveraged sampling performance.

Simple, focused metrics to answer specific questions about why leveraged
sampling might underperform uniform sampling.
"""
import numpy as np
from typing import Tuple, Optional


def compute_leverage_concentration(
    leverage_max: float, leverage_mean: float
) -> float:
    """Measure how concentrated leverage scores are.

    Question: Are leverage scores actually non-uniform?

    Args:
        leverage_max: Maximum leverage score
        leverage_mean: Mean leverage score

    Returns:
        Concentration ratio: max/mean
        - 1.0 = perfectly uniform (no concentration)
        - >> 1.0 = concentrated (some entries are much more important)
        - Good leveraged sampling should have ratio > 5
    """
    if leverage_mean == 0 or np.isnan(leverage_mean):
        return np.nan
    return leverage_max / leverage_mean


def compute_effective_rank(singular_values: np.ndarray) -> float:
    """Compute effective rank from singular values.

    Question: Is Phase 1 giving clean rank-2 structure?

    Args:
        singular_values: Array of singular values (sorted descending)

    Returns:
        Effective rank: (Σσᵢ)² / Σσᵢ²
        - Should be ≈ 2 for good rank-2 structure
        - Higher values indicate diffuse spectrum (bad for recovery)

    References:
        Vershynin, "High-Dimensional Probability" (2018)
    """
    s = np.array(singular_values)
    s = s[~np.isnan(s)]

    if len(s) == 0:
        return np.nan

    s_sum = np.sum(s)
    s_sum_sq = np.sum(s ** 2)

    if s_sum_sq == 0:
        return np.nan

    return (s_sum ** 2) / s_sum_sq


def compute_spectral_gap_ratio(s2: float, s3: float) -> float:
    """Measure spectral gap quality for rank-2 structure.

    Question: Is there a clear gap after σ₂?

    Args:
        s2: Second singular value
        s3: Third singular value

    Returns:
        Gap ratio: σ₂ / σ₃
        - Should be >> 1 (ideally > 10) for clean rank-2
        - Small ratio means no clear rank structure
    """
    if s3 == 0 or np.isnan(s3):
        return np.nan
    return s2 / s3


def compute_phase1_sample_fraction(
    p: float, n: int, target_rank: int = 2, theta: float = 0.3
) -> float:
    """Compute fraction of sample budget used for Phase 1 uniform sampling.

    Question: How much budget is wasted on Phase 1?

    Args:
        p: Total sampling probability
        n: Number of taxa (matrix dimension)
        target_rank: Target rank for Phase 1 SVD (typically 2)
        theta: Oversampling parameter (default 0.3)

    Returns:
        Fraction of samples used for Phase 1 (0.0 to 1.0+)
        - < 0.2: Phase 1 is small overhead (good)
        - 0.2-0.5: Moderate overhead
        - > 0.5: Phase 1 dominates budget (bad - just doing expensive uniform!)

    Note:
        Phase 1 uses O(nr log(n) / θ) uniform samples
        Total budget is p * n² samples
    """
    if p == 0 or n == 0:
        return np.nan

    # Phase 1 sample requirement: r * n * log(n) / theta
    phase1_samples = target_rank * n * np.log(n) / theta

    # Total sample budget
    total_samples = p * n * n

    if total_samples == 0:
        return np.nan

    fraction = phase1_samples / total_samples

    # Cap at 1.0 for visualization (can exceed 1.0 if p is very small)
    return fraction


def theoretical_p_star(
    n: int, r: int = 2, method: str = "uniform"
) -> float:
    """Compute theoretical minimum sampling probability from matrix completion theory.

    Question: What does theory predict for p*?

    Args:
        n: Number of taxa (matrix dimension)
        r: Matrix rank (typically 2 for Fiedler vector)
        method: "uniform" or "leveraged"

    Returns:
        Theoretical p*: C * r * log(n) / n
        - Uniform uses C ≈ 4 (conservative)
        - Leveraged should achieve C ≈ 2 (2x better)

    References:
        - Candès & Recht, "Exact Matrix Completion via Convex Optimization" (2009)
        - Achlioptas & McSherry, "Fast Computation of Low-Rank Approximations" (2007)
    """
    if n <= 1:
        return np.nan

    # Conservative constants from theory
    C_uniform = 4.0
    C_leveraged = 2.0

    C = C_uniform if method == "uniform" else C_leveraged

    # Sample complexity: p ≥ C * r * log(n) / n
    p_theory = C * r * np.log(n) / n

    return p_theory


def compute_leverage_coefficient_of_variation(
    leverage_std: float, leverage_mean: float
) -> float:
    """Measure relative spread of leverage scores.

    Question: How variable are leverage scores?

    Args:
        leverage_std: Standard deviation of leverage scores
        leverage_mean: Mean leverage score

    Returns:
        Coefficient of variation: std/mean
        - 0 = all scores identical (uniform)
        - > 0.5 = high variation (concentrated distribution)
    """
    if leverage_mean == 0 or np.isnan(leverage_mean):
        return np.nan
    return leverage_std / leverage_mean


def diagnose_phase1_quality(
    s1: float, s2: float, s3: float, target_rank: int = 2
) -> Tuple[str, str]:
    """Give human-readable verdict on Phase 1 SVD quality.

    Args:
        s1, s2, s3: First three singular values
        target_rank: Expected rank (typically 2)

    Returns:
        (status, message) where status is "GOOD", "WARNING", or "BAD"
    """
    if np.isnan(s1) or np.isnan(s2) or np.isnan(s3):
        return "UNKNOWN", "Singular values contain NaN"

    if s1 == 0:
        return "BAD", "σ₁ = 0 (degenerate matrix)"

    # Check rank-2 quality
    rank_ratio = s2 / s1 if s1 > 0 else 0
    gap_ratio = s2 / s3 if s3 > 0 else np.inf

    if rank_ratio < 0.01:
        return "BAD", f"σ₂/σ₁ = {rank_ratio:.4f} (too small, no rank-2 structure)"

    if gap_ratio < 3:
        return "BAD", f"σ₂/σ₃ = {gap_ratio:.2f} (no spectral gap, unclear rank)"

    if gap_ratio < 10:
        return "WARNING", f"σ₂/σ₃ = {gap_ratio:.2f} (weak spectral gap)"

    return "GOOD", f"σ₂/σ₁ = {rank_ratio:.4f}, σ₂/σ₃ = {gap_ratio:.2f} (clean rank-2)"


def diagnose_sample_allocation(phase1_fraction: float) -> Tuple[str, str]:
    """Give human-readable verdict on sample allocation efficiency.

    Args:
        phase1_fraction: Fraction of budget used for Phase 1 (from compute_phase1_sample_fraction)

    Returns:
        (status, message) where status is "GOOD", "WARNING", or "BAD"
    """
    if np.isnan(phase1_fraction):
        return "UNKNOWN", "Phase 1 fraction is NaN"

    if phase1_fraction > 0.5:
        return "BAD", f"Phase 1 uses {phase1_fraction:.1%} of budget (too much overhead!)"

    if phase1_fraction > 0.2:
        return "WARNING", f"Phase 1 uses {phase1_fraction:.1%} of budget (moderate overhead)"

    return "GOOD", f"Phase 1 uses {phase1_fraction:.1%} of budget (small overhead)"


def compare_to_theoretical_bound(
    actual_p_star: float, theoretical_p_star: float
) -> Tuple[str, str]:
    """Compare actual p* to theoretical prediction.

    Args:
        actual_p_star: Observed critical sampling probability
        theoretical_p_star: Theoretical minimum from matrix completion theory

    Returns:
        (status, message) where status is "GOOD", "WARNING", or "BAD"
    """
    if np.isnan(actual_p_star) or np.isnan(theoretical_p_star):
        return "UNKNOWN", "Cannot compare (NaN values)"

    ratio = actual_p_star / theoretical_p_star

    if ratio < 1.0:
        # Actual is better than theory - might be lucky or theory is too conservative
        return "GOOD", f"Actual p* is {ratio:.2f}x theoretical (better than predicted!)"

    if ratio < 3.0:
        return "GOOD", f"Actual p* is {ratio:.2f}x theoretical (reasonable overhead)"

    if ratio < 10.0:
        return "WARNING", f"Actual p* is {ratio:.2f}x theoretical (high overhead)"

    return "BAD", f"Actual p* is {ratio:.2f}x theoretical (far from optimal!)"
