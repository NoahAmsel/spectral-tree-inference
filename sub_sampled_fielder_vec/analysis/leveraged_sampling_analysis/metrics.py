"""Metrics computation for leveraged sampling analysis.

This module provides functions to compute various metrics related to leveraged
matrix completion, including phase transitions, leverage score quality, and
matrix recovery quality.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.optimize import curve_fit

# Import utilities from comparison module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from analysis.comparison.phase_transition_utils import (
    fit_sigmoid_with_interpolation_fallback,
    fit_power_law,
    evaluate_power_law,
    find_discrete_threshold,
)


def compute_critical_p(
    p_vals: np.ndarray,
    agreements: np.ndarray,
    threshold: float = 95.0,
) -> Tuple[float, np.ndarray]:
    """Find critical p* using sigmoid/interpolation.

    Args:
        p_vals: Sampling probabilities (sorted)
        agreements: Agreement values (0-100 scale)
        threshold: Target agreement threshold (default 95%)

    Returns:
        (p_star, sigmoid_params): Critical p* and sigmoid parameters [L, k, p0]
    """
    params, p_star = fit_sigmoid_with_interpolation_fallback(
        p_vals, agreements, threshold=threshold
    )
    return p_star, params


def fit_scaling_law(
    n_vals: np.ndarray, p_star_vals: np.ndarray
) -> Tuple[float, float, str, np.ndarray]:
    """Fit power law: p* = A * n^α

    Args:
        n_vals: Number of taxa values
        p_star_vals: Critical p* values

    Returns:
        (alpha, A, equation_string, fitted_curve)
        equation_string: "p* = {A:.2e} × n^{α:.3f}"
        fitted_curve: Evaluated power law at n_vals
    """
    alpha, A, equation = fit_power_law(n_vals, p_star_vals)

    # Evaluate fitted curve
    if not np.isnan(alpha) and not np.isnan(A):
        fitted_curve = evaluate_power_law(n_vals, alpha, A)
    else:
        fitted_curve = np.full_like(n_vals, np.nan)

    return alpha, A, equation, fitted_curve


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


def compute_recovery_success_rate(
    agreement_vals: np.ndarray, threshold: float = 95.0
) -> float:
    """Compute fraction of experiments reaching agreement threshold.

    Args:
        agreement_vals: Array of agreement values
        threshold: Target agreement threshold (default 95%)

    Returns:
        Fraction of values >= threshold (0.0 to 1.0)
    """
    if len(agreement_vals) == 0:
        return 0.0

    valid_vals = agreement_vals[~np.isnan(agreement_vals)]
    if len(valid_vals) == 0:
        return 0.0

    return np.mean(valid_vals >= threshold)


def compute_sample_efficiency(
    p_uniform_star: float, p_leveraged_star: float
) -> Tuple[float, float]:
    """Compute sample efficiency savings ratio.

    Args:
        p_uniform_star: Critical p* for uniform sampling
        p_leveraged_star: Critical p* for leveraged sampling

    Returns:
        (savings_ratio, improvement_factor)
        savings_ratio: (p_uniform - p_leveraged) / p_uniform (fraction saved)
        improvement_factor: p_uniform / p_leveraged (how many times better)
    """
    if np.isnan(p_uniform_star) or np.isnan(p_leveraged_star):
        return np.nan, np.nan

    if p_uniform_star == 0:
        return np.nan, np.nan

    savings_ratio = (p_uniform_star - p_leveraged_star) / p_uniform_star

    if p_leveraged_star == 0:
        improvement_factor = np.inf
    else:
        improvement_factor = p_uniform_star / p_leveraged_star

    return savings_ratio, improvement_factor


def compute_convergence_rate(
    ialm_iterations: np.ndarray, operator_norm_error: np.ndarray
) -> Tuple[float, float]:
    """Track IALM solver performance.

    Args:
        ialm_iterations: Array of IALM iteration counts
        operator_norm_error: Array of operator norm errors

    Returns:
        (mean_iterations, mean_error)
    """
    valid_iter = ialm_iterations[~np.isnan(ialm_iterations)]
    valid_error = operator_norm_error[~np.isnan(operator_norm_error)]

    mean_iter = np.mean(valid_iter) if len(valid_iter) > 0 else np.nan
    mean_error = np.mean(valid_error) if len(valid_error) > 0 else np.nan

    return mean_iter, mean_error


def compute_phase_transitions_from_dataframe(
    df: pd.DataFrame, method_col: Optional[str] = None
) -> pd.DataFrame:
    """Compute phase transitions from DataFrame.

    Args:
        df: DataFrame with columns: p, partition_agreement_M, num_taxa
        method_col: Optional column name for method (if comparing multiple methods)

    Returns:
        DataFrame with phase transition metrics for each (n_taxa, method) combination
    """
    rows = []

    if method_col and method_col in df.columns:
        groups = df.groupby(["num_taxa", method_col])
        for (n_taxa, method_val), group_df in groups:
            p_vals = group_df["p"].values
            agreements = group_df["partition_agreement_M"].values

            # Sort by p
            sort_idx = np.argsort(p_vals)
            p_vals = p_vals[sort_idx]
            agreements = agreements[sort_idx]

            # Compute critical p*
            p_star_95, sigmoid_params = compute_critical_p(p_vals, agreements, threshold=95.0)
            L, k, p0 = sigmoid_params

            # Discrete threshold
            p_star_100 = find_discrete_threshold(p_vals, agreements, threshold=100.0)

            row = {
                "n_taxa": n_taxa,
                "p_star_sigmoid_95": p_star_95,
                "p_star_discrete_100": p_star_100 if p_star_100 is not None else np.nan,
                "sigmoid_L": L,
                "sigmoid_k": k,
                "sigmoid_p0": p0,
                method_col: method_val,
            }
            rows.append(row)
    else:
        groups = df.groupby("num_taxa")
        for n_taxa, group_df in groups:
            p_vals = group_df["p"].values
            agreements = group_df["partition_agreement_M"].values

            # Sort by p
            sort_idx = np.argsort(p_vals)
            p_vals = p_vals[sort_idx]
            agreements = agreements[sort_idx]

            # Compute critical p*
            p_star_95, sigmoid_params = compute_critical_p(p_vals, agreements, threshold=95.0)
            L, k, p0 = sigmoid_params

            # Discrete threshold
            p_star_100 = find_discrete_threshold(p_vals, agreements, threshold=100.0)

            row = {
                "n_taxa": n_taxa,
                "p_star_sigmoid_95": p_star_95,
                "p_star_discrete_100": p_star_100 if p_star_100 is not None else np.nan,
                "sigmoid_L": L,
                "sigmoid_k": k,
                "sigmoid_p0": p0,
            }
            rows.append(row)

    return pd.DataFrame(rows)
