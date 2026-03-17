"""Recovery and convergence metrics for leveraged sampling analysis.

This module provides functions to compute metrics related to matrix recovery
quality and solver convergence.
"""
import numpy as np
from typing import Tuple


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
