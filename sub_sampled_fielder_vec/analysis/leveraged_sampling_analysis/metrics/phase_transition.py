"""Phase transition metrics for leveraged sampling analysis.

This module provides functions to compute phase transitions, including
critical sampling probabilities and scaling laws.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Import utilities from comparison module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
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
