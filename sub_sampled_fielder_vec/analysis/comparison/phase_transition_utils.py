"""Phase transition analysis utilities for comparing sampling methods.

Functions for:
- Loading comparison results
- Fitting sigmoid curves to agreement vs p
- Computing critical p* thresholds
- Power law fitting for scaling analysis
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import curve_fit
import pandas as pd


def load_comparison_data(comp_dir: Path) -> Dict[str, Dict[int, List[Tuple[float, float]]]]:
    """Load comparison results for both methods.

    Args:
        comp_dir: Comparison directory (e.g., results/XXX-method_comparison_YYY)

    Returns:
        Nested dict: {method: {n_taxa: [(p, agreement), ...]}}
    """
    comp_dir = Path(comp_dir)
    data = {}

    for method in ["uniform", "leveraged"]:
        method_dir = comp_dir / method
        if not method_dir.exists():
            raise FileNotFoundError(f"Method directory not found: {method_dir}")

        data[method] = {}

        # Find all n{X}_L{Y} subdirectories
        for subdir in sorted(method_dir.iterdir()):
            if not subdir.is_dir() or not subdir.name.startswith("n"):
                continue

            # Extract n from directory name (e.g., n512_L10000 -> 512)
            n_taxa = int(subdir.name.split("_")[0][1:])

            # Load results.json
            results_file = subdir / "results.json"
            if not results_file.exists():
                continue

            with results_file.open() as f:
                results = json.load(f)

            # Extract (p, partition_agreement_M) pairs
            p_agreement_pairs = []
            for row in results["rows"]:
                p = row["p"]
                agreement = row["partition_agreement_M"]
                p_agreement_pairs.append((p, agreement))

            # Sort by p
            p_agreement_pairs.sort(key=lambda x: x[0])
            data[method][n_taxa] = p_agreement_pairs

    return data


def sigmoid(p, L, k, p0):
    """Logistic sigmoid in log(p) space.

    Args:
        p: Sampling probability
        L: Maximum agreement (upper asymptote)
        k: Steepness
        p0: Inflection point (p where agreement = L/2)

    Returns:
        Agreement value
    """
    return L / (1 + np.exp(-k * (np.log(p) - np.log(p0))))


def fit_sigmoid_with_interpolation_fallback(p_vals: np.ndarray, agreements: np.ndarray,
                                             threshold: float = 95.0) -> Tuple[np.ndarray, float]:
    """Fit sigmoid or use interpolation fallback to find p* at threshold.

    Strategy:
    1. Try sigmoid fit
    2. If fit gives unreasonable p* (e.g., > 1.0 or < min(p)), use linear interpolation instead

    Args:
        p_vals: Sampling probabilities (sorted)
        agreements: Agreement values (0-100 scale)
        threshold: Target agreement (default 95%)

    Returns:
        (params, p_star): Sigmoid params (or NaN if interpolation used) and p* at threshold
    """
    # Try sigmoid fit first
    params, p_star_sigmoid = fit_sigmoid(p_vals, agreements, threshold)

    # Check if result is reasonable
    if (not np.isnan(p_star_sigmoid) and
        p_vals.min() <= p_star_sigmoid <= p_vals.max()):
        # Sigmoid fit succeeded and gave reasonable result
        return params, p_star_sigmoid

    # Fallback: use linear interpolation in log-space
    print(f"  Using interpolation fallback (sigmoid fit gave p*={p_star_sigmoid:.4f})")

    # Find points bracketing threshold
    below_threshold = agreements < threshold
    above_threshold = agreements >= threshold

    if not np.any(above_threshold):
        # Never reaches threshold - extrapolate from last two points
        if len(p_vals) >= 2:
            # Linear interpolation in log-log space
            log_p1, log_p2 = np.log(p_vals[-2]), np.log(p_vals[-1])
            agr1, agr2 = agreements[-2], agreements[-1]

            if agr2 > agr1:  # Increasing trend
                slope = (agr2 - agr1) / (log_p2 - log_p1)
                log_p_star = log_p2 + (threshold - agr2) / slope
                p_star = np.exp(log_p_star)
            else:
                p_star = np.nan
        else:
            p_star = np.nan
    elif not np.any(below_threshold):
        # Always above threshold - return minimum p
        p_star = p_vals.min()
    else:
        # Interpolate between last point below and first point above
        idx_below = np.where(below_threshold)[0][-1]
        idx_above = np.where(above_threshold)[0][0]

        p1, p2 = p_vals[idx_below], p_vals[idx_above]
        agr1, agr2 = agreements[idx_below], agreements[idx_above]

        # Linear interpolation in log(p) space
        log_p1, log_p2 = np.log(p1), np.log(p2)
        frac = (threshold - agr1) / (agr2 - agr1)
        log_p_star = log_p1 + frac * (log_p2 - log_p1)
        p_star = np.exp(log_p_star)

    return np.array([np.nan, np.nan, np.nan]), p_star


def fit_sigmoid(p_vals: np.ndarray, agreements: np.ndarray, threshold: float = 95.0) -> Tuple[np.ndarray, float]:
    """Fit logistic sigmoid to agreement vs p data.

    Args:
        p_vals: Sampling probabilities (should be sorted)
        agreements: Agreement values (0-100 scale)
        threshold: Target threshold for p* (default 95%)

    Returns:
        (params, p_star): Fitted parameters [L, k, p0] and p* where sigmoid = threshold
    """
    # Find a smarter initial guess for p0 (inflection point)
    # Look for p where agreement crosses threshold or is closest to it
    min_agr = agreements.min()
    max_agr = agreements.max()

    # Find where the transition happens
    idx_mid = np.argmin(np.abs(agreements - threshold))
    p0_init = p_vals[idx_mid]

    # Initial guess: L = max observed + buffer, k=10, p0 from data
    L_init = min(max(max_agr, threshold + 5), 105)
    initial_guess = [L_init, 5.0, p0_init]

    # Bounds: L in [threshold, 105], k > 0, p0 in [min_p, max_p]
    bounds = ([max(threshold, max_agr - 10), 0.1, p_vals.min() * 0.5],
              [105, 50, p_vals.max() * 2.0])

    try:
        params, _ = curve_fit(sigmoid, p_vals, agreements,
                             p0=initial_guess, bounds=bounds,
                             maxfev=10000)
        L, k, p0 = params

        # Find p* where sigmoid(p*) = threshold
        if L > threshold:
            p_star = p0 * np.exp(-np.log((L / threshold) - 1) / k)
        else:
            # Can't reach threshold
            p_star = np.nan

        return params, p_star

    except (RuntimeError, ValueError) as e:
        # Fit failed
        return np.array([np.nan, np.nan, np.nan]), np.nan


def find_discrete_threshold(p_vals: np.ndarray, agreements: np.ndarray, threshold: float = 100.0) -> Optional[float]:
    """Find first p where agreement >= threshold.

    Args:
        p_vals: Sampling probabilities (sorted)
        agreements: Agreement values
        threshold: Threshold value (default 100 for perfect agreement)

    Returns:
        First p >= threshold, or None if never reached
    """
    for p, agreement in zip(p_vals, agreements):
        if agreement >= threshold:
            return p
    return None


def compute_phase_transitions(data: Dict[str, Dict[int, List[Tuple[float, float]]]]) -> pd.DataFrame:
    """Compute phase transition p* for all (n, method) combinations.

    Args:
        data: Nested dict from load_comparison_data()

    Returns:
        DataFrame with columns: [n_taxa, method, p_star_sigmoid_95, p_star_discrete_100,
                                  sigmoid_L, sigmoid_k, sigmoid_p0]
    """
    rows = []

    for method in data:
        for n_taxa in sorted(data[method].keys()):
            pairs = data[method][n_taxa]
            p_vals = np.array([p for p, _ in pairs])
            agreements = np.array([a for _, a in pairs])

            print(f"Processing {method} n={n_taxa}...")

            # Sigmoid fit with interpolation fallback
            sigmoid_params, p_star_95 = fit_sigmoid_with_interpolation_fallback(
                p_vals, agreements, threshold=95.0)
            L, k, p0 = sigmoid_params

            # Discrete threshold
            p_star_100 = find_discrete_threshold(p_vals, agreements, threshold=100.0)

            rows.append({
                "n_taxa": n_taxa,
                "method": method,
                "p_star_sigmoid_95": p_star_95,
                "p_star_discrete_100": p_star_100 if p_star_100 is not None else np.nan,
                "sigmoid_L": L,
                "sigmoid_k": k,
                "sigmoid_p0": p0,
            })

    return pd.DataFrame(rows)


def fit_power_law(n_vals: np.ndarray, p_vals: np.ndarray) -> Tuple[float, float, str]:
    """Fit power law: p* = A * n^α

    Args:
        n_vals: Number of taxa
        p_vals: Critical p* values

    Returns:
        (exponent α, coefficient A, equation_string)
        equation_string: "p* = {A:.2e} × n^{α:.3f}"
    """
    # Fit in log-space: log(p*) = log(A) + α*log(n)
    # Remove NaN values
    mask = ~np.isnan(p_vals)
    n_clean = n_vals[mask]
    p_clean = p_vals[mask]

    if len(n_clean) < 2:
        return np.nan, np.nan, "Insufficient data"

    log_n = np.log(n_clean)
    log_p = np.log(p_clean)

    # Linear fit
    coeffs = np.polyfit(log_n, log_p, deg=1)
    alpha = coeffs[0]  # Slope
    log_A = coeffs[1]  # Intercept
    A = np.exp(log_A)

    equation = f"p* = {A:.2e} × n^{alpha:.3f}"

    return alpha, A, equation


def evaluate_power_law(n_vals: np.ndarray, alpha: float, A: float) -> np.ndarray:
    """Evaluate fitted power law at given n values.

    Args:
        n_vals: Number of taxa values
        alpha: Exponent
        A: Coefficient

    Returns:
        p* = A * n^α
    """
    return A * (n_vals ** alpha)
