"""Fit scaling law for critical sampling rate p_crit ~ f(N, L)."""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.optimize import curve_fit


def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law: p_crit = a * x^b.

    Args:
        x: Independent variable (e.g., L or N)
        a: Scale coefficient
        b: Exponent

    Returns:
        Fitted values
    """
    return a * np.power(x, b)


def fit_p_critical(
    critical_points: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fit scaling laws p_crit vs L and p_crit vs N.

    We test the hypothesis:
    - p_crit ∝ L^β (does more data reduce sampling requirements?)
    - p_crit ∝ N^α (how does tree size affect requirements?)

    Args:
        critical_points: List of dicts with keys {N, L, p_crit}

    Returns:
        Tuple of (L_fit, N_fit) where each contains {a, b, r_squared}
    """
    if not critical_points:
        raise ValueError("No critical points provided")

    # Extract data
    N_vals = np.array([pt["N"] for pt in critical_points])
    L_vals = np.array([pt["L"] for pt in critical_points])
    p_crit_vals = np.array([pt["p_crit"] for pt in critical_points])

    # Fit p_crit vs L (holding N constant or averaging over N)
    unique_L = np.unique(L_vals)
    if len(unique_L) > 1:
        L_mean_p = []
        for L in unique_L:
            mask = L_vals == L
            L_mean_p.append(np.mean(p_crit_vals[mask]))

        (a_L, b_L), _ = curve_fit(power_law, unique_L, L_mean_p)
        predictions_L = power_law(unique_L, a_L, b_L)
        r_squared_L = 1 - np.sum((L_mean_p - predictions_L) ** 2) / np.sum(
            (L_mean_p - np.mean(L_mean_p)) ** 2
        )
        L_fit = {"a": a_L, "b": b_L, "r_squared": r_squared_L}
    else:
        L_fit = None

    # Fit p_crit vs N
    unique_N = np.unique(N_vals)
    if len(unique_N) > 1:
        N_mean_p = []
        for N in unique_N:
            mask = N_vals == N
            N_mean_p.append(np.mean(p_crit_vals[mask]))

        (a_N, b_N), _ = curve_fit(power_law, unique_N, N_mean_p)
        predictions_N = power_law(unique_N, a_N, b_N)
        r_squared_N = 1 - np.sum((N_mean_p - predictions_N) ** 2) / np.sum(
            (N_mean_p - np.mean(N_mean_p)) ** 2
        )
        N_fit = {"a": a_N, "b": b_N, "r_squared": r_squared_N}
    else:
        N_fit = None

    return L_fit, N_fit
