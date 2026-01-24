"""Operator norm error scaling analysis utilities."""
import numpy as np
from scipy.optimize import curve_fit


def fit_power_law_scaling(p_values: np.ndarray, errors: np.ndarray) -> tuple:
    """Fit ||S-M|| ~ A * p^(-alpha). Returns (alpha, A, r_squared).
    
    Args:
        p_values: Array of sampling probabilities
        errors: Array of operator norm errors ||S - M||_op
        
    Returns:
        Tuple of (alpha, A, r_squared) where:
        - alpha: Power law exponent
        - A: Prefactor
        - r_squared: Coefficient of determination
    """
    # Remove NaN and invalid values
    mask = ~(np.isnan(p_values) | np.isnan(errors) | (p_values <= 0) | (errors <= 0))
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan
    
    p_clean = p_values[mask]
    err_clean = errors[mask]
    
    # Use log space for fitting: log(err) = log(A) - alpha * log(p)
    log_p = np.log(p_clean)
    log_err = np.log(err_clean)
    
    try:
        # Linear fit in log space
        coeffs = np.polyfit(log_p, log_err, 1)
        alpha = -coeffs[0]  # Negative because we expect err ~ p^(-alpha)
        log_A = coeffs[1]
        A = np.exp(log_A)
        
        # Compute R-squared
        log_err_pred = log_A - alpha * log_p
        ss_res = np.sum((log_err - log_err_pred) ** 2)
        ss_tot = np.sum((log_err - np.mean(log_err)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        return float(alpha), float(A), float(r_squared)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return np.nan, np.nan, np.nan


def compute_theoretical_bound(p, n: int):
    """Compute theoretical operator norm bound O(sqrt(n/p)).
    
    For random subsampling with probability p, the operator norm error
    scales as ||S - M||_op = O(sqrt(n/p)) in expectation.
    
    Args:
        p: Sampling probability (float or array)
        n: Matrix dimension (number of taxa)
        
    Returns:
        Theoretical bound value(s): sqrt(n / p)
    """
    p = np.asarray(p)
    if np.any(p <= 0) or n <= 0:
        result = np.full_like(p, np.nan, dtype=float)
        result[p > 0] = np.sqrt(n / p[p > 0])
        return result if p.ndim > 0 else float(result.item())
    
    result = np.sqrt(n / p)
    return result if p.ndim > 0 else float(result.item())
