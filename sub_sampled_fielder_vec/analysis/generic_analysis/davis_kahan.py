"""Davis-Kahan perturbation analysis utilities."""
import numpy as np
from scipy.interpolate import interp1d


def find_dk_crossing(p_values: np.ndarray, dk_ratios: np.ndarray, threshold: float = 1.0) -> float:
    """Find p where Davis-Kahan ratio crosses threshold (interpolated).
    
    Args:
        p_values: Array of sampling probabilities
        dk_ratios: Array of Davis-Kahan ratios
        threshold: Threshold value to find crossing (default: 1.0)
        
    Returns:
        Interpolated p value where dk_ratio crosses threshold.
        Returns np.nan if no crossing found or insufficient data.
    """
    if len(p_values) < 2 or len(dk_ratios) < 2:
        return np.nan
    
    # Remove NaN values
    mask = ~(np.isnan(p_values) | np.isnan(dk_ratios))
    if np.sum(mask) < 2:
        return np.nan
    
    p_clean = p_values[mask]
    dk_clean = dk_ratios[mask]
    
    # Check if threshold is crossed
    if np.all(dk_clean < threshold) or np.all(dk_clean > threshold):
        return np.nan
    
    # Find crossing point by interpolation
    try:
        # Sort by p for interpolation
        sort_idx = np.argsort(p_clean)
        p_sorted = p_clean[sort_idx]
        dk_sorted = dk_clean[sort_idx]
        
        # Interpolate
        interp_func = interp1d(dk_sorted, p_sorted, kind='linear', 
                               bounds_error=False, fill_value=np.nan)
        p_crossing = interp_func(threshold)
        
        return float(p_crossing) if not np.isnan(p_crossing) else np.nan
    except (ValueError, TypeError):
        return np.nan


def check_stability(dk_ratio: float, threshold: float = 0.5) -> bool:
    """Check if DK ratio indicates stable eigenvector recovery.
    
    Args:
        dk_ratio: Davis-Kahan ratio value
        threshold: Stability threshold (default: 0.5)
        
    Returns:
        True if stable (dk_ratio < threshold), False otherwise
    """
    if np.isnan(dk_ratio) or np.isinf(dk_ratio):
        return False
    
    return dk_ratio < threshold
