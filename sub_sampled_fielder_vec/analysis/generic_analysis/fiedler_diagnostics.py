"""Fiedler vector quality diagnostics utilities."""
import numpy as np


def compute_ipr(ipr_values: np.ndarray, n: int) -> np.ndarray:
    """Normalize IPR by 1/n to get localization ratio (1 = delocalized, n = localized).
    
    Inverse Participation Ratio (IPR) measures eigenvector localization:
    - IPR = sum(v_i^4) / (sum(v_i^2))^2
    - For delocalized vector: IPR ~ 1/n (ratio = 1)
    - For localized vector: IPR ~ 1 (ratio = n)
    
    Args:
        ipr_values: Array of IPR values
        n: Vector dimension (number of taxa)
        
    Returns:
        Array of localization ratios: IPR * n
        Values near 1 indicate delocalized (good), near n indicate localized (bad)
    """
    if n <= 0:
        return np.full_like(ipr_values, np.nan)
    
    return ipr_values * n


def analyze_sign_stability(sign_agreements: np.ndarray) -> dict:
    """Compute statistics on sign agreement across bootstrap reps.
    
    Args:
        sign_agreements: Array of sign agreement percentages (0-100)
        
    Returns:
        Dictionary with keys:
        - 'mean': Mean sign agreement
        - 'median': Median sign agreement
        - 'std': Standard deviation
        - 'min': Minimum value
        - 'max': Maximum value
    """
    mask = ~np.isnan(sign_agreements)
    if np.sum(mask) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
        }
    
    clean_values = sign_agreements[mask]
    
    return {
        'mean': float(np.mean(clean_values)),
        'median': float(np.median(clean_values)),
        'std': float(np.std(clean_values)),
        'min': float(np.min(clean_values)),
        'max': float(np.max(clean_values)),
    }
