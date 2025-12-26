"""Detect eigenvalue crossing point (BBP transition)."""

from typing import List, Dict, Any, Optional
import numpy as np


def detect_eigenvalue_crossing(config_data: List[Dict[str, Any]]) -> Optional[float]:
    """Find sampling rate where λ₂ first crosses λ₃ (spectral gap opens).

    The BBP (Baik-Ben Arous-Péché) transition occurs when the signal eigenvalue
    separates from the noise bulk.

    Args:
        config_data: List of result rows for single (N, L), sorted by p

    Returns:
        p where λ₂ > λ₃ first occurs, or None if crossing never happens or data unavailable
    """
    for row in config_data:
        lambda_2 = row.get("mean_lambda2_L_S")
        lambda_3 = row.get("mean_lambda3_L_S")

        # Skip if data missing or NaN
        if lambda_2 is None or lambda_3 is None:
            continue
        if np.isnan(lambda_2) or np.isnan(lambda_3):
            continue

        # Check if spectral gap has opened
        if lambda_2 > lambda_3:
            return row["p"]

    return None
