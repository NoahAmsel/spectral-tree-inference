"""Verify Davis-Kahan stability condition at critical point."""

from typing import Dict, Any


def verify_stability(row: Dict[str, Any], threshold: float = 0.5) -> bool:
    """Check if Davis-Kahan ratio is below stability threshold.

    Davis-Kahan theorem states that eigenvector perturbation is bounded by:
    ||v₂_S - v₂_M|| ≤ ||L_S - L_M||_op / (λ₂ - λ₃)

    When the ratio is < 0.5, the subsampled eigenvector is a valid approximation.

    Args:
        row: Result row containing mean_dk_ratio_S
        threshold: Stability threshold (default: 0.5)

    Returns:
        True if stable (ratio < threshold), False otherwise
    """
    if "mean_dk_ratio_S" not in row:
        raise ValueError("Missing 'mean_dk_ratio_S' in data row")

    return row["mean_dk_ratio_S"] < threshold
