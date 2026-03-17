"""Classify sampling regimes based on partition agreement."""

from typing import Literal

RegimeType = Literal["noise_bulk", "spectral_emergence", "perturbation_plateau"]


def classify_regime(
    partition_agreement: float,
    noise_threshold: float = 55.0,
    plateau_threshold: float = 95.0,
) -> RegimeType:
    """Classify operating regime based on partition agreement.

    Regimes:
    - noise_bulk: λ₂ buried in noise (agreement ≤ noise_threshold)
    - spectral_emergence: BBP transition zone (noise < agreement < plateau)
    - perturbation_plateau: Davis-Kahan stability (agreement ≥ plateau_threshold)

    Args:
        partition_agreement: Partition agreement percentage
        noise_threshold: Upper bound for noise regime (default: 55%)
        plateau_threshold: Lower bound for plateau regime (default: 95%)

    Returns:
        Regime classification
    """
    if partition_agreement <= noise_threshold:
        return "noise_bulk"
    elif partition_agreement >= plateau_threshold:
        return "perturbation_plateau"
    else:
        return "spectral_emergence"
