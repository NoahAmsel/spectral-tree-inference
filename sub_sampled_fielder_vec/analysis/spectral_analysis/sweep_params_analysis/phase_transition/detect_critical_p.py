"""Detect critical sampling rate p_crit where phase transition occurs."""

from typing import List, Dict, Any, Optional


def detect_critical_p(
    config_data: List[Dict[str, Any]], threshold: float = 95.0
) -> Optional[float]:
    """Find critical sampling rate where partition_agreement_M reaches threshold.

    Args:
        config_data: List of result rows for single (N, L) configuration, sorted by p
        threshold: Partition agreement threshold for transition (default: 95%)

    Returns:
        p_crit: Critical sampling rate, or None if never reached
    """
    for row in config_data:
        if row["partition_agreement_M"] >= threshold:
            return row["p"]

    return None
