"""Utilities for detecting phase transition thresholds in partition agreement curves."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

ThresholdPoint = Optional[Tuple[float, float]]


def compute_phase_transition_thresholds(
    series: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    target: float = 50.0,
    plateau_tolerance: float = 1.0,
    first_rise_threshold: float = 55.0,
    second_rise_threshold: float = 60.0,
) -> Dict[str, ThresholdPoint]:
    """
    Identify the last ``target``% point before a sustained rise in agreement.

    We scan each curve from left (low p) to right. Whenever a point sits within
    ``plateau_tolerance`` of ``target`` it becomes the current candidate
    threshold. A candidate is confirmed only when the data shows a sustained
    increase: the first point after the plateau must be >= ``first_rise_threshold``
    and the following point must be >= ``second_rise_threshold``. This avoids
    treating brief spikes as true phase transitions.

    Args:
        series: Mapping of label -> {"x": [...], "mean": [...]} sequences.
        target: Expected plateau level (defaults to 50%).
        plateau_tolerance: Allowed deviation from the plateau value.
        first_rise_threshold: Minimum agreement for the first point after the
            plateau.
        second_rise_threshold: Minimum agreement for the second point after the
            plateau; ensures the rise persists.

    Returns:
        Dict mapping each label to a tuple ``(p_value, y_value)`` or ``None`` if
        no transition is detected.
    """

    def on_plateau(value: float) -> bool:
        return value is not None and abs(value - target) <= plateau_tolerance

    thresholds: Dict[str, ThresholdPoint] = {}

    for label, curve in series.items():
        x_vals = list(curve.get("x") or [])
        y_vals = list(curve.get("mean") or [])

        last_plateau: ThresholdPoint = None
        threshold_point: ThresholdPoint = None
        post_plateau_values: list[float] = []

        for y_idx, (p_val, y_val) in enumerate(zip(x_vals, y_vals)):
            if on_plateau(y_val):
                last_plateau = (p_val, y_val)
                post_plateau_values = []
                continue

            if last_plateau is None:
                continue

            post_plateau_values.append(y_val)

            if len(post_plateau_values) == 1:
                if y_val < first_rise_threshold:
                    # First point after plateau failed the guard; wait for a new plateau.
                    last_plateau = None
                    post_plateau_values = []
                continue

            if len(post_plateau_values) == 2:
                first_ok = post_plateau_values[0] >= first_rise_threshold
                second_ok = post_plateau_values[1] >= second_rise_threshold
                if first_ok and second_ok:
                    threshold_point = last_plateau
                    break

                # Guard failed; abandon this plateau candidate until another plateau appears.
                last_plateau = None
                post_plateau_values = []

        if threshold_point is None and last_plateau is not None:
            threshold_point = last_plateau

        thresholds[label] = threshold_point

    return thresholds


__all__ = ["compute_phase_transition_thresholds"]

