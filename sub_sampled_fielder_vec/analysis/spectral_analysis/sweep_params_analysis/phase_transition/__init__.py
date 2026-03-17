"""Phase transition detection and regime classification."""

from .detect_critical_p import detect_critical_p
from .classify_regime import classify_regime
from .detect_eigenvalue_crossing import detect_eigenvalue_crossing

__all__ = ["detect_critical_p", "classify_regime", "detect_eigenvalue_crossing"]
