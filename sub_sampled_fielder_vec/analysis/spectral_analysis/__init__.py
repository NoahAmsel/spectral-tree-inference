"""Top-level namespace for spectral analysis pipelines."""

from . import sweep_params_analysis
from . import target_quality_anlysis

__all__ = ["sweep_params_analysis", "target_quality_anlysis"]
__version__ = getattr(sweep_params_analysis, "__version__", "0.1.0")
















