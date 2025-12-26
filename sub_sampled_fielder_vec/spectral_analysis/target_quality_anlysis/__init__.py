"""Target Quality Analysis: Pre-flight diagnostics for full similarity matrices.

This package provides diagnostic tools to analyze full N×N similarity matrices
BEFORE running subsampling experiments. It computes:
- Matrix coherence
- Numerical rank
- Fiedler partition and σ₂ quality
- Spectral gaps (absolute and relative)
- Eigenvalue scree plots

Usage:
    python -m spectral_analysis.target_quality_anlysis.cli.main config_template.json
"""

from . import computation
from . import visualization
from . import output
from . import cli

__all__ = ["computation", "visualization", "output", "cli"]
__version__ = "0.1.0"
