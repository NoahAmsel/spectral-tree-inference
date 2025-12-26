"""
Sub-sampled STDR Experiment Framework.

This package provides a complete framework for running sub-sampled STDR experiments
with support for multiple tree topologies and sequence evolution models.
"""

__version__ = "1.0.0"

from .config import StructuredConfig
from .runners import ExperimentRunner

__all__ = [
    "StructuredConfig",
    "ExperimentRunner",
    "__version__"
]
