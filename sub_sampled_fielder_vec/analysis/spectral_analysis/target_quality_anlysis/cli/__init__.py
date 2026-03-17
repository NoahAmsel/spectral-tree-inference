"""CLI modules for command-line interface."""

from .target_analysis_main import run_analysis, main
from .config import load_config
from .directory import create_output_directory

# Import stability analysis functions (but don't add to __all__ to avoid name collision)
from .stability_analysis_main import run_stability_analysis, main as stability_main

__all__ = [
    'run_analysis',
    'main',
    'load_config',
    'create_output_directory',
    'run_stability_analysis',
    'stability_main'
]

