"""Save analysis outputs to timestamped directory."""

from pathlib import Path
from datetime import datetime


def save_outputs(results_dir: Path) -> Path:
    """Create timestamped analysis subdirectory.

    Args:
        results_dir: Parent results directory

    Returns:
        Path to analysis output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = results_dir / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
