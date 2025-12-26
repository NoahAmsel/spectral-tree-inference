"""Create output directory for analysis results."""
from pathlib import Path
from datetime import datetime


def create_output_directory(experiment_name: str, base_dir: Path = None) -> Path:
    """
    Create output directory for analysis results.

    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for outputs (defaults to analysis_results/)

    Returns:
        Path to created output directory
    """
    if base_dir is None:
        # Create results directory in target_quality_anlysis folder
        base_dir = Path(__file__).resolve().parents[1] / "analysis_results"

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = base_dir / f"{timestamp}-{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir

