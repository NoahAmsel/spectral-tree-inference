"""Load configuration from JSON file."""
import json
from pathlib import Path
from typing import Dict


def load_config(config_path: Path) -> Dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

