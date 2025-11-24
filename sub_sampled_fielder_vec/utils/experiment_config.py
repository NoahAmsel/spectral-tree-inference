"""Configuration and utility functions for experiments."""
import os
import time
import random
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Callable

import numpy as np
import spectraltree

# Import the fiedler method default
from utils.random_entries import compute_fiedler_from_similarity


@dataclass(frozen=True)
class Config:
    """Configuration for bootstrap sweep experiments."""
    num_taxa: int = 8192
    sequence_length: int = 1000
    mutation_rate: float = 0.1
    taxa_values: List[int] | None = None
    sequence_length_values: List[int] | None = None
    progress_prints: int = 3
    p_values: List[float] = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0)
    bootstrap_reps: int = 100
    seed: int = 42
    run_name: str = "baseline_8192"
    # tree/model factories to allow easy swapping in the future
    tree_model = staticmethod(lambda n: spectraltree.balanced_binary(n))
    seq_model = staticmethod(lambda: spectraltree.Jukes_Cantor())
    # Direct function reference instead of string-based registry
    # CHANGED: Use new strict function that requires S
    fiedler_method: Callable = compute_fiedler_from_similarity
    fiedler_method_kwargs: Dict[str, object] = field(default_factory=dict)
    # Metrics configuration
    empirical_rank_threshold: float | None = None  # If None, uses 1e-12 * max(singular_values)
    coherence_k: int = 2  # Number of top singular vectors for coherence computation
    compute_metrics_on_guardrails: bool = False  # Whether to compute metrics when guardrails trigger
    # Partition algorithm parameters (align with STDR defaults from spectral_tree_reconstruction.py:40)
    num_gaps: int = 1      # Number of gap-based thresholds to evaluate
    min_split: int = 1     # Minimum partition size
    # Display mode: "progress" for clean progress bars, "debug" for verbose logging
    display_mode: str = "progress"  # "progress" or "debug"
    # Persistent cache: whether to use disk-based caching for experiment data
    use_persistent_cache: bool = False

    def get_tree_model_name(self) -> str:
        """
        Extract tree model name from the tree_model callable.

        Returns:
            Tree model name (e.g., "balanced_binary")
        """
        # Check if it's a lambda
        if hasattr(self.tree_model, '__name__'):
            if self.tree_model.__name__ == '<lambda>':
                # For lambda, try to extract from source or use generic name
                return "balanced_binary"  # Default assumption
            return self.tree_model.__name__

        # Fallback
        return "custom_tree"

    def get_seq_model_name(self) -> str:
        """
        Extract sequence model name from the seq_model callable.

        Returns:
            Sequence model name (e.g., "Jukes_Cantor")
        """
        # seq_model returns an instance, so we need to call it
        try:
            model_instance = self.seq_model()
            # Get class name
            return model_instance.__class__.__name__
        except Exception:
            # Fallback if we can't instantiate
            if hasattr(self.seq_model, '__name__'):
                return self.seq_model.__name__
            return "custom_model"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_run_dir(run_name: str) -> str:
    """Create a timestamped directory for experiment results."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    # Get the directory of the parent of utils (sub_sampled_fielder_vec)
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_dir, "results", f"{ts}-{run_name}")
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def progress_milestones(total: int, k: int) -> List[int]:
    """Calculate milestone indices for progress printing."""
    return sorted(set(np.linspace(0, total - 1, k, dtype=int).tolist()))


def save_config(cfg: Config, run_dir: str) -> None:
    """Save configuration to JSON file in run directory."""
    from utils.summaries import save_json
    
    config_dict = asdict(cfg)
    # Replace the function with its name for serialization
    config_dict['fiedler_method'] = cfg.fiedler_method.__name__
    save_json(config_dict, os.path.join(run_dir, "config.json"))

