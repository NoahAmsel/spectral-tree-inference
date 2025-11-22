"""Experiment configuration presets and factories."""
import numpy as np
import spectraltree

from utils.experiment_config import Config
from utils.random_entries import compute_fiedler_estimate


def create_quick_test_config() -> Config:
    """
    Create a quick test configuration for development/debugging.
    
    Small matrices, few bootstrap reps, for quick iteration.
    
    Returns:
        Config object for quick testing
    """
    return Config(
        num_taxa=32,
        sequence_length=300,
        mutation_rate=0.1,
        p_values=(0.01, 0.1, 0.5, 1.0),
        bootstrap_reps=5,
        seed=42,
        run_name="quick_test",
        fiedler_method=compute_fiedler_estimate,
        progress_prints=2
    )


def create_standard_config() -> Config:
    """
    Create a standard experiment configuration.
    
    Medium-sized matrices with standard bootstrap reps.
    
    Returns:
        Config object for standard experiments
    """
    return Config(
        num_taxa=1024,
        sequence_length=1000,
        mutation_rate=0.1,
        p_values=tuple(np.logspace(-4, 0, 15)),
        bootstrap_reps=100,
        seed=42,
        run_name="standard_experiment",
        fiedler_method=compute_fiedler_estimate,
        progress_prints=3
    )


def create_large_scale_config() -> Config:
    """
    Create a large-scale experiment configuration.
    
    Large matrices (8192 taxa) with standard bootstrap reps.
    
    Returns:
        Config object for large-scale experiments
    """
    return Config(
        num_taxa=8192,
        sequence_length=1000,
        mutation_rate=0.1,
        p_values=tuple(np.logspace(-4, 0, 15)),
        bootstrap_reps=10,
        seed=42,
        run_name="large_scale_8192",
        fiedler_method=compute_fiedler_estimate,
        progress_prints=3
    )


def create_taxa_sweep_config(taxa_values=None, sequence_length=1000) -> Config:
    """
    Create a taxa sweep configuration (varying taxa, fixed sequence length).
    
    Args:
        taxa_values: List of taxa values to sweep (default: [1024, 2048, 4096])
        sequence_length: Fixed sequence length (default: 1000)
    
    Returns:
        Config object for taxa sweep
    """
    if taxa_values is None:
        taxa_values = [1024, 2048, 4096]
    
    return Config(
        taxa_values=taxa_values,
        sequence_length=sequence_length,
        mutation_rate=0.1,
        p_values=tuple(np.logspace(-4, 0, 15)),
        bootstrap_reps=100,
        seed=42,
        run_name="taxa_sweep",
        fiedler_method=compute_fiedler_estimate,
        progress_prints=3
    )


def create_grid_search_config(taxa_values=None, sequence_length_values=None) -> Config:
    """
    Create a grid search configuration (varying both taxa and sequence length).
    
    Args:
        taxa_values: List of taxa values (default: [1024, 2048])
        sequence_length_values: List of sequence lengths (default: [500, 1000, 5000])
    
    Returns:
        Config object for grid search
    """
    if taxa_values is None:
        taxa_values = [1024, 2048]
    
    if sequence_length_values is None:
        sequence_length_values = [500, 1000, 5000]
    
    return Config(
        taxa_values=taxa_values,
        sequence_length_values=sequence_length_values,
        mutation_rate=0.1,
        p_values=tuple(np.logspace(-4, 0, 15)),
        bootstrap_reps=10,
        seed=42,
        run_name="grid_search",
        fiedler_method=compute_fiedler_estimate,
        progress_prints=3
    )


def create_custom_config(
    num_taxa=8192,
    sequence_length=1000,
    mutation_rate=0.1,
    p_values=None,
    bootstrap_reps=100,
    seed=42,
    run_name="custom",
    taxa_values=None,
    sequence_length_values=None,
    compute_metrics_on_guardrails=False,
    display_mode="progress"
) -> Config:
    """
    Create a custom configuration with specified parameters.
    
    Args:
        num_taxa: Number of taxa (for single experiment)
        sequence_length: Sequence length (for single experiment)
        mutation_rate: Mutation rate
        p_values: Sampling probabilities (default: logspace from 1e-4 to 1.0)
        bootstrap_reps: Number of bootstrap replicates
        seed: Random seed
        run_name: Experiment name
        taxa_values: List of taxa values (for sweep/grid)
        sequence_length_values: List of sequence lengths (for grid)
        compute_metrics_on_guardrails: Whether to compute metrics when guardrails trigger
        display_mode: Display mode ("progress" or "debug")
    
    Returns:
        Config object with custom parameters
    """
    if p_values is None:
        p_values = tuple(np.logspace(-4, 0, 15))
    
    return Config(
        num_taxa=num_taxa,
        sequence_length=sequence_length,
        mutation_rate=mutation_rate,
        taxa_values=taxa_values,
        sequence_length_values=sequence_length_values,
        p_values=p_values,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
        run_name=run_name,
        fiedler_method=compute_fiedler_estimate,
        progress_prints=3,
        compute_metrics_on_guardrails=compute_metrics_on_guardrails,
        display_mode=display_mode
    )

