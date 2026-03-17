"""
Experiment configuration presets and factories.

This module provides convenient preset configurations for common experiment scenarios
using the new StructuredConfig system.
"""

import numpy as np
from .base_config import (
    StructuredConfig, TreeConfig, SequenceConfig, ExperimentConfig,
    SamplingConfig, MetricsConfig, GuardrailsConfig, CacheConfig, OutputConfig
)


def get_default_config() -> StructuredConfig:
    """
    Get default configuration template.

    Returns basic StructuredConfig with balanced_binary tree and JC69 model.
    Suitable as a base for customization or sweeps.

    Returns:
        StructuredConfig: Default configuration
    """
    return StructuredConfig(
        tree=TreeConfig(
            model="balanced_binary",
            params={"num_taxa": 256, "edge_length": 1.0}
        ),
        sequence=SequenceConfig(
            model="JC69",
            len=1000,
            params={"mutation_rate": 0.1}
        ),
        experiment=ExperimentConfig(
            p_values=[0.001, 0.01, 0.1, 0.5, 1.0],
            bootstrap_reps=50,
            seed=42,
            run_name="default_experiment"
        )
    )


def quick_test_config() -> StructuredConfig:
    """
    Create a quick test configuration for development/debugging.

    Small matrices (32 taxa), short sequences (300bp), few bootstrap reps (5)
    for quick iteration and testing.

    Returns:
        StructuredConfig: Quick test configuration
    """
    return StructuredConfig(
        tree=TreeConfig(
            model="balanced_binary",
            params={"num_taxa": 32, "edge_length": 1.0}
        ),
        sequence=SequenceConfig(
            model="JC69",
            len=300,
            params={"mutation_rate": 0.1}
        ),
        experiment=ExperimentConfig(
            p_values=[0.01, 0.1, 0.5, 1.0],
            bootstrap_reps=5,
            seed=42,
            run_name="quick_test",
            display_mode="progress"
        ),
        guardrails=GuardrailsConfig(enabled=False)  # Disable for quick tests
    )


def standard_config() -> StructuredConfig:
    """
    Create a standard experiment configuration.

    Medium-sized matrices (1024 taxa), standard bootstrap reps (100),
    logarithmic p-value spacing for thorough coverage.

    Returns:
        StructuredConfig: Standard experiment configuration
    """
    return StructuredConfig(
        tree=TreeConfig(
            model="balanced_binary",
            params={"num_taxa": 1024, "edge_length": 1.0}
        ),
        sequence=SequenceConfig(
            model="JC69",
            len=1000,
            params={"mutation_rate": 0.1}
        ),
        experiment=ExperimentConfig(
            p_values=list(np.logspace(-4, 0, 15)),  # 15 points from 1e-4 to 1.0
            bootstrap_reps=100,
            seed=42,
            run_name="standard_experiment",
            display_mode="progress"
        )
    )


def large_scale_config() -> StructuredConfig:
    """
    Create a large-scale experiment configuration.

    Large matrices (8192 taxa), long sequences (5000bp), optimized for
    parallel execution with middle-out strategy.

    Returns:
        StructuredConfig: Large-scale experiment configuration
    """
    return StructuredConfig(
        tree=TreeConfig(
            model="balanced_binary",
            params={"num_taxa": 8192, "edge_length": 1.0}
        ),
        sequence=SequenceConfig(
            model="JC69",
            len=5000,
            params={"mutation_rate": 0.1}
        ),
        experiment=ExperimentConfig(
            p_values=list(np.logspace(-4, 0, 15)),
            bootstrap_reps=10,  # Fewer reps for large scale (still statistically meaningful)
            seed=42,
            run_name="large_scale_8192",
            display_mode="progress",
            num_workers=8,  # Parallel execution
            use_middle_out=True  # Middle-out p-value processing
        ),
        cache=CacheConfig(use_persistent_cache=True)  # Use persistent cache for large experiments
    )


def custom_config(
    num_taxa: int = 1024,
    sequence_length: int = 1000,
    mutation_rate: float = 0.1,
    tree_model: str = "balanced_binary",
    seq_model: str = "JC69",
    p_values: list = None,
    bootstrap_reps: int = 100,
    seed: int = 42,
    run_name: str = "custom",
    display_mode: str = "progress",
    num_workers: int = 1,
    use_middle_out: bool = False,
    validate_partition_in_tree: bool = True,
    sampling_method: str = "uniform",
    sampling_theta: float = 0.3,
    sampling_target_rank: int = 2,
    sampling_ialm_max_iter: int = 100,
    sampling_ialm_tol: float = 1e-6,
    sampling_ialm_bypass_threshold: float = 0.1,
    sampling_force_leveraged: bool = False,
    sampling_allow_uniform_fallback: bool = True,
    log_sampling_diagnostics: bool = False,
    truncation_threshold: float = 0.0,
    use_persistent_cache: bool = False,
    **kwargs
) -> StructuredConfig:
    """
    Create a custom configuration with specified parameters.

    Convenient builder for one-off experiments without manually constructing
    the full StructuredConfig hierarchy.

    Args:
        num_taxa: Number of taxa
        sequence_length: Sequence length
        mutation_rate: Mutation rate
        tree_model: Tree model name (balanced_binary, lopsided, kingman, kingman_mean, birth_death)
        seq_model: Sequence model name (JC69, HKY, GTR, TN93, T92)
        p_values: Sampling probabilities (default: logspace from 1e-4 to 1.0)
        bootstrap_reps: Number of bootstrap replicates
        seed: Random seed
        run_name: Experiment name
        display_mode: Display mode ("progress" or "debug")
        num_workers: Number of parallel workers (1 = sequential)
        use_middle_out: Use middle-out p-value processing strategy
        validate_partition_in_tree: Validate that Fiedler partition corresponds to a real tree edge before running experiment
        sampling_method: Sampling method ("uniform" or "leveraged")
        sampling_theta: Phase 1 budget ratio for leveraged sampling (0 < theta < 1)
        sampling_target_rank: Rank for SVD in leverage estimation
        sampling_ialm_max_iter: Maximum IALM solver iterations
        sampling_ialm_tol: IALM convergence tolerance
        sampling_ialm_bypass_threshold: Skip IALM when p >= this threshold (default: 0.1)
        sampling_force_leveraged: Force leveraged sampling even when Phase 1 budget is insufficient
        sampling_allow_uniform_fallback: Allow fallback to uniform sampling when p is too small (default: True)
        truncation_threshold: Minimum similarity threshold - values below this are set to 0.0 (default: 0.0, i.e., no truncation)
        use_persistent_cache: Enable disk-based caching of experiment data (tree, observations, matrices)
        **kwargs: Additional model-specific parameters (e.g., kappa, edge_length, etc.)

    Returns:
        StructuredConfig: Custom configuration

    Examples:
        >>> # Simple custom config
        >>> cfg = custom_config(num_taxa=512, bootstrap_reps=50)

        >>> # HKY model with custom kappa
        >>> cfg = custom_config(seq_model="HKY", kappa=3.0, num_taxa=1024)

        >>> # Kingman coalescent tree
        >>> cfg = custom_config(tree_model="kingman", pop_size=2.0, num_taxa=512)
    """
    if p_values is None:
        p_values = list(np.logspace(-4, 0, 15))

    # Build tree params
    tree_params = {"num_taxa": num_taxa}

    # Add model-specific defaults and kwargs
    if tree_model == "balanced_binary":
        tree_params["edge_length"] = kwargs.get("edge_length", 1.0)
    elif tree_model == "lopsided":
        tree_params["edge_length"] = kwargs.get("edge_length", 1.0)
    elif tree_model in {"kingman", "kingman_mean"}:
        tree_params["pop_size"] = kwargs.get("pop_size", 1.0)
    elif tree_model == "birth_death":
        tree_params["birth_rate"] = kwargs.get("birth_rate", 0.5)
        tree_params["death_rate"] = kwargs.get("death_rate", 0.0)

    # Build sequence params
    seq_params = {"mutation_rate": mutation_rate}

    if seq_model == "HKY":
        seq_params["kappa"] = kwargs.get("kappa", 2.0)
    elif seq_model == "GTR":
        seq_params["transition_rates"] = kwargs.get("transition_rates", [1, 2, 1, 1, 2, 1])
    elif seq_model == "TN93":
        seq_params["kappa1"] = kwargs.get("kappa1", 2.0)
        seq_params["kappa2"] = kwargs.get("kappa2", 2.0)
    elif seq_model == "T92":
        seq_params["theta"] = kwargs.get("theta", 0.5)
        seq_params["kappa1"] = kwargs.get("kappa1", 2.0)
        seq_params["kappa2"] = kwargs.get("kappa2", 2.0)

    # Build metrics config
    metrics = MetricsConfig(validate_partition_in_tree=validate_partition_in_tree)
    
    # Build sampling config
    sampling = SamplingConfig(
        method=sampling_method,
        theta=sampling_theta,
        target_rank=sampling_target_rank,
        ialm_max_iter=sampling_ialm_max_iter,
        ialm_tol=sampling_ialm_tol,
        ialm_bypass_threshold=sampling_ialm_bypass_threshold,
        force_leveraged=sampling_force_leveraged,
        allow_uniform_fallback=sampling_allow_uniform_fallback,
        log_sampling_diagnostics=log_sampling_diagnostics,
        truncation_threshold=truncation_threshold
    )

    # Build cache config
    cache = CacheConfig(use_persistent_cache=use_persistent_cache)

    return StructuredConfig(
        tree=TreeConfig(model=tree_model, params=tree_params),
        sequence=SequenceConfig(model=seq_model, len=sequence_length, params=seq_params),
        experiment=ExperimentConfig(
            p_values=p_values,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
            run_name=run_name,
            display_mode=display_mode,
            num_workers=num_workers,
            use_middle_out=use_middle_out
        ),
        sampling=sampling,
        metrics=metrics,
        cache=cache
    )
