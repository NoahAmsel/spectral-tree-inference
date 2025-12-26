"""
Predefined sweep configurations for common experimental patterns.

This module provides utilities to generate lists of configurations for parameter sweeps.
Common sweep types include:
- Taxa sweep: Vary number of taxa
- Sequence length sweep: Vary sequence length
- Tree model sweep: Compare different tree topologies
- Sequence model sweep: Compare different evolution models
- Mutation rate sweep: Vary mutation rate
"""

from typing import List
from .base_config import (
    StructuredConfig, TreeConfig, SequenceConfig,
    ExperimentConfig, MetricsConfig, GuardrailsConfig, CacheConfig, OutputConfig
)
import numpy as np


class SweepType:
    """Factory for creating predefined sweep configurations."""

    @staticmethod
    def taxa_sweep(
        taxa_values: List[int] = None,
        base_config: StructuredConfig = None
    ) -> List[StructuredConfig]:
        """
        Generate configs sweeping over num_taxa.

        Args:
            taxa_values: List of taxa counts to sweep over
            base_config: Base configuration to modify (if None, uses default)

        Returns:
            List of StructuredConfig objects, one per taxa value

        Example:
            >>> configs = SweepType.taxa_sweep([128, 256, 512])
            >>> for cfg in configs:
            ...     print(cfg.get_num_taxa())
            128
            256
            512
        """
        if taxa_values is None:
            taxa_values = [128, 256, 512, 1024]

        if base_config is None:
            base_config = get_default_config()

        configs = []
        for n_taxa in taxa_values:
            cfg_dict = base_config.dict()
            cfg_dict["tree"]["params"]["num_taxa"] = n_taxa
            cfg_dict["experiment"]["run_name"] = f"{base_config.experiment.run_name}_n{n_taxa}"
            configs.append(StructuredConfig(**cfg_dict))

        return configs

    @staticmethod
    def sequence_length_sweep(
        seq_lens: List[int] = None,
        base_config: StructuredConfig = None
    ) -> List[StructuredConfig]:
        """
        Generate configs sweeping over sequence length.

        Args:
            seq_lens: List of sequence lengths to sweep over
            base_config: Base configuration to modify (if None, uses default)

        Returns:
            List of StructuredConfig objects, one per sequence length
        """
        if seq_lens is None:
            seq_lens = [500, 1000, 5000]

        if base_config is None:
            base_config = get_default_config()

        configs = []
        for seq_len in seq_lens:
            cfg_dict = base_config.dict()
            cfg_dict["sequence"]["len"] = seq_len
            cfg_dict["experiment"]["run_name"] = f"{base_config.experiment.run_name}_L{seq_len}"
            configs.append(StructuredConfig(**cfg_dict))

        return configs

    @staticmethod
    def tree_model_sweep(
        tree_models: List[str] = None,
        base_config: StructuredConfig = None
    ) -> List[StructuredConfig]:
        """
        Generate configs sweeping over tree models.

        Args:
            tree_models: List of tree model names to sweep over
            base_config: Base configuration to modify (if None, uses default)

        Returns:
            List of StructuredConfig objects, one per tree model

        Note:
            When switching to birth_death model, default birth_rate and death_rate
            will be added to params if not already present.
        """
        if tree_models is None:
            tree_models = ["balanced_binary", "lopsided", "kingman", "birth_death"]

        if base_config is None:
            base_config = get_default_config()

        configs = []
        for tree_model in tree_models:
            cfg_dict = base_config.dict()
            cfg_dict["tree"]["model"] = tree_model

            # Add model-specific parameter defaults if needed
            if tree_model == "birth_death":
                if "birth_rate" not in cfg_dict["tree"]["params"]:
                    cfg_dict["tree"]["params"]["birth_rate"] = 0.5
                if "death_rate" not in cfg_dict["tree"]["params"]:
                    cfg_dict["tree"]["params"]["death_rate"] = 0.0

            cfg_dict["experiment"]["run_name"] = f"{base_config.experiment.run_name}_{tree_model}"
            configs.append(StructuredConfig(**cfg_dict))

        return configs

    @staticmethod
    def sequence_model_sweep(
        seq_models: List[str] = None,
        base_config: StructuredConfig = None
    ) -> List[StructuredConfig]:
        """
        Generate configs sweeping over sequence models.

        Args:
            seq_models: List of sequence model names to sweep over
            base_config: Base configuration to modify (if None, uses default)

        Returns:
            List of StructuredConfig objects, one per sequence model

        Note:
            GTR model requires transition_rates, which will be set to default
            [1, 2, 1, 1, 2, 1] if not already present in base_config.
        """
        if seq_models is None:
            seq_models = ["JC69", "HKY", "GTR"]

        if base_config is None:
            base_config = get_default_config()

        configs = []
        for seq_model in seq_models:
            cfg_dict = base_config.dict()
            cfg_dict["sequence"]["model"] = seq_model

            # Add model-specific defaults
            if seq_model == "HKY" and "kappa" not in cfg_dict["sequence"]["params"]:
                cfg_dict["sequence"]["params"]["kappa"] = 2.0

            elif seq_model == "GTR":
                if "transition_rates" not in cfg_dict["sequence"]["params"]:
                    # Default: slightly higher for transitions (AG, CT) vs transversions
                    cfg_dict["sequence"]["params"]["transition_rates"] = [1, 2, 1, 1, 2, 1]

            elif seq_model == "TN93":
                if "kappa1" not in cfg_dict["sequence"]["params"]:
                    cfg_dict["sequence"]["params"]["kappa1"] = 2.0
                if "kappa2" not in cfg_dict["sequence"]["params"]:
                    cfg_dict["sequence"]["params"]["kappa2"] = 2.0

            elif seq_model == "T92":
                if "theta" not in cfg_dict["sequence"]["params"]:
                    cfg_dict["sequence"]["params"]["theta"] = 0.5
                if "kappa1" not in cfg_dict["sequence"]["params"]:
                    cfg_dict["sequence"]["params"]["kappa1"] = 2.0
                if "kappa2" not in cfg_dict["sequence"]["params"]:
                    cfg_dict["sequence"]["params"]["kappa2"] = 2.0

            cfg_dict["experiment"]["run_name"] = f"{base_config.experiment.run_name}_{seq_model}"
            configs.append(StructuredConfig(**cfg_dict))

        return configs

    @staticmethod
    def mutation_rate_sweep(
        mutation_rates: List[float] = None,
        base_config: StructuredConfig = None
    ) -> List[StructuredConfig]:
        """
        Generate configs sweeping over mutation rate.

        Args:
            mutation_rates: List of mutation rates to sweep over
            base_config: Base configuration to modify (if None, uses default)

        Returns:
            List of StructuredConfig objects, one per mutation rate
        """
        if mutation_rates is None:
            mutation_rates = [0.05, 0.1, 0.2, 0.5]

        if base_config is None:
            base_config = get_default_config()

        configs = []
        for mu in mutation_rates:
            cfg_dict = base_config.dict()
            cfg_dict["sequence"]["params"]["mutation_rate"] = mu
            mu_str = f"{mu:.2f}".replace(".", "")
            cfg_dict["experiment"]["run_name"] = f"{base_config.experiment.run_name}_mu{mu_str}"
            configs.append(StructuredConfig(**cfg_dict))

        return configs

    @staticmethod
    def grid_search(
        taxa_values: List[int] = None,
        seq_lens: List[int] = None,
        base_config: StructuredConfig = None
    ) -> List[StructuredConfig]:
        """
        Generate configs for grid search over taxa and sequence length.

        Args:
            taxa_values: List of taxa counts
            seq_lens: List of sequence lengths
            base_config: Base configuration to modify

        Returns:
            List of StructuredConfig objects for all combinations
        """
        if taxa_values is None:
            taxa_values = [128, 256]
        if seq_lens is None:
            seq_lens = [500, 1000]

        if base_config is None:
            base_config = get_default_config()

        configs = []
        for n_taxa in taxa_values:
            for seq_len in seq_lens:
                cfg_dict = base_config.dict()
                cfg_dict["tree"]["params"]["num_taxa"] = n_taxa
                cfg_dict["sequence"]["len"] = seq_len
                cfg_dict["experiment"]["run_name"] = f"{base_config.experiment.run_name}_n{n_taxa}_L{seq_len}"
                configs.append(StructuredConfig(**cfg_dict))

        return configs


def get_default_config() -> StructuredConfig:
    """
    Get default experiment configuration.

    This serves as a template for creating custom configurations or sweeps.

    Returns:
        StructuredConfig with reasonable defaults
    """
    return StructuredConfig(
        tree=TreeConfig(
            model="balanced_binary",
            params={"num_taxa": 128, "edge_length": 1.0}
        ),
        sequence=SequenceConfig(
            model="JC69",
            len=1000,
            params={"mutation_rate": 0.1}
        ),
        experiment=ExperimentConfig(
            p_values=list(np.logspace(-4, 0, 15)),
            bootstrap_reps=100,
            seed=42,
            run_name="default_experiment",
            display_mode="progress"
        ),
        metrics=MetricsConfig(),
        guardrails=GuardrailsConfig(),
        cache=CacheConfig(),
        output=OutputConfig()
    )


# Convenience functions for quick config creation

def quick_test_config() -> StructuredConfig:
    """Create a quick test configuration with small parameters."""
    return StructuredConfig(
        tree=TreeConfig(model="balanced_binary", params={"num_taxa": 32, "edge_length": 1.0}),
        sequence=SequenceConfig(model="JC69", len=300, params={"mutation_rate": 0.1}),
        experiment=ExperimentConfig(
            p_values=[0.01, 0.1, 0.5, 1.0],
            bootstrap_reps=5,
            seed=42,
            run_name="quick_test",
            display_mode="debug"
        )
    )


def standard_config() -> StructuredConfig:
    """Create a standard configuration for typical experiments."""
    return StructuredConfig(
        tree=TreeConfig(model="balanced_binary", params={"num_taxa": 1024, "edge_length": 1.0}),
        sequence=SequenceConfig(model="JC69", len=1000, params={"mutation_rate": 0.1}),
        experiment=ExperimentConfig(
            p_values=list(np.logspace(-4, 0, 15)),
            bootstrap_reps=100,
            seed=42,
            run_name="standard_experiment"
        )
    )


def large_scale_config() -> StructuredConfig:
    """Create a large-scale configuration for high-taxa experiments."""
    return StructuredConfig(
        tree=TreeConfig(model="balanced_binary", params={"num_taxa": 8192, "edge_length": 1.0}),
        sequence=SequenceConfig(model="JC69", len=5000, params={"mutation_rate": 0.1}),
        experiment=ExperimentConfig(
            p_values=list(np.logspace(-4, 0, 25)),
            bootstrap_reps=10,
            seed=42,
            run_name="large_scale_8192",
            num_workers=8,
            use_middle_out=True
        )
    )
