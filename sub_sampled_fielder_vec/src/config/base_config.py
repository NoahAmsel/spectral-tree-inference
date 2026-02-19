"""
Pydantic-based structured configuration system for sub-sampled STDR experiments.

This module provides a hierarchical, type-safe configuration schema using Pydantic.
The config is organized into logical sections: tree, sequence, experiment, metrics,
guardrails, cache, and output.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Literal
import numpy as np
import json
from pathlib import Path


class TreeConfig(BaseModel):
    """Tree topology configuration.

    Attributes:
        model: Tree topology type (balanced_binary, lopsided, kingman, birth_death)
        params: Model-specific parameters (must include num_taxa)
    """
    model: Literal["balanced_binary", "lopsided", "kingman", "kingman_mean", "birth_death"]
    params: Dict[str, Any] = Field(default_factory=dict)

    @validator("params")
    def validate_tree_params(cls, v, values):
        """Validate tree parameters based on model type."""
        model = values.get("model")

        # Ensure num_taxa is present
        if "num_taxa" not in v:
            raise ValueError(f"num_taxa is required in tree.params")

        num_taxa = v["num_taxa"]
        if not isinstance(num_taxa, int) or num_taxa <= 0:
            raise ValueError(f"num_taxa must be a positive integer, got {num_taxa}")

        # Model-specific validation and defaults
        if model == "balanced_binary":
            # Check if num_taxa is power of 2
            if num_taxa & (num_taxa - 1) != 0:
                raise ValueError(f"balanced_binary requires num_taxa to be power of 2, got {num_taxa}")
            if "edge_length" not in v:
                v["edge_length"] = 1.0

        elif model == "lopsided":
            if "edge_length" not in v:
                v["edge_length"] = 1.0

        elif model in {"kingman", "kingman_mean"}:
            if "pop_size" not in v:
                v["pop_size"] = 1.0

        elif model == "birth_death":
            if "birth_rate" not in v:
                v["birth_rate"] = 0.5
            if "death_rate" not in v:
                v["death_rate"] = 0.0

        return v

    class Config:
        extra = "forbid"  # Reject unknown fields


class SequenceConfig(BaseModel):
    """Sequence evolution configuration.

    Attributes:
        model: Sequence evolution model (JC69, HKY, GTR, TN93, T92)
        len: Sequence length (number of sites)
        params: Model-specific parameters (must include mutation_rate)
    """
    model: Literal["JC69", "HKY", "GTR", "TN93", "T92"]
    len: int = Field(gt=0, description="Sequence length (number of sites)")
    params: Dict[str, Any] = Field(default_factory=dict)

    @validator("params")
    def validate_seq_params(cls, v, values):
        """Validate sequence model parameters."""
        model = values.get("model")

        # Ensure mutation_rate is present
        if "mutation_rate" not in v:
            v["mutation_rate"] = 0.1  # Default mutation rate

        mutation_rate = v["mutation_rate"]
        if not isinstance(mutation_rate, (int, float)) or mutation_rate <= 0:
            raise ValueError(f"mutation_rate must be positive, got {mutation_rate}")

        # Model-specific validation and defaults
        if model == "JC69":
            if "num_classes" not in v:
                v["num_classes"] = 4  # DNA: A, C, G, T

        elif model == "HKY":
            if "kappa" not in v:
                v["kappa"] = 2.0  # Default transition/transversion ratio
            if v["kappa"] <= 0:
                raise ValueError(f"HKY kappa must be positive, got {v['kappa']}")
            # stationary_freqs is optional, will use uniform if not provided

        elif model == "GTR":
            if "transition_rates" not in v:
                raise ValueError("GTR model requires 'transition_rates' parameter (length 6 array)")
            rates = v["transition_rates"]
            if not isinstance(rates, (list, tuple)) or len(rates) != 6:
                raise ValueError(f"GTR transition_rates must be array of length 6, got {rates}")
            if any(r <= 0 for r in rates):
                raise ValueError("GTR transition_rates must all be positive")

        elif model == "TN93":
            if "kappa1" not in v:
                v["kappa1"] = 2.0
            if "kappa2" not in v:
                v["kappa2"] = 2.0
            if v["kappa1"] <= 0 or v["kappa2"] <= 0:
                raise ValueError("TN93 kappa1 and kappa2 must be positive")

        elif model == "T92":
            if "theta" not in v:
                v["theta"] = 0.5
            if not (0 <= v["theta"] <= 1):
                raise ValueError(f"T92 theta must be in [0, 1], got {v['theta']}")
            if "kappa1" not in v:
                v["kappa1"] = 2.0
            if "kappa2" not in v:
                v["kappa2"] = 2.0

        return v

    class Config:
        extra = "forbid"


class ExperimentConfig(BaseModel):
    """Experiment execution parameters.

    Attributes:
        p_values: List of sampling probabilities to test
        bootstrap_reps: Number of bootstrap replicates per p-value
        seed: Random seed for reproducibility
        run_name: Experiment name (used for output directory)
        display_mode: Display mode (progress bars or debug logging)
        num_workers: Number of parallel workers (1 = sequential)
        use_middle_out: Use middle-out p-value processing strategy
    """
    p_values: List[float] = Field(min_items=1)
    bootstrap_reps: int = Field(gt=0)
    seed: int = 42
    run_name: str = "experiment"
    display_mode: Literal["progress", "debug"] = "progress"
    num_workers: int = Field(ge=1, default=1)
    use_middle_out: bool = False

    @validator("p_values")
    def validate_p_values(cls, v):
        """Ensure p_values are in valid range (0, 1]."""
        for p in v:
            if not (0 < p <= 1):
                raise ValueError(f"All p_values must be in (0, 1], got {p}")
        return v

    class Config:
        extra = "forbid"


class SamplingConfig(BaseModel):
    """Sampling method configuration.

    Attributes:
        method: Sampling method ("uniform", "leveraged", or "hldt")
               - uniform: Simple uniform sampling (baseline)
               - leveraged: IALM-based matrix completion (high accuracy, slow)
               - hldt: HLDT debiased estimator (high speed, good accuracy)
        theta: Phase 1 budget ratio for leveraged/hldt sampling (0 < theta < 1)
        target_rank: Rank r for SVD in leverage score computation
        tau_floor_multiplier: Multiplier for regularization floor in HLDT (default: 1.0)
                             τ_floor = multiplier × mean(leverage_scores)
        ialm_max_iter: Maximum iterations for IALM solver (leveraged method only)
        ialm_tol: Convergence tolerance for IALM (leveraged method only)
        ialm_bypass_threshold: Skip IALM when p >= this threshold (leveraged method only)
        force_leveraged: Force leveraged/lds sampling even when Phase 1 budget is insufficient
        allow_uniform_fallback: Allow fallback to uniform sampling when p is too small (default: True)
        log_sampling_diagnostics: Save detailed sampling diagnostics for analysis
    """
    method: Literal["uniform", "leveraged", "lds"] = "uniform"
    theta: float = Field(default=0.3, gt=0.0, lt=1.0, description="Phase 1 budget ratio")
    target_rank: int = Field(default=2, ge=1, description="SVD rank for leverage estimation")
    tau_floor_multiplier: float = Field(default=1.0, gt=0.0, description="LDS regularization floor multiplier")
    ialm_max_iter: int = Field(default=100, ge=1, description="IALM maximum iterations")
    ialm_tol: float = Field(default=1e-6, gt=0.0, description="IALM convergence tolerance")
    ialm_bypass_threshold: float = Field(default=0.1, gt=0.0, le=1.0, description="Skip IALM when p >= threshold")
    force_leveraged: bool = Field(default=False, description="Force leveraged sampling even when Phase 1 budget is insufficient")
    allow_uniform_fallback: bool = Field(default=True, description="Allow fallback to uniform sampling when p is too small")
    log_sampling_diagnostics: bool = Field(default=False, description="Save detailed sampling diagnostics for analysis")

    class Config:
        extra = "forbid"


class MetricsConfig(BaseModel):
    """Metrics computation parameters.

    Attributes:
        empirical_rank_threshold: Threshold for empirical rank computation (None = auto)
        coherence_k: Number of top singular vectors for coherence computation
        num_gaps: Number of gap-based thresholds for partition (STDR parameter).
            0 = always use threshold=0 (sign-based partition, no optimization).
            >0 = evaluate num_gaps gap-based thresholds and pick best by σ₂.
        min_split: Minimum partition size (STDR parameter)
        validate_partition_in_tree: Validate that Fiedler partition corresponds to a real tree edge before running experiment
    """
    empirical_rank_threshold: Optional[float] = None
    coherence_k: int = Field(ge=1, default=2)
    num_gaps: int = Field(ge=0, default=0)  # 0 = always use threshold=0 (no gap-based optimization)
    min_split: int = Field(ge=1, default=1)
    validate_partition_in_tree: bool = Field(
        default=True,
        description="Validate that Fiedler partition corresponds to a real tree edge before running experiment"
    )

    class Config:
        extra = "forbid"


class GuardrailsConfig(BaseModel):
    """Early stopping guardrails configuration.

    Attributes:
        enabled: Whether guardrails are enabled
        metric: Metric to use for guardrails decision
        low_side_threshold: Stop low-side expansion when metric < threshold
        low_side_epsilon: Epsilon margin for threshold check
        compute_metrics_on_trigger: Whether to compute full metrics when guardrails trigger
    """
    enabled: bool = True
    metric: Literal["sign_agreement", "partition_agreement_M", "partition_agreement_S"] = "partition_agreement_M"
    low_side_threshold: float = 50.0
    low_side_epsilon: float = 0.99
    compute_metrics_on_trigger: bool = False

    @validator("low_side_threshold")
    def validate_threshold(cls, v):
        """Ensure threshold is in valid range."""
        if not (0 <= v <= 100):
            raise ValueError(f"low_side_threshold should be percentage in [0, 100], got {v}")
        return v

    class Config:
        extra = "forbid"


class CacheConfig(BaseModel):
    """Caching behavior configuration.

    Attributes:
        use_persistent_cache: Whether to use disk-based caching
        cache_dir: Directory for cache files (None = default)
    """
    use_persistent_cache: bool = False
    cache_dir: Optional[str] = None

    class Config:
        extra = "forbid"


class OutputConfig(BaseModel):
    """Output configuration.

    Attributes:
        dir: Base output directory for results
        save_matrices: Whether to save similarity matrices
        save_fiedler_vectors: Whether to save Fiedler vectors
    """
    dir: str = "results"
    save_matrices: bool = False
    save_fiedler_vectors: bool = True

    class Config:
        extra = "forbid"


class StructuredConfig(BaseModel):
    """Complete experiment configuration with hierarchical structure.

    This is the main configuration class that combines all sub-configurations.
    It provides type-safe, validated configuration with clear organization.

    Example:
        >>> cfg = StructuredConfig(
        ...     tree=TreeConfig(model="balanced_binary", params={"num_taxa": 128}),
        ...     sequence=SequenceConfig(model="JC69", len=1000, params={"mutation_rate": 0.1}),
        ...     experiment=ExperimentConfig(p_values=[0.1, 0.5, 1.0], bootstrap_reps=10, run_name="test")
        ... )
        >>> cfg.to_json_file("my_config.json")

    Attributes:
        tree: Tree topology configuration
        sequence: Sequence evolution configuration
        experiment: Experiment execution parameters
        sampling: Sampling method configuration (optional, has defaults)
        metrics: Metrics computation parameters (optional, has defaults)
        guardrails: Early stopping guardrails (optional, has defaults)
        cache: Caching behavior (optional, has defaults)
        output: Output configuration (optional, has defaults)
    """
    tree: TreeConfig
    sequence: SequenceConfig
    experiment: ExperimentConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    class Config:
        extra = "forbid"  # Reject unknown fields
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),  # Convert numpy arrays to lists for JSON
        }

    def to_json_file(self, path: str) -> None:
        """Save configuration to JSON file.

        Compatible with all Pydantic v2.x versions.

        Args:
            path: File path to save configuration
        """
        import json
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            # Try different Pydantic v2 serialization methods in order of preference
            if hasattr(self, 'model_dump_json'):
                # Pydantic v2.1+ preferred method
                f.write(self.model_dump_json(indent=2))
            elif hasattr(self, 'model_dump'):
                # Pydantic v2.0+ with manual JSON encoding
                f.write(json.dumps(self.model_dump(), indent=2))
            else:
                # Very early Pydantic v2.0 fallback
                f.write(json.dumps(self.dict(), indent=2))

    @classmethod
    def from_json_file(cls, path: str) -> 'StructuredConfig':
        """Load configuration from JSON file.

        Compatible with all Pydantic v2.x versions.

        Args:
            path: File path to load configuration from

        Returns:
            StructuredConfig object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If config is invalid
        """
        import json
        with open(path, 'r') as f:
            content = f.read()
            # Try different Pydantic v2 deserialization methods in order of preference
            if hasattr(cls, 'model_validate_json'):
                # Pydantic v2.1+ preferred method
                return cls.model_validate_json(content)
            elif hasattr(cls, 'model_validate'):
                # Pydantic v2.0+ with manual JSON decoding
                return cls.model_validate(json.loads(content))
            else:
                # Very early Pydantic v2.0 fallback
                return cls.parse_obj(json.loads(content))

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Compatible with all Pydantic v2.x versions.

        Returns:
            Dictionary representation of config
        """
        if hasattr(self, 'model_dump'):
            # Pydantic v2.0+ preferred method
            return self.model_dump()
        else:
            # Very early Pydantic v2.0 fallback
            return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredConfig':
        """Create config from dictionary.

        Args:
            data: Dictionary with config data

        Returns:
            StructuredConfig object

        Raises:
            ValidationError: If config is invalid
        """
        return cls(**data)

    def get_tree_model_name(self) -> str:
        """Get tree model name for logging/caching.

        Returns:
            Tree model name
        """
        return self.tree.model

    def get_seq_model_name(self) -> str:
        """Get sequence model name for logging/caching.

        Returns:
            Sequence model name
        """
        return self.sequence.model

    def get_num_taxa(self) -> int:
        """Get number of taxa from tree params.

        Returns:
            Number of taxa
        """
        return self.tree.params["num_taxa"]

    def get_sequence_length(self) -> int:
        """Get sequence length.

        Returns:
            Sequence length
        """
        return self.sequence.len

    def get_mutation_rate(self) -> float:
        """Get mutation rate from sequence params.

        Returns:
            Mutation rate
        """
        return self.sequence.params["mutation_rate"]

    def summary(self) -> str:
        """Generate human-readable summary of configuration.

        Returns:
            Multi-line summary string
        """
        lines = [
            "=" * 80,
            "Experiment Configuration Summary",
            "=" * 80,
            f"Tree Model:        {self.tree.model}",
            f"  num_taxa:        {self.get_num_taxa()}",
            f"  params:          {self.tree.params}",
            "",
            f"Sequence Model:    {self.sequence.model}",
            f"  length:          {self.sequence.len}",
            f"  mutation_rate:   {self.get_mutation_rate()}",
            f"  params:          {self.sequence.params}",
            "",
            f"Experiment:        {self.experiment.run_name}",
            f"  p_values:        {len(self.experiment.p_values)} values from {min(self.experiment.p_values):.2e} to {max(self.experiment.p_values):.2e}",
            f"  bootstrap_reps:  {self.experiment.bootstrap_reps}",
            f"  seed:            {self.experiment.seed}",
            f"  display_mode:    {self.experiment.display_mode}",
            f"  num_workers:     {self.experiment.num_workers}",
            f"  use_middle_out:  {self.experiment.use_middle_out}",
            "",
            f"Metrics:           coherence_k={self.metrics.coherence_k}, num_gaps={self.metrics.num_gaps}",
            f"Guardrails:        enabled={self.guardrails.enabled}, metric={self.guardrails.metric}",
            f"Cache:             persistent={self.cache.use_persistent_cache}",
            f"Output:            dir={self.output.dir}",
            "=" * 80,
        ]
        return "\n".join(lines)
