"""Compare uniform vs leveraged sampling on identical matrices.

This script runs both sampling methods on the SAME tree and sequences to enable
fair comparison. Results are saved in nested structure:
    results/{timestamp}-method_comparison[_{run_name}]/
        ├── uniform/
        ├── leveraged/
        └── comparison_config.json

Set "run_name" in COMPARISON_CONFIG below to customize the output directory name.
"""
import os
import sys
import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np

from src import ExperimentRunner
from src.config.presets import custom_config
from src.config.base_config import SamplingConfig

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
COMPARISON_CONFIG: Dict[str, Any] = {
    # Run name suffix (creates: {timestamp}-method_comparison_{run_name})
    # Set to None or "" for default: {timestamp}-method_comparison
    "run_name": "balanced_binary_500_1000_3000",

    "tree_model": "balanced_binary",
    "taxa_values": [500, 1000, 3000],
    "sequence_length": 10000,
    "mutation_rate": 0.1,
    "bootstrap_reps": 20,
    "num_workers": 8,
    "p_values": list(np.logspace(-4, 0, 20)),
    "tree_params": {},

    # Shared settings
    "seed": 42,  # CRITICAL: same seed for both methods
    "coherence_k": 4,
    "num_gaps": 0,
    "guardrails_enabled": False,
    "use_middle_out": False,

    # Leveraged-specific params
    "leveraged_theta": 0.7,
    "leveraged_target_rank": 10,
    "leveraged_ialm_max_iter": 100,
    "leveraged_ialm_tol": 1e-6,
}


def _setup_comparison_directory(run_name: Optional[str] = None) -> Path:
    """Create timestamped comparison directory.
    
    Args:
        run_name: Optional suffix to append (creates method_comparison_{run_name})
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    if run_name:
        dir_name = f"{ts}-method_comparison_{run_name}"
    else:
        dir_name = f"{ts}-method_comparison"
    base_dir = Path(__file__).resolve().parents[2] / "results" / dir_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = base_dir / "comparison_config.json"
    with config_path.open("w") as f:
        json.dump(COMPARISON_CONFIG, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Method Comparison Directory: {base_dir}")
    print(f"{'='*80}\n")

    return base_dir


def _create_config_for_method(
    method: str,
    n_taxa: int,
    config: Dict[str, Any]
) -> Any:
    """Create experiment config for specific method and n_taxa.

    Args:
        method: "uniform" or "leveraged"
        n_taxa: Number of taxa
        config: Base configuration dict

    Returns:
        StructuredConfig object
    """
    # Create base config
    cfg = custom_config(
        num_taxa=n_taxa,
        sequence_length=config["sequence_length"],
        mutation_rate=config["mutation_rate"],
        tree_model=config["tree_model"],
        p_values=config["p_values"],
        bootstrap_reps=config["bootstrap_reps"],
        run_name=f"{method}_n{n_taxa}",
        display_mode="progress",
        num_workers=config["num_workers"],
        use_middle_out=config["use_middle_out"],
        seed=config["seed"],  # CRITICAL: same seed
        **config["tree_params"],
    )

    # Set sampling method
    if method == "uniform":
        cfg.sampling = SamplingConfig(method="uniform")
    elif method == "leveraged":
        cfg.sampling = SamplingConfig(
            method="leveraged",
            theta=config["leveraged_theta"],
            target_rank=config["leveraged_target_rank"],
            ialm_max_iter=config["leveraged_ialm_max_iter"],
            ialm_tol=config["leveraged_ialm_tol"],
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply shared settings
    cfg.metrics.coherence_k = config["coherence_k"]
    cfg.metrics.num_gaps = config["num_gaps"]
    cfg.guardrails.enabled = config["guardrails_enabled"]

    return cfg


def _run_method_for_taxa(
    method: str,
    n_taxa: int,
    base_dir: Path,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run single method for one n_taxa value.

    Args:
        method: "uniform" or "leveraged"
        n_taxa: Number of taxa
        base_dir: Base comparison directory
        config: Configuration dict

    Returns:
        Result metadata dict
    """
    cfg = _create_config_for_method(method, n_taxa, config)

    # Results go to {base_dir}/{method}/n{n_taxa}_L{seq_len}/
    method_dir = base_dir / method
    subdir_name = f"n{n_taxa}_L{config['sequence_length']}"

    print(f"\n{'-'*80}")
    print(f"Running {method.upper()} method: n={n_taxa}")
    print(f"Output: {method_dir / subdir_name}")
    if method == "leveraged":
        print(f"  theta={config['leveraged_theta']}, rank={config['leveraged_target_rank']}")
    print(f"{'-'*80}")

    runner = ExperimentRunner(cfg, base_dir=str(method_dir), subdir_name=subdir_name)
    run_dir, results = runner.run()

    return {
        "method": method,
        "num_taxa": n_taxa,
        "run_dir": run_dir,
    }


def run_comparison():
    """Run full comparison: both methods across all n_taxa values.

    Strategy:
    1. For each n_taxa:
       a. Run uniform method (seed=42)
       b. Run leveraged method (same seed=42)
    2. Both methods see identical tree + sequences due to shared seed
    """
    config = COMPARISON_CONFIG.copy()
    run_name = config.get("run_name") or None
    base_dir = _setup_comparison_directory(run_name)

    # Print comparison plan
    print(f"Comparison Plan:")
    print(f"  Methods:       uniform vs leveraged")
    print(f"  Taxa values:   {config['taxa_values']}")
    print(f"  Seq length:    {config['sequence_length']}")
    print(f"  Tree model:    {config['tree_model']}")
    print(f"  P-values:      {len(config['p_values'])} from {min(config['p_values']):.2e} to {max(config['p_values']):.2e}")
    print(f"  Bootstrap:     {config['bootstrap_reps']} reps")
    print(f"  Seed:          {config['seed']} (SHARED)")
    print(f"\n{'='*80}\n")

    all_results = []

    for n_taxa in config["taxa_values"]:
        # Run uniform
        result_uniform = _run_method_for_taxa("uniform", n_taxa, base_dir, config)
        all_results.append(result_uniform)

        # Run leveraged (SAME seed -> same data)
        result_leveraged = _run_method_for_taxa("leveraged", n_taxa, base_dir, config)
        all_results.append(result_leveraged)

    # Save summary
    summary_path = base_dir / "comparison_summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "config": config,
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Comparison Complete!")
    print(f"{'='*80}")
    print(f"Results directory: {base_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nResult structure:")
    for method in ["uniform", "leveraged"]:
        method_dir = base_dir / method
        if method_dir.exists():
            print(f"  {method}/")
            for subdir in sorted(method_dir.iterdir()):
                if subdir.is_dir():
                    print(f"    {subdir.name}/")
    print(f"{'='*80}\n")

    return base_dir, all_results


def main():
    """Main entry point."""
    base_dir, results = run_comparison()
    print(f"\nNext steps:")
    print(f"1. Merge results: python scripts/merge_results.py {base_dir}/uniform")
    print(f"2. Merge results: python scripts/merge_results.py {base_dir}/leveraged")
    print(f"3. Compare plots from both merged JSONs")
    return base_dir, results


if __name__ == "__main__":
    main()
