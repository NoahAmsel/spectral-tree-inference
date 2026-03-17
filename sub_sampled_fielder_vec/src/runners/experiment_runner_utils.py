"""Utility functions for running experiment sweeps."""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from src import ExperimentRunner
from src.config.presets import custom_config


def extract_config_values(config: Dict[str, Any], wide_sweep_p_values: List[float]) -> Dict[str, Any]:
    """Extract all config values with defaults."""
    return {
        "tree_model": config["tree_model"],
        "taxa_values": config["taxa_values"],
        "sequence_length_values": config["sequence_length_values"],
        "mutation_rate": config["mutation_rate"],
        "p_values": config.get("p_values", wide_sweep_p_values),
        "tree_kwargs": config.get("tree_params", {}),
        "bootstrap_reps": config.get("bootstrap_reps", 10),
        "num_workers": config.get("num_workers", 8),
        "use_middle_out": config.get("use_middle_out", True),
        "coherence_k": config.get("coherence_k"),
        "num_gaps": config.get("num_gaps"),
        "guardrails_enabled": config.get("guardrails_enabled"),
        "run_name_prefix": config.get("run_name_prefix"),
        "sampling_method": config.get("sampling_method", "uniform"),
    }


def generate_run_prefix(config: Dict[str, Any], mutation_rate: float, taxa_values: List[int]) -> str:
    """Generate run name prefix from config."""
    prefix = config.get("run_name_prefix")
    if not prefix:
        mu_str = str(mutation_rate).replace(".", "p")
        pref_taxa = f"n{taxa_values[0]}" if len(taxa_values) == 1 else f"n{min(taxa_values)}-{max(taxa_values)}"
        sampling_method = config.get("sampling_method", "uniform")
        tree_model = config["tree_model"]
        prefix = f"{tree_model}_{pref_taxa}_mu_{mu_str}_{sampling_method}"
    else:
        # Append sampling method to prefix if not already included
        sampling_method = config.get("sampling_method", "uniform")
        if sampling_method not in prefix and sampling_method != "uniform":
            prefix = f"{prefix}_{sampling_method}"
    return prefix


def setup_experiment_directory(
    config: Dict[str, Any],
    prefix: str,
    sweep_config: Dict[str, Any],
    script_dir: str = None
) -> Path:
    """Create experiment directory and save config."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    if "base_dir_root" in config:
        base_dir_root = config["base_dir_root"]
    else:
        if script_dir:
            # If script_dir is provided (from run_experiment.py), use it
            base_dir_root = os.path.abspath(os.path.join(script_dir, "..", "results"))
        else:
            # Fallback: assume we're in src/runners, go up to project root then to results
            base_dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))

    # Organize by tree_model/sampling_method for better organization
    tree_model = config.get("tree_model", "unknown")
    sampling_method = config.get("sampling_method", "uniform")
    base_dir = os.path.abspath(os.path.join(base_dir_root, tree_model, sampling_method, f"{ts}-{prefix}"))
    os.makedirs(base_dir, exist_ok=True)

    # Save SWEEP_CONFIG to experiment directory
    sweep_config_path = os.path.join(base_dir, "sweep_config.json")
    with open(sweep_config_path, "w") as f:
        json.dump(sweep_config, f, indent=2)
    print(f"Saved sweep config to {sweep_config_path}")

    print(f"\n{'='*80}")
    print(f"Grid Sweep Base Directory: {base_dir}")
    print(f"{'='*80}\n")
    
    return Path(base_dir)


def get_sampling_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all sampling-related parameters."""
    return {
        "sampling_method": config.get("sampling_method", "uniform"),
        "sampling_theta": config.get("sampling_theta", 0.3),
        "sampling_target_rank": config.get("sampling_target_rank", 2),
        "sampling_ialm_max_iter": config.get("sampling_ialm_max_iter", 100),
        "sampling_ialm_tol": config.get("sampling_ialm_tol", 1e-6),
        "sampling_ialm_bypass_threshold": config.get("sampling_ialm_bypass_threshold", 0.1),
        "sampling_force_leveraged": config.get("sampling_force_leveraged", False),
        "sampling_allow_uniform_fallback": config.get("sampling_allow_uniform_fallback", True),
        "log_sampling_diagnostics": config.get("log_sampling_diagnostics", False),
        "truncation_threshold": config.get("truncation_threshold", 0.0),
    }


def create_experiment_config(
    config: Dict[str, Any],
    n_taxa: int,
    seq_len: int,
    tree_model: str,
    mutation_rate: float,
    p_values: List[float],
    bootstrap_reps: int,
    num_workers: int,
    use_middle_out: bool,
    prefix: str,
    tree_kwargs: Dict[str, Any],
) -> Any:
    """Create and configure experiment config object."""
    sampling_config = get_sampling_config(config)

    cfg = custom_config(
        num_taxa=n_taxa,
        sequence_length=seq_len,
        mutation_rate=mutation_rate,
        tree_model=tree_model,
        p_values=p_values,
        bootstrap_reps=bootstrap_reps,
        run_name=prefix,
        display_mode="progress",
        num_workers=num_workers,
        use_middle_out=use_middle_out,
        sampling_method=sampling_config["sampling_method"],
        sampling_theta=sampling_config["sampling_theta"],
        sampling_target_rank=sampling_config["sampling_target_rank"],
        sampling_ialm_max_iter=sampling_config["sampling_ialm_max_iter"],
        sampling_ialm_tol=sampling_config["sampling_ialm_tol"],
        sampling_ialm_bypass_threshold=sampling_config["sampling_ialm_bypass_threshold"],
        sampling_force_leveraged=sampling_config["sampling_force_leveraged"],
        sampling_allow_uniform_fallback=sampling_config["sampling_allow_uniform_fallback"],
        log_sampling_diagnostics=sampling_config["log_sampling_diagnostics"],
        truncation_threshold=sampling_config["truncation_threshold"],
        use_persistent_cache=config.get("use_persistent_cache", False),
        **tree_kwargs,
    )

    # Apply sweep-level metric/guardrail overrides
    if "coherence_k" in config:
        cfg.metrics.coherence_k = config["coherence_k"]
    if "num_gaps" in config:
        cfg.metrics.num_gaps = config["num_gaps"]
    if "guardrails_enabled" in config:
        cfg.guardrails.enabled = config["guardrails_enabled"]
    
    return cfg


def run_single_experiment(
    config: Dict[str, Any],
    base_dir: Path,
    n_taxa: int,
    seq_len: int,
    tree_model: str,
    mutation_rate: float,
    p_values: List[float],
    bootstrap_reps: int,
    num_workers: int,
    use_middle_out: bool,
    prefix: str,
    tree_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    subdir_name = f"n{n_taxa}_L{seq_len}"
    sampling_config = get_sampling_config(config)
    
    cfg = create_experiment_config(
        config, n_taxa, seq_len, tree_model, mutation_rate, p_values,
        bootstrap_reps, num_workers, use_middle_out, prefix, tree_kwargs
    )

    print(f"\n{'-'*80}")
    print(f"Launching experiment for {tree_model} tree: n={n_taxa}, L={seq_len}")
    print(f"Sampling method: {sampling_config['sampling_method']}")
    if sampling_config["sampling_method"] == "leveraged":
        print(f"  Phase 1 ratio (theta): {sampling_config['sampling_theta']}")
        print(f"  Target rank: {sampling_config['sampling_target_rank']}")
        print(f"  IALM max_iter: {sampling_config['sampling_ialm_max_iter']}, tol: {sampling_config['sampling_ialm_tol']}")
        print(f"  IALM bypass threshold: {sampling_config['sampling_ialm_bypass_threshold']}")
        print(f"  Force leveraged: {sampling_config['sampling_force_leveraged']}")
    
    runner = ExperimentRunner(cfg, base_dir=str(base_dir), subdir_name=subdir_name)
    run_dir, results = runner.run()
    
    return {
        "num_taxa": n_taxa,
        "sequence_length": seq_len,
        "run_dir": run_dir,
        "results": results,
    }


def auto_generate_plots(base_dir: Path) -> None:
    """Automatically merge results and generate combined plot."""
    try:
        from scripts.merge_results import merge_run_directory
        from src.utils.plotting import plot_taxa_sweep, _extract_model_name
        
        merged = merge_run_directory(base_dir)
        output_json = base_dir / "results_grid_merged.json"
        with output_json.open("w") as fh:
            json.dump(merged, fh, indent=2, allow_nan=True)
        print(f"\nWrote merged grid with {len(merged['rows'])} rows to {output_json}")
        
        # Generate plot
        plot_path = base_dir / "partition_agreement.png"
        model_name = _extract_model_name(base_dir.name)
        
        # Read config for subtitle
        config_path = base_dir / "sweep_config.json"
        if config_path.exists():
            with config_path.open("r") as f:
                sweep_config = json.load(f)
            seq_len = sweep_config["sequence_length_values"][0]
            mu = sweep_config["mutation_rate"]
            ne = sweep_config.get("tree_params", {}).get("pop_size", "N/A")
            bootstrap_reps = sweep_config.get("bootstrap_reps", "N/A")
            subtitle = f"$L = {seq_len}$, $\\mu = {mu}$, $N_e = {ne}$, {bootstrap_reps} bootstrap reps"
        else:
            subtitle = None
        
        plot_taxa_sweep(
            json_path=str(output_json),
            output_path=str(plot_path),
            model_name=model_name,
            subtitle=subtitle,
        )
        print(f"Generated combined plot: {plot_path}")
    except Exception as e:
        print(f"Warning: Could not auto-generate merged plot: {e}")
        import traceback
        traceback.print_exc()
