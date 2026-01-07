"""Example script for running experiments with leveraged sampling."""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.presets import custom_config
from src import ExperimentRunner

def main():
    """Run a simple experiment with leveraged sampling."""
    
    # Create configuration with leveraged sampling
    cfg = custom_config(
        num_taxa=512,                    # Number of taxa
        sequence_length=1000,            # Sequence length
        mutation_rate=0.1,                # Mutation rate
        tree_model="balanced_binary",    # Tree topology
        p_values=[0.01, 0.05, 0.1, 0.5, 1.0],  # Sampling probabilities
        bootstrap_reps=10,               # Bootstrap replicates (use more for real experiments)
        run_name="leveraged_example",
        # Leveraged sampling parameters:
        sampling_method="leveraged",     # Use leveraged method
        sampling_theta=0.3,              # 30% for Phase 1 (uniform)
        sampling_target_rank=2,          # Rank for SVD
        sampling_ialm_max_iter=100,     # IALM iterations
        sampling_ialm_tol=1e-6          # IALM tolerance
    )
    
    print("=" * 80)
    print("Running experiment with LEVERAGED sampling")
    print("=" * 80)
    print(f"Method: {cfg.sampling.method}")
    print(f"Theta (Phase 1 ratio): {cfg.sampling.theta}")
    print(f"Target rank: {cfg.sampling.target_rank}")
    print(f"Number of taxa: {cfg.tree.params['num_taxa']}")
    print(f"Sequence length: {cfg.sequence.len}")
    print(f"P-values: {cfg.experiment.p_values}")
    print("=" * 80)
    
    # Run experiment
    base_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    runner = ExperimentRunner(cfg, base_dir=base_dir)
    run_dir, results = runner.run()
    
    print(f"\n✓ Experiment completed!")
    print(f"Results saved to: {run_dir}")
    
    return run_dir, results


if __name__ == "__main__":
    main()

