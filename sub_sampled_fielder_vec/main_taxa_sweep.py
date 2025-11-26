"""Main entry point for bootstrap sweep experiments."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from utils.experiment_config import Config
from experiment.experiment_runner import ExperimentRunner
from experiment.presets import (
    create_quick_test_config,
    create_standard_config,
    create_large_scale_config,
    create_taxa_sweep_config,
    create_grid_search_config,
    create_custom_config
)


def main():
    """
    Main entry point for running experiments.
    
    Choose one of the following approaches:
    1. Use a preset configuration
    2. Create a custom configuration
    3. Manually construct a Config object
    """
    
    # Original logspace(-4, 0, 15) values:
    original_p_values = list(np.logspace(-4, 0, 15))

    # Gap-filling values for transition regions:
    gap_fill_p_values = [
        0.00193,   # Between 0.00139 and 0.00268
        0.00373,   # Between 0.00268 and 0.00518
        0.0072,    # Between 0.00518 and 0.01
        0.0139,    # Between 0.01 and 0.0193
        0.0268,    # Between 0.0193 and 0.0373
        0.05       # Between 0.0373 and 0.072
    ]

    # Refinement values for smoother transition curves (4 additional points):
    # Added based on historical data showing transition typically occurs in p ≈ 0.005-0.02 range
    refinement_p_values = [
        0.0062,    # Between 0.00518 and 0.0072 (finer resolution in steep transition)
        0.0082,    # Between 0.0072 and 0.01 (captures mid-transition behavior)
        0.0115,    # Between 0.01 and 0.0139 (post-transition detail)
        0.0165,    # Between 0.0139 and 0.0193 (saturation onset)
    ]

    # Combine and sort all p values (21 base + 4 refinement = 25 total)
    all_p_values = sorted(original_p_values + gap_fill_p_values + refinement_p_values)

    cfg = create_custom_config(
        taxa_values=[8192],
        sequence_length_values=[1000],
        mutation_rate=0.1,
        p_values=tuple(all_p_values),
        bootstrap_reps=10,
        compute_metrics_on_guardrails=True,
        run_name="8192_1k_mu_01_parallel",
        display_mode="debug",
        # Parallel execution with middle-out strategy
        num_workers=8,
        use_middle_out=True,
        low_side_threshold=50.0,
        low_side_epsilon=0.99,  # Triggers when performance < 50.99%
        guardrails_metric='partition_agreement_M'
    )
    
    # Option 3: Manually construct Config (for full control)
    # cfg = Config(
    #     taxa_values=[1024, 2048],
    #     sequence_length_values=[500, 1000],
    #     mutation_rate=0.1,
    #     p_values=tuple(np.logspace(-4, 0, 15)),
    #     bootstrap_reps=100,
    #     run_name="custom_experiment",
    #     compute_metrics_on_guardrails=False  # Optional: compute metrics when guardrails trigger
    # )
    
    # Create and run experiment
    runner = ExperimentRunner(cfg)
    run_dir, results = runner.run()
    
    print(f"\n{'='*80}")
    print(f"Experiment completed successfully!")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*80}")
    
    return run_dir, results


if __name__ == "__main__":
    main()
