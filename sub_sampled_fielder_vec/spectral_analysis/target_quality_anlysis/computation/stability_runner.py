"""Run multiple trials of diagnostics to assess statistical stability.

This module extends the single-run diagnostics to K independent trials,
computing mean, std, min, max for each metric, plus success rates for
partition validity.
"""
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys

# Add parent directories to path for imports
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from .computation_orchestrator import run_full_diagnostics


def run_stability_diagnostics(
    tree_config: Dict,
    seq_config: Dict,
    num_trials: int = 10,
    num_gaps: int = 1,
    min_split: int = 2,
    k_scree: int = 20,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Run K independent trials of full diagnostics and aggregate statistics.

    This function generates K different trees and sequence sets for the same
    configuration, computing metrics for each trial. It then reports mean ± std
    for numeric metrics and success rates for boolean metrics.

    Args:
        tree_config: Tree configuration dict with 'model' and 'params'
        seq_config: Sequence configuration dict with 'model', 'len', 'params'
        num_trials: Number of independent trials to run (K)
        num_gaps: Number of gap-based thresholds for partition
        min_split: Minimum partition size
        k_scree: Number of eigenvalues for scree plots (not used here)
        verbose: Print trial-by-trial progress

    Returns:
        Dictionary with aggregated statistics:
        - Metadata (tree_model, seq_model, n, L, mu)
        - For each numeric metric: mean, std, min, max
        - For partition_split: (mean_small, mean_large), std_small, std_large
        - For is_valid_partition: success_count, num_trials
    """
    # Extract metadata
    n = tree_config['params']['num_taxa']
    L = seq_config['len']
    mu = seq_config['params']['mutation_rate']

    if verbose:
        print(f"  Running {num_trials} trials for n={n}, L={L}, μ={mu}...")

    # Storage for trial results
    trial_results = []

    # Run K independent trials
    for trial_idx in range(num_trials):
        if verbose and num_trials > 1:
            print(f"    Trial {trial_idx + 1}/{num_trials}...", end="", flush=True)

        try:
            result = run_full_diagnostics(
                tree_config=tree_config,
                seq_config=seq_config,
                num_gaps=num_gaps,
                min_split=min_split,
                k_scree=k_scree
            )
            trial_results.append(result)

            if verbose and num_trials > 1:
                valid_str = "✓" if result['is_valid_partition'] else "✗"
                partition_split = result['partition_split']
                valid_label = "valid" if result['is_valid_partition'] else "invalid"
                print(f" {valid_str} partition={partition_split[0]}|{partition_split[1]} ({valid_label}), σ₂={result['sigma2']:.4f}")

        except Exception as e:
            if verbose:
                print(f" ✗ ERROR: {e}")
            continue

    if not trial_results:
        raise RuntimeError(f"All {num_trials} trials failed for n={n}, L={L}, mu={mu}")

    if len(trial_results) < num_trials:
        print(f"  ⚠️  Only {len(trial_results)}/{num_trials} trials succeeded")

    # Aggregate numeric metrics
    coherences = [r['coherence'] for r in trial_results]
    num_ranks = [r['numerical_rank'] for r in trial_results]
    sigma2s = [r['sigma2'] for r in trial_results]
    gaps = [r['spectral_gap'] for r in trial_results]
    rel_gaps = [r['relative_spectral_gap'] for r in trial_results]
    lambda2s = [r['lambda2'] for r in trial_results]
    lambda3s = [r['lambda3'] for r in trial_results]

    # Aggregate partition splits (extract small and large counts)
    partition_splits = [r['partition_split'] for r in trial_results]
    small_splits = [min(split) for split in partition_splits]
    large_splits = [max(split) for split in partition_splits]

    # Aggregate partition validity (boolean -> success rate)
    validity_results = [r['is_valid_partition'] for r in trial_results]
    success_count = sum(validity_results)

    # Compute statistics
    def compute_stats(values: List[float]) -> Dict[str, float]:
        """Helper to compute mean, std, min, max."""
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    # Build aggregated result
    aggregated = {
        # Metadata
        'tree_model': trial_results[0]['tree_model'],
        'seq_model': trial_results[0]['seq_model'],
        'n': n,
        'L': L,
        'mu': mu,
        'num_trials': len(trial_results),

        # Aggregated numeric metrics
        'coherence': compute_stats(coherences),
        'numerical_rank': compute_stats(num_ranks),
        'sigma2': compute_stats(sigma2s),
        'spectral_gap': compute_stats(gaps),
        'relative_spectral_gap': compute_stats(rel_gaps),
        'lambda2': compute_stats(lambda2s),
        'lambda3': compute_stats(lambda3s),

        # Partition split statistics
        'partition_split': {
            'mean_small': float(np.mean(small_splits)),
            'mean_large': float(np.mean(large_splits)),
            'std_small': float(np.std(small_splits)),
            'std_large': float(np.std(large_splits)),
            'min_small': int(np.min(small_splits)),
            'max_small': int(np.max(small_splits)),
            'min_large': int(np.min(large_splits)),
            'max_large': int(np.max(large_splits))
        },

        # Partition validity success rate
        'is_valid_partition': {
            'success_count': success_count,
            'num_trials': len(trial_results),
            'success_rate': success_count / len(trial_results)
        }
    }

    if verbose:
        valid_str = f"{success_count}/{len(trial_results)}"
        print(f"  ✓ Completed: σ₂={aggregated['sigma2']['mean']:.4f}±{aggregated['sigma2']['std']:.4f}, Valid={valid_str}")

    return aggregated
