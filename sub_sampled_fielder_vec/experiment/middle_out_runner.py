"""Middle-out p-value processing with parallel execution."""
import os
import json
import time
from typing import Tuple, List, Dict
import multiprocessing
import numpy as np

from utils.experiment_config import Config
from utils.logging import log_info, log_warning
from .parallel_bootstrap import (
    process_p_value_worker,
    create_perfect_result,
    create_below_threshold_result
)


def save_progress_checkpoint(run_dir: str, results: List[Dict], p_values: List[float],
                             round_num: int, elapsed_total: float):
    """
    Save progress checkpoint to file for crash recovery.

    Args:
        run_dir: Output directory
        results: Current results list
        p_values: List of p-values
        round_num: Current round number
        elapsed_total: Total elapsed time
    """
    checkpoint = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'round': round_num,
        'elapsed_seconds': elapsed_total,
        'p_values': p_values,
        'results': []
    }

    # Extract key info from results (avoid saving large S_avg matrices)
    for i, result in enumerate(results):
        if result is None:
            checkpoint['results'].append({'p': p_values[i], 'status': 'pending'})
        elif np.isnan(result.get('sign_agreement', 0)):
            checkpoint['results'].append({'p': p_values[i], 'status': 'skipped'})
        else:
            checkpoint['results'].append({
                'p': p_values[i],
                'status': 'completed',
                'sign_agreement': result['sign_agreement'],
                'partition_agreement_M': result['partition_agreement_M'],
                'partition_agreement_S': result['partition_agreement_S'],
                'dot_product': result['dot_product']
            })

    checkpoint_path = os.path.join(run_dir, 'progress_checkpoint.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    log_info('middle_out', f"  💾 Checkpoint saved: {checkpoint_path}", force=True)


def check_guardrails_high(results: List[Dict], guardrails_metric: str) -> bool:
    """
    Check if high-side guardrails should trigger.

    Returns True if last two results both have metric >= 99.9%.

    Args:
        results: List of result dicts (must be in p-value order)
        guardrails_metric: Which metric to check

    Returns:
        True if guardrails triggered
    """
    if len(results) < 2:
        return False

    # Get last two non-None results
    valid_results = [r for r in results if r is not None and guardrails_metric in r]
    if len(valid_results) < 2:
        return False

    return (valid_results[-1][guardrails_metric] >= 99.9 and
            valid_results[-2][guardrails_metric] >= 99.9)


def check_guardrails_low(results: List[Dict], guardrails_metric: str, threshold: float, epsilon: float = 0.0) -> bool:
    """
    Check if low-side guardrails should trigger.

    Returns True if last two results both have metric < threshold + epsilon.

    Args:
        results: List of result dicts (must be in reverse p-value order for low side)
        guardrails_metric: Which metric to check
        threshold: Threshold value (default 50.0)
        epsilon: Epsilon margin for threshold (default 0.0)

    Returns:
        True if guardrails triggered
    """
    if len(results) < 2:
        return False

    # Get last two non-None results
    valid_results = [r for r in results if r is not None and guardrails_metric in r]
    if len(valid_results) < 2:
        return False

    # Check if both below threshold + epsilon (and not NaN)
    effective_threshold = threshold + epsilon
    last_two = valid_results[-2:]
    return all(
        not np.isnan(r[guardrails_metric]) and r[guardrails_metric] < effective_threshold
        for r in last_two
    )


def sweep_for_params_middle_out(
    cfg: Config,
    n_taxa: int,
    seq_len: int,
    M: np.ndarray,
    fiedler_ref: np.ndarray,
    L_M: np.ndarray,
    M_constants: Dict,
    run_dir: str
) -> Tuple[List[float], List[float], List[float], List[float], Dict[str, List[Tuple[float, float, float]]]]:
    """
    Process p-values using middle-out strategy with parallel execution.

    Strategy:
    1. Start at middle p-value
    2. Expand symmetrically: 4 workers on high side, 4 on low side
    3. Check guardrails after each round:
       - High side: two consecutive >= 99.9% -> stop, fill rest with 100%
       - Low side: two consecutive < threshold% -> stop, fill rest with NaN
    4. Reallocate workers if one side completes early

    Args:
        cfg: Configuration
        n_taxa: Number of taxa
        seq_len: Sequence length
        M: Full similarity matrix
        fiedler_ref: Reference Fiedler vector
        L_M: Laplacian of M
        M_constants: Pre-computed M-based metrics
        run_dir: Output directory

    Returns:
        Tuple of (sign_agreements, partition_agreement_M, partition_agreement_S,
                  dot_products, metrics_dict)
    """
    p_values = list(cfg.p_values)
    n_p_values = len(p_values)
    middle_idx = n_p_values // 2

    log_info('middle_out', f"Starting middle-out processing: {n_p_values} p-values, middle_idx={middle_idx}", force=True)
    log_info('middle_out', f"Progress checkpoints will be saved to: {run_dir}/progress_checkpoint.json", force=True)

    # Initialize results storage (None = not computed yet)
    results = [None] * n_p_values

    # Tracking
    high_complete = False
    low_complete = False
    experiment_start_time = time.time()

    # Phase 1: Compute middle seed
    log_info('middle_out', f"\nPhase 1: Computing middle seed p[{middle_idx}]={p_values[middle_idx]:.4g}", force=True)
    log_info('middle_out', f"  This will take ~{cfg.bootstrap_reps * 6}s for {cfg.bootstrap_reps} bootstrap reps...", force=True)

    round_start_time = time.time()
    results[middle_idx] = process_p_value_worker(M, fiedler_ref, L_M, p_values[middle_idx], cfg)
    elapsed = time.time() - round_start_time

    log_info('middle_out',
             f"Middle result (took {elapsed:.1f}s): {cfg.guardrails_metric}={results[middle_idx][cfg.guardrails_metric]:.2f}%",
             force=True)

    # Save initial checkpoint
    save_progress_checkpoint(run_dir, results, p_values, 0, time.time() - experiment_start_time)

    # Phase 2: Expand outward in rounds
    workers_per_side = cfg.num_workers // 2  # Split workers between directions
    offset = 1
    round_num = 1

    with multiprocessing.Pool(cfg.num_workers) as pool:
        while not (high_complete and low_complete):
            to_process = []  # List of (index, direction, p_value)

            log_info('middle_out', f"\nRound {round_num}: offset={offset}", force=True)

            # Collect high-side indices (if not complete)
            if not high_complete:
                high_start = middle_idx + offset
                high_end = min(middle_idx + offset + workers_per_side, n_p_values)
                high_indices = list(range(high_start, high_end))

                if high_indices:
                    log_info('middle_out', f"  High side: indices {high_indices} (p={[p_values[i] for i in high_indices]})", force=True)
                    to_process.extend([(i, 'high', p_values[i]) for i in high_indices])
                else:
                    log_info('middle_out', "  High side: no more p-values", force=True)
                    high_complete = True

            # Collect low-side indices (if not complete)
            if not low_complete:
                low_start = middle_idx - offset
                low_end = max(middle_idx - offset - workers_per_side, -1)
                low_indices = list(range(low_start, low_end, -1))

                if low_indices:
                    log_info('middle_out', f"  Low side: indices {low_indices} (p={[p_values[i] for i in low_indices]})", force=True)
                    to_process.extend([(i, 'low', p_values[i]) for i in low_indices])
                else:
                    log_info('middle_out', "  Low side: no more p-values", force=True)
                    low_complete = True

            if not to_process:
                log_info('middle_out', "No more work, breaking", force=True)
                break

            # Process chunk in parallel
            log_info('middle_out', f"  Processing {len(to_process)} p-values in parallel...", force=True)

            round_start_time = time.time()
            worker_args = [(M, fiedler_ref, L_M, p, cfg) for (_, _, p) in to_process]
            chunk_results = pool.starmap(process_p_value_worker, worker_args)
            elapsed = time.time() - round_start_time

            log_info('middle_out', f"  ✓ Completed in {elapsed:.1f}s", force=True)

            # Store results
            for (idx, direction, p), result in zip(to_process, chunk_results):
                results[idx] = result
                log_info('middle_out',
                        f"    p[{idx}]={p:.4g}: {cfg.guardrails_metric}={result[cfg.guardrails_metric]:.2f}%",
                        force=True)

            # Save checkpoint after each round
            save_progress_checkpoint(run_dir, results, p_values, round_num, time.time() - experiment_start_time)

            # Check guardrails for high side
            if not high_complete:
                # Get results for high side in order
                high_results_so_far = [results[i] for i in range(middle_idx, n_p_values) if results[i] is not None]

                if check_guardrails_high(high_results_so_far, cfg.guardrails_metric):
                    log_info('middle_out', "  🛑 High-side guardrails triggered!", force=True)
                    high_complete = True

                    # Fill remaining high p-values with perfect results
                    for i in range(middle_idx + 1, n_p_values):
                        if results[i] is None:
                            results[i] = create_perfect_result(p_values[i], M_constants)
                            log_info('middle_out', f"    Filled p[{i}]={p_values[i]:.4g} with 100%", force=True)

            # Check guardrails for low side
            if not low_complete:
                # Get results for low side in reverse order (descending p)
                low_results_so_far = [results[i] for i in range(middle_idx, -1, -1) if results[i] is not None]

                if check_guardrails_low(low_results_so_far, cfg.guardrails_metric, cfg.low_side_threshold, cfg.low_side_epsilon):
                    effective_threshold = cfg.low_side_threshold + cfg.low_side_epsilon
                    log_info('middle_out', f"  🛑 Low-side guardrails triggered (< {effective_threshold:.2f}%)!", force=True)
                    low_complete = True

                    # Fill remaining low p-values
                    for i in range(middle_idx - 1, -1, -1):
                        if results[i] is None:
                            results[i] = create_below_threshold_result(p_values[i])
                            log_info('middle_out', f"    Filled p[{i}]={p_values[i]:.4g} with 50% (below threshold)", force=True)

            # If one side is complete, reallocate workers to the other side
            if high_complete and not low_complete:
                workers_per_side = cfg.num_workers  # All workers to low side
                log_info('middle_out', "  Reallocating all workers to low side", force=True)
            elif low_complete and not high_complete:
                workers_per_side = cfg.num_workers  # All workers to high side
                log_info('middle_out', "  Reallocating all workers to high side", force=True)

            offset += workers_per_side if not (high_complete and low_complete) else cfg.num_workers // 2
            round_num += 1

    # Extract results into lists (in p-value order)
    sign_agreements = []
    partition_agreement_M = []
    partition_agreement_S = []
    dot_products = []
    metrics_dict = {key: [] for key in results[middle_idx]['metrics'].keys()}

    for i, result in enumerate(results):
        if result is None:
            # Should not happen, but handle gracefully
            log_warning('middle_out', f"Result for p[{i}]={p_values[i]:.4g} is None!")
            sign_agreements.append(float('nan'))
            partition_agreement_M.append(float('nan'))
            partition_agreement_S.append(float('nan'))
            dot_products.append(float('nan'))
            for key in metrics_dict.keys():
                metrics_dict[key].append((float('nan'), float('nan'), float('nan')))
        else:
            sign_agreements.append(result['sign_agreement'])
            partition_agreement_M.append(result['partition_agreement_M'])
            partition_agreement_S.append(result['partition_agreement_S'])
            dot_products.append(result['dot_product'])
            for key, value in result['metrics'].items():
                metrics_dict[key].append(value)

    # Final summary
    total_elapsed = time.time() - experiment_start_time
    num_computed = sum(1 for r in results if r is not None and not np.isnan(r['sign_agreement']))
    num_skipped = sum(1 for r in results if r is not None and np.isnan(r['sign_agreement']))

    log_info('middle_out', f"\n{'='*80}", force=True)
    log_info('middle_out', f"Middle-out processing complete!", force=True)
    log_info('middle_out', f"  Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f}s)", force=True)
    log_info('middle_out', f"  Computed: {num_computed}/{n_p_values} p-values", force=True)
    log_info('middle_out', f"  Skipped (guardrails): {num_skipped} p-values", force=True)
    log_info('middle_out', f"  Speedup vs sequential: ~{n_p_values * total_elapsed / num_computed / total_elapsed:.1f}x", force=True)
    log_info('middle_out', f"{'='*80}", force=True)

    # Final checkpoint
    save_progress_checkpoint(run_dir, results, p_values, round_num, total_elapsed)

    return (sign_agreements, partition_agreement_M, partition_agreement_S,
            dot_products, metrics_dict)
