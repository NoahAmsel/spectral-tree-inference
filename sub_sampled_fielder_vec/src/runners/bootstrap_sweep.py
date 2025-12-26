"""Bootstrap sweep logic for parameter estimation."""
import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import spectraltree

# Add spectral_analysis to path for partition validation
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SPECTRAL_ANALYSIS_PATH = PACKAGE_ROOT / "spectral_analysis" / "target_quality_anlysis" / "output"
if str(SPECTRAL_ANALYSIS_PATH) not in sys.path:
    sys.path.insert(0, str(SPECTRAL_ANALYSIS_PATH))
from partition_validity import check_partition_valid_in_tree

from ..core.utils import generate_sequences, align_fiedler_vector, compute_laplacian
from ..utils.metrics import (
    compute_sign_agreement,
    compute_partition_agreement,
    compute_fiedler_dot_product,
    compute_reference_partition_and_quality,
    metric_composer,
    estimate_operator_norm_diff,
    compute_ipr,
    compute_dk_ratio
)
from ..config import StructuredConfig
from ..models import get_tree_factory, get_sequence_factory
from ..utils.summaries import save_single_results, save_json
from ..utils.random_entries import _get_cached_similarity_matrix, _subsample_matrix_entries, compute_fiedler_from_similarity, compute_fiedler_from_laplacian
from ..utils.logging import log_info, log_warning, create_progress_bar, suppress_warnings
from ..utils.persistent_cache import (
    _get_cache_key,
    save_experiment_data,
    load_experiment_data
)

# Import middle-out runner (conditional on use_middle_out flag)
try:
    from .middle_out_runner import sweep_for_params_middle_out
    MIDDLE_OUT_AVAILABLE = True
except ImportError:
    MIDDLE_OUT_AVAILABLE = False
    log_warning('bootstrap', "Middle-out runner not available, falling back to sequential")


def align_fiedler_by_dot_product(fiedler_vector: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
    """
    Align a Fiedler vector using magnitude-based dot product alignment.

    Algorithm:
    1. Normalize both vectors
    2. Compute dot product
    3. If dot product is negative, flip the sign

    Args:
        fiedler_vector: Fiedler vector to align
        reference_vector: Reference vector for alignment

    Returns:
        Aligned and normalized Fiedler vector
    """
    from ..utils.metrics import _normalize_vector

    try:
        v_normalized = _normalize_vector(fiedler_vector)
        u_normalized = _normalize_vector(reference_vector)
    except ValueError as e:
        log_warning('align', f"Normalization failed: {e}")
        return fiedler_vector

    # Compute dot product (wrapped to catch numerical warnings)
    with suppress_warnings('align'):
        dot_product = np.dot(v_normalized, u_normalized)

    # Flip sign if needed
    if dot_product < 0:
        return -v_normalized
    else:
        return v_normalized


def check_guardrails_trigger(sign_agreements: List[float],
                              threshold: float = 99.9) -> bool:
    """
    Check if the last two sign agreements are both >= threshold.

    Returns True if guardrails should trigger (skip remaining computations).
    """
    if len(sign_agreements) < 2:
        return False
    return sign_agreements[-1] >= threshold and sign_agreements[-2] >= threshold


def _get_or_generate_experiment_data(
    cfg: StructuredConfig,
    n_taxa: int,
    seq_len: int
) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get experiment data from persistent cache or generate fresh.

    Single responsibility: Implements get-or-create pattern for experiment data.

    Args:
        cfg: Experiment configuration
        n_taxa: Number of taxa
        seq_len: Sequence length

    Returns:
        Tuple of (tree, observations, similarity_matrix, fiedler_ref)
    """
    # Try to load from persistent cache if enabled
    if cfg.cache.use_persistent_cache:
        cache_key = _get_cache_key(
            n_taxa=n_taxa,
            seq_len=seq_len,
            mutation_rate=cfg.sequence.params["mutation_rate"],
            tree_model_name=cfg.tree.model,
            seq_model_name=cfg.sequence.model
        )

        cached = load_experiment_data(cache_key)
        if cached is not None:
            log_info('cache', f"Loaded from persistent cache: {cache_key}", force=True)
            return (
                cached['tree'],
                cached['observations'],
                cached['similarity_matrix'],
                cached['fiedler_ref']
            )

        log_info('cache', f"Cache miss, generating fresh data for: {cache_key}", force=True)

    # Generate fresh data
    log_info('bootstrap', f"Generating tree and sequences...", force=True)
    tree_factory = get_tree_factory(cfg.tree.model, cfg.tree.params)
    tree = tree_factory()
    seq_factory = get_sequence_factory(cfg.sequence.model, cfg.sequence.params)
    seq_model = seq_factory()
    observations = generate_sequences(
        n_taxa, seq_len, cfg.sequence.params["mutation_rate"],
        tree_model=tree, seq_model=seq_model
    )

    log_info('bootstrap', "Computing full similarity matrix...", force=True)
    M = _get_cached_similarity_matrix(observations)

    log_info('bootstrap', "Computing reference Fiedler vector...", force=True)
    # Use new-style method: compute_fiedler_from_similarity(similarity_matrix)
    fiedler_ref = compute_fiedler_from_similarity(M)

    # Save to persistent cache if enabled
    if cfg.cache.use_persistent_cache:
        metadata = {
            'n_taxa': n_taxa,
            'seq_len': seq_len,
            'mutation_rate': cfg.sequence.params["mutation_rate"],
            'tree_model': cfg.tree.model,
            'seq_model': cfg.sequence.model,
            'seed': cfg.experiment.seed
        }

        save_experiment_data(
            cache_key=cache_key,
            tree=tree,
            observations=observations,
            similarity_matrix=M,
            fiedler_ref=fiedler_ref,
            metadata=metadata
        )

    return (tree, observations, M, fiedler_ref)


def sweep_for_params(
    cfg: StructuredConfig,
    n_taxa: int,
    seq_len: int,
    run_dir: str,
    incremental_save: bool = False,
    show_progress: bool = True,
    progress_callback: callable = None
) -> Tuple[np.ndarray, List[float], List[float], List[float], List[float],
           Dict[str, List[Tuple[float, float, float]]], float, List[float], List[float],
           List[Tuple[int, int] | None], List[Tuple[int, int] | None], List[str]]:
    """
    Run sweep for a specific (n_taxa, seq_len) combination.

    New bootstrap methodology:
    - For each p-value, collects aligned Fiedler vectors across bootstrap iterations
    - Averages the aligned vectors to create a single mean Fiedler vector
    - Computes partition-based agreement metrics

    Args:
        cfg: Experiment configuration
        n_taxa: Number of taxa
        seq_len: Sequence length
        run_dir: Directory for saving results
        incremental_save: Whether to save results incrementally
        show_progress: Whether to show bootstrap progress bars
        progress_callback: Optional callback function(p_idx) to update parent progress bar

    Returns:
        Tuple of (fiedler_ref, sign_agreements, partition_agreement_M,
                  partition_agreement_S, dot_products, metrics_dict, reference_partition_quality,
                  sigma2_avg_M_list, sigma2_avg_S_list, partition_split_M_list,
                  partition_split_S_list, result_source_list)
        - fiedler_ref: Reference Fiedler vector from full matrix (p=1.0)
        - sign_agreements: Legacy sign agreement metric (kept for comparison)
        - partition_agreement_M: Agreement using M for both partitions (ideal)
        - partition_agreement_S: Agreement using M vs S_avg (realistic)
        - dot_products: Vector alignment metric (0-1)
        - metrics_dict: Aggregated metrics (mean, median, std) for each metric type
        - reference_partition_quality: σ₂ quality score for reference partition
        - sigma2_avg_M_list: σ₂ quality scores using M
        - sigma2_avg_S_list: σ₂ quality scores using S_avg
        - partition_split_M_list: Partition split sizes (n_small, n_large) for M, or None
        - partition_split_S_list: Partition split sizes (n_small, n_large) for S_avg, or None
        - result_source_list: Source of each result ('computed', 'guardrail_high', 'p_is_one')
    """
    log_info('bootstrap', f"n={n_taxa}, L={seq_len} preparing experiment data…")

    # Get or generate experiment data (with optional persistent caching)
    tree, observations, M, fiedler_ref = _get_or_generate_experiment_data(cfg, n_taxa, seq_len)

    # Compute Laplacian of M once (for metrics that need it)
    L_M = compute_laplacian(M)

    # Compute reference partition and quality ONCE (constant across all p-values)
    reference_partition_split = None  # Track split sizes for reference partition
    try:
        partition_ref, reference_partition_quality, reference_partition_split = compute_reference_partition_and_quality(
            fiedler_ref, M,
            num_gaps=cfg.metrics.num_gaps,
            min_split=1  # Default min_split
        )
        log_info('bootstrap', f"Reference partition: {np.sum(partition_ref)} vs {np.sum(~partition_ref)} taxa (split: {reference_partition_split[0]}-{reference_partition_split[1]})")
        log_info('bootstrap', f"Reference partition quality (σ₂): {reference_partition_quality:.6f}")
        
        # Validate partition corresponds to a real tree edge (if enabled)
        if cfg.metrics.validate_partition_in_tree:
            log_info('bootstrap', "Validating reference partition against ground truth tree...")
            is_valid = check_partition_valid_in_tree(tree, partition_ref)
            if not is_valid:
                raise ValueError(
                    f"Reference Fiedler partition does not correspond to a valid tree edge. "
                    f"This indicates the partition is not a valid bipartition in the ground truth tree. "
                    f"Aborting experiment to prevent invalid results. "
                    f"Partition split: {reference_partition_split[0]}-{reference_partition_split[1]} taxa."
                )
            log_info('bootstrap', "✓ Reference partition is valid (corresponds to a real tree edge)")
    except Exception as e:
        log_warning('bootstrap', f"Failed to compute reference partition: {e}")
        reference_partition_quality = float('nan')
        partition_ref = None

    # Get config parameters for metrics
    empirical_rank_threshold = None  # Not used in current implementation
    coherence_k = cfg.metrics.coherence_k

    # Dispatch to middle-out parallel runner if enabled
    if cfg.experiment.use_middle_out and cfg.experiment.num_workers > 1:
        if not MIDDLE_OUT_AVAILABLE:
            log_warning('bootstrap', "Middle-out requested but not available, falling back to sequential")
        else:
            log_info('bootstrap', f"Using middle-out parallel processing with {cfg.experiment.num_workers} workers")

            # Compute M-based metrics once (needed for filling guardrail results)
            try:
                M_metrics = metric_composer(
                    M=M, S=M, L_M=L_M, L_S=L_M,
                    p=1.0,
                    empirical_rank_threshold=empirical_rank_threshold,
                    coherence_k=coherence_k
                )
                M_constants = {
                    'operator_norm_error': float('nan'),
                    'empirical_rank_M': M_metrics.get('empirical_rank_M', float('nan')),
                    'empirical_rank_L_M': M_metrics.get('empirical_rank_L_M', float('nan')),
                    'spectral_gap_M': M_metrics.get('spectral_gap_M', float('nan')),
                    'spectral_gap_L_M': M_metrics.get('spectral_gap_L_M', float('nan')),
                    'coherence_M': M_metrics.get('coherence_M', float('nan')),
                    'coherence_L_M': M_metrics.get('coherence_L_M', float('nan')),
                    'min_separation_M': M_metrics.get('min_separation_M', float('nan')),
                    'min_separation_L_M': M_metrics.get('min_separation_L_M', float('nan')),
                }
            except Exception as e:
                log_warning('bootstrap', f"Failed to compute M-based metrics: {e}")
                M_constants = {}

            # Call middle-out runner
            sign_agreements, partition_agreement_M, partition_agreement_S, dot_products, metrics_dict, \
            ref_quality, sigma2_avg_M_list, sigma2_avg_S_list, \
            partition_split_M_list, partition_split_S_list, result_source_list = \
                sweep_for_params_middle_out(
                    cfg=cfg,
                    n_taxa=n_taxa,
                    seq_len=seq_len,
                    M=M,
                    fiedler_ref=fiedler_ref,
                    partition_ref=partition_ref,
                    L_M=L_M,
                    M_constants=M_constants,
                    run_dir=run_dir,
                    reference_partition_split=reference_partition_split
                )

            return (fiedler_ref, sign_agreements, partition_agreement_M, partition_agreement_S, 
                    dot_products, metrics_dict, reference_partition_quality, 
                    sigma2_avg_M_list, sigma2_avg_S_list,
                    partition_split_M_list, partition_split_S_list, result_source_list)

    # Fall through to sequential processing
    log_info('bootstrap', "Using sequential processing")

    sign_agreements: List[float] = []            # Legacy metric (kept for comparison)
    partition_agreement_M: List[float] = []      # NEW: ideal scenario (both use M)
    partition_agreement_S: List[float] = []      # NEW: realistic scenario (M vs S_avg)
    dot_products: List[float] = []               # NEW: vector alignment
    sigma2_avg_M_list: List[float] = []          # NEW: sigma2 quality score using M
    sigma2_avg_S_list: List[float] = []          # NEW: sigma2 quality score using S_avg
    partition_split_M_list: List[Tuple[int, int] | None] = []  # NEW: partition split sizes for M
    partition_split_S_list: List[Tuple[int, int] | None] = []  # NEW: partition split sizes for S_avg
    result_source_list: List[str] = []           # NEW: 'computed', 'guardrail_high', 'p_is_one'

    # Initialize metric storage - all metrics for M, S, L_M, L_S
    # Plus new spectral metrics: lambda2/lambda3 eigenvalues, IPR, DK ratio
    metric_keys = [
        'operator_norm_error',
        'empirical_rank_M', 'empirical_rank_S', 'empirical_rank_L_M', 'empirical_rank_L_S',
        'spectral_gap_M', 'spectral_gap_S', 'spectral_gap_L_M', 'spectral_gap_L_S',
        'coherence_M', 'coherence_S', 'coherence_L_M', 'coherence_L_S',
        'min_separation_M', 'min_separation_S', 'min_separation_L_M', 'min_separation_L_S',
        # New spectral metrics
        'lambda2_L_M', 'lambda3_L_M', 'lambda2_L_S', 'lambda3_L_S',
        'ipr_S', 'dk_ratio_S'
    ]
    metrics_dict: Dict[str, List[Tuple[float, float, float]]] = {
        key: [] for key in metric_keys
    }

    # Progress milestones for logging
    def progress_milestones(total: int, k: int = 3) -> List[int]:
        """Calculate milestone indices for progress printing."""
        return sorted(set(np.linspace(0, total - 1, k, dtype=int).tolist()))

    milestones = progress_milestones(cfg.experiment.bootstrap_reps, 3)

    # OPTIMIZATION (Issue #10): Compute M-based metrics ONCE before p-value loop
    # These metrics are constant across all p-values since they depend only on M and L_M
    log_info('bootstrap', "Computing M-based metrics (once for all p-values)...")
    try:
        M_metrics = metric_composer(
            M=M, S=M, L_M=L_M, L_S=L_M,  # Use M/L_M for both since we're computing M metrics
            p=1.0,  # Not used for M metrics
            empirical_rank_threshold=empirical_rank_threshold,
            coherence_k=coherence_k
        )
        # Extract M and L_M metrics (ignore the S/L_S columns which are duplicates)
        M_constants = {
            'operator_norm_error': float('nan'),  # Not applicable for M alone
            'empirical_rank_M': M_metrics.get('empirical_rank_M', float('nan')),
            'empirical_rank_L_M': M_metrics.get('empirical_rank_L_M', float('nan')),
            'spectral_gap_M': M_metrics.get('spectral_gap_M', float('nan')),
            'spectral_gap_L_M': M_metrics.get('spectral_gap_L_M', float('nan')),
            'coherence_M': M_metrics.get('coherence_M', float('nan')),
            'coherence_L_M': M_metrics.get('coherence_L_M', float('nan')),
            'min_separation_M': M_metrics.get('min_separation_M', float('nan')),
            'min_separation_L_M': M_metrics.get('min_separation_L_M', float('nan')),
            # New spectral metrics for M/L_M
            'lambda2_L_M': M_metrics.get('lambda2_L_M', float('nan')),
            'lambda3_L_M': M_metrics.get('lambda3_L_M', float('nan')),
        }
        # Compute reference IPR from fiedler_ref (constant across p-values)
        ref_ipr = compute_ipr(fiedler_ref)
        M_constants['ipr_ref'] = ref_ipr
    except Exception as e:
        log_warning('bootstrap', f"Failed to compute M-based metrics: {e}")
        M_constants = {key: float('nan') for key in metric_keys}
        M_constants['ipr_ref'] = float('nan')

    for p_idx, p in enumerate(cfg.experiment.p_values):
        log_info('bootstrap', f"Processing p-value {p_idx+1}/{len(cfg.experiment.p_values)}: p={p:.4g}")

        agreements_for_p: List[float] = []

        # Initialize metric lists for this p-value - one list per metric
        metric_values: Dict[str, List[float]] = {
            key: [] for key in metric_keys
        }
        
        # Check if p=1.0 (or very close): S=M and L_S=L_M, no need to bootstrap
        p_is_one = p >= 0.9999
        
        if p_is_one:
            log_info('bootstrap', f"p={p:.4g} is 1.0 (or very close), skipping bootstrap (S=M deterministically)")
            # When p=1.0, S=M and L_S=L_M deterministically - no bootstrap needed
            # Sign agreement will be 100% (or very close)
            agreement = 100.0
            # p=1.0 means S=M, so partition splits are same as reference
            partition_split_M = reference_partition_split
            partition_split_S = reference_partition_split
            result_source = 'p_is_one'
            
            # Metrics: S/L_S are identical to M/L_M
            # Operator norm error is 0 since S=M
            for key in metric_keys:
                if key == 'operator_norm_error':
                    metric_values[key] = [0.0]
                elif key == 'ipr_S':
                    # IPR of reference Fiedler vector
                    metric_values[key] = [M_constants.get('ipr_ref', float('nan'))]
                elif key == 'dk_ratio_S':
                    # DK ratio is 0 when S=M (operator norm diff is 0)
                    metric_values[key] = [0.0]
                elif key.endswith('_L_S'):
                    # L_S metrics = L_M metrics (check _L_S first to avoid matching _S)
                    l_m_key = key.replace('_L_S', '_L_M')
                    metric_values[key] = [M_constants.get(l_m_key, float('nan'))]
                elif key.endswith('_S'):
                    # S metrics = M metrics
                    m_key = key.replace('_S', '_M')
                    metric_values[key] = [M_constants.get(m_key, float('nan'))]
                else:
                    # M/L_M metrics (already in M_constants, but we populate for consistency)
                    metric_values[key] = [M_constants.get(key, float('nan'))]
        else:
            # Normal bootstrap loop for p < 1.0
            # NEW ALGORITHM: Collect aligned vectors, average them, then compute single sign agreement
            aligned_vectors = []  # Collect aligned, normalized Fiedler vectors
            S_avg = None  # Running average of S matrices
            n_bootstrap_collected = 0  # Counter for streaming average

            # Create progress bar for bootstrap iterations only if requested (for debugging)
            bootstrap_pbar = None
            if show_progress:
                bootstrap_pbar = create_progress_bar(
                    total=cfg.experiment.bootstrap_reps,
                    desc=f"    Bootstraps p={p:.4g}",
                    unit='rep',
                    leave=False,
                    position=100  # High position so it doesn't interfere with config bars
                )

            for i in range(cfg.experiment.bootstrap_reps):

                # Set bootstrap-specific seed for reproducibility
                bootstrap_seed = cfg.experiment.seed + i

                # Compute S once per bootstrap rep
                S = _subsample_matrix_entries(M, p, seed=bootstrap_seed)

                # Update running average of S (streaming - no storage!)
                # Uses Welford's online algorithm for numerical stability
                if S_avg is None:
                    S_avg = S.copy()
                    n_bootstrap_collected = 1
                else:
                    n_bootstrap_collected += 1
                    S_avg += (S - S_avg) / n_bootstrap_collected

                # Compute L_S for metrics
                try:
                    L_S = compute_laplacian(S)
                except Exception as e:
                    log_warning('bootstrap', f"Failed to compute Laplacian of S: {e}")
                    L_S = None

                # Compute all metrics efficiently using the metric composer
                if L_S is not None:
                    try:
                        all_metrics = metric_composer(
                            M=M, S=S, L_M=L_M, L_S=L_S,
                            p=p,
                            empirical_rank_threshold=empirical_rank_threshold,
                            coherence_k=coherence_k
                        )
                    except Exception as e:
                        log_warning('bootstrap', f"Failed to compute metrics: {e}")
                        all_metrics = None
                else:
                    all_metrics = None

                # Store metrics (existing code)
                if all_metrics is None:
                    all_metrics = {key: float('nan') for key in metric_keys}

                for key in metric_keys:
                    # Skip new metrics that are computed separately below
                    if key in ('ipr_S', 'dk_ratio_S'):
                        continue
                    metric_values[key].append(all_metrics.get(key, float('nan')))

                # Compute Fiedler from L_S directly (reuse Laplacian computed for metrics)
                # This avoids recomputing the Laplacian (2x speedup on Laplacian work)
                if L_S is not None:
                    f_est = compute_fiedler_from_laplacian(L_S)
                else:
                    # Fallback: compute from similarity if Laplacian computation failed
                    f_est = compute_fiedler_from_similarity(S)

                # Align using dot product and normalize
                f_aligned_normalized = align_fiedler_by_dot_product(f_est, fiedler_ref)
                aligned_vectors.append(f_aligned_normalized)

                # NEW: Compute IPR immediately (before discarding vector)
                ipr_val = compute_ipr(f_aligned_normalized)
                metric_values['ipr_S'].append(ipr_val)

                # NEW: Compute DK ratio using power-iteration operator norm
                lambda2_val = all_metrics.get('lambda2_L_S', float('nan'))
                lambda3_val = all_metrics.get('lambda3_L_S', float('nan'))
                if not (np.isnan(lambda2_val) or np.isnan(lambda3_val)):
                    op_norm_diff = estimate_operator_norm_diff(S, M)
                    dk_ratio_val = compute_dk_ratio(op_norm_diff, lambda2_val, lambda3_val)
                else:
                    dk_ratio_val = float('nan')
                metric_values['dk_ratio_S'].append(dk_ratio_val)

                # Update bootstrap progress bar
                if bootstrap_pbar:
                    bootstrap_pbar.update(1)
            
            # After bootstrap loop: Average the aligned vectors
            log_info('bootstrap', f"Completed {cfg.experiment.bootstrap_reps} bootstrap iterations for p={p:.4g}")
            if len(aligned_vectors) > 0 and S_avg is not None:
                v_avg = np.mean(aligned_vectors, axis=0)

                # Normalize the averaged vector
                from ..utils.metrics import _normalize_vector
                try:
                    v_avg = _normalize_vector(v_avg)
                except ValueError as e:
                    log_warning('bootstrap', f"Averaged vector normalization failed: {e}")
                    # Fallback: zero vector (will produce 0% agreement)
                    v_avg = np.zeros_like(v_avg)

                # OLD metric (keep for comparison)
                sign_agreement = compute_sign_agreement(fiedler_ref, v_avg)

                # NEW METRIC 1: partition_agreement_M
                # Tests: How well does averaged Fiedler perform with clean STDR?
                # Uses pre-computed partition_ref, computes partition_taxa(fiedler_avg, M)
                if partition_ref is not None:
                    try:
                        partition_agr_M, sigma2_avg_M, partition_split_M = compute_partition_agreement(
                            partition_ref, v_avg, M,  # Use pre-computed partition_ref
                            num_gaps=cfg.metrics.num_gaps,
                            min_split=1
                        )
                    except Exception as e:
                        log_warning('bootstrap', f"partition_agreement (M) failed: {e}")
                        partition_agr_M = float('nan')
                        sigma2_avg_M = float('nan')
                        partition_split_M = None
                else:
                    partition_agr_M = float('nan')
                    sigma2_avg_M = float('nan')
                    partition_split_M = None

                # NEW METRIC 2: partition_agreement_S
                # Tests: Realistic scenario where test uses averaged subsampled data
                # Uses pre-computed partition_ref, computes partition_taxa(fiedler_avg, S_avg)
                if partition_ref is not None:
                    try:
                        partition_agr_S, sigma2_avg_S, partition_split_S = compute_partition_agreement(
                            partition_ref, v_avg, S_avg,  # Use pre-computed partition_ref
                            num_gaps=cfg.metrics.num_gaps,
                            min_split=1
                        )
                    except Exception as e:
                        log_warning('bootstrap', f"partition_agreement (S_avg) failed: {e}")
                        partition_agr_S = float('nan')
                        sigma2_avg_S = float('nan')
                        partition_split_S = None
                else:
                    partition_agr_S = float('nan')
                    sigma2_avg_S = float('nan')
                    partition_split_S = None
                
                result_source = 'computed'

                # NEW METRIC 3: Vector alignment
                try:
                    dot_prod = compute_fiedler_dot_product(fiedler_ref, v_avg)
                except Exception as e:
                    log_warning('bootstrap', f"dot_product failed: {e}")
                    dot_prod = float('nan')
            else:
                log_warning('bootstrap', f"No valid aligned vectors for p={p:.4g}")
                sign_agreement = 0.0
                partition_agr_M = 0.0
                partition_agr_S = 0.0
                dot_prod = 0.0
                sigma2_avg_M = float('nan')
                sigma2_avg_S = float('nan')
                partition_split_M = None
                partition_split_S = None
                result_source = 'computed'  # Still computed, just failed

            # Close bootstrap progress bar
            if bootstrap_pbar:
                bootstrap_pbar.close()

        # Store all metrics
        sign_agreements.append(float(sign_agreement))
        partition_agreement_M.append(float(partition_agr_M))
        partition_agreement_S.append(float(partition_agr_S))
        dot_products.append(float(dot_prod))
        sigma2_avg_M_list.append(float(sigma2_avg_M))
        sigma2_avg_S_list.append(float(sigma2_avg_S))
        partition_split_M_list.append(partition_split_M)
        partition_split_S_list.append(partition_split_S)
        result_source_list.append(result_source)
        
        split_M_str = f"{partition_split_M[0]}-{partition_split_M[1]}" if partition_split_M else "N/A"
        split_S_str = f"{partition_split_S[0]}-{partition_split_S[1]}" if partition_split_S else "N/A"
        log_info('bootstrap', f"p={p:.4g} results: sign={sign_agreement:.2f}%, part_M={partition_agr_M:.2f}%, part_S={partition_agr_S:.2f}%, dot={dot_prod:.4f}, σ2_M={sigma2_avg_M:.4f}, σ2_S={sigma2_avg_S:.4f}, split_M={split_M_str}, split_S={split_S_str}, source={result_source}")

        # Update parent progress bar via callback
        if progress_callback:
            progress_callback(p_idx)
        
        # Aggregate metrics
        def aggregate_metric(values: List[float]) -> Tuple[float, float, float]:
            """Aggregate metric values (mean, median, std), handling NaN."""
            clean_values = [v for v in values if not np.isnan(v)]
            if len(clean_values) == 0:
                return (float('nan'), float('nan'), float('nan'))
            return (
                float(np.mean(clean_values)),
                float(np.median(clean_values)),
                float(np.std(clean_values))
            )
        
        # Aggregate metrics
        # Metrics that are constant (M and L_M based): use constant values
        constant_metrics = [
            'empirical_rank_M', 'empirical_rank_L_M',
            'spectral_gap_M', 'spectral_gap_L_M',
            'coherence_M', 'coherence_L_M',
            'min_separation_M', 'min_separation_L_M',
            'lambda2_L_M', 'lambda3_L_M'
        ]
        
        # Metrics that vary per bootstrap (S and L_S based, plus operator_norm_error): aggregate
        varying_metrics = [
            'operator_norm_error',
            'empirical_rank_S', 'empirical_rank_L_S',
            'spectral_gap_S', 'spectral_gap_L_S',
            'coherence_S', 'coherence_L_S',
            'min_separation_S', 'min_separation_L_S',
            'lambda2_L_S', 'lambda3_L_S',
            'ipr_S', 'dk_ratio_S'
        ]
        
        # Store constant metrics
        for key in constant_metrics:
            val = M_constants.get(key, float('nan'))
            metrics_dict[key].append((float(val), float(val), 0.0))
        
        # Aggregate varying metrics
        for key in varying_metrics:
            values = metric_values.get(key, [])
            metrics_dict[key].append(aggregate_metric(values))
        
        # Check if guardrails trigger (two consecutive 100% agreements)
        if check_guardrails_trigger(sign_agreements):
            log_info('bootstrap', "Guardrails triggered: skipping remaining p-values (all 100%)")

            # Fill remaining p-values with perfect agreement values
            for remaining_p_idx in range(p_idx + 1, len(cfg.experiment.p_values)):
                remaining_p = cfg.experiment.p_values[remaining_p_idx]

                # Fill all agreement metrics with best values
                sign_agreements.append(100.0)
                partition_agreement_M.append(100.0)  # NEW
                partition_agreement_S.append(100.0)  # NEW
                dot_products.append(1.0)             # NEW - perfect alignment
                sigma2_avg_M_list.append(reference_partition_quality)  # Same as reference since v_avg = v_ref
                sigma2_avg_S_list.append(reference_partition_quality)  # Same as reference since S_avg = M
                partition_split_M_list.append(reference_partition_split)  # Same as reference
                partition_split_S_list.append(reference_partition_split)  # Same as reference
                result_source_list.append('guardrail_high')  # Filled by high-side guardrail

                # Handle metrics based on flag
                if cfg.guardrails.compute_metrics_on_trigger:
                    # Compute metrics for p=1.0 case (S=M, L_S=L_M)
                    # Operator norm error is 0 since S=M
                    for key in metric_keys:
                        if key == 'operator_norm_error':
                            metrics_dict[key].append((0.0, 0.0, 0.0))
                        elif key == 'ipr_S':
                            # IPR of reference Fiedler vector
                            val = M_constants.get('ipr_ref', float('nan'))
                            metrics_dict[key].append((float(val), float(val), 0.0))
                        elif key == 'dk_ratio_S':
                            # DK ratio is 0 when S=M
                            metrics_dict[key].append((0.0, 0.0, 0.0))
                        elif key.endswith('_L_S'):
                            # L_S metrics = L_M metrics (check _L_S first to avoid matching _S)
                            l_m_key = key.replace('_L_S', '_L_M')
                            val = M_constants.get(l_m_key, float('nan'))
                            metrics_dict[key].append((float(val), float(val), 0.0))
                        elif key.endswith('_S'):
                            # S metrics = M metrics
                            m_key = key.replace('_S', '_M')
                            val = M_constants.get(m_key, float('nan'))
                            metrics_dict[key].append((float(val), float(val), 0.0))
                        else:
                            # M/L_M metrics (constant values)
                            val = M_constants.get(key, float('nan'))
                            metrics_dict[key].append((float(val), float(val), 0.0))
                else:
                    # Fill with constant M metrics for M/L_M, NaN for varying metrics
                    for key in constant_metrics:
                        val = M_constants.get(key, float('nan'))
                        metrics_dict[key].append((float(val), float(val), 0.0))
                    for key in varying_metrics:
                        metrics_dict[key].append((float('nan'), float('nan'), float('nan')))

                # Update parent progress bar via callback for skipped p-values
                if progress_callback:
                    progress_callback(remaining_p_idx)

            # Break out of p-value loop
            break
        
        # Incremental save after each p-value is completed
        if incremental_save:
            # Save current progress
            current_p_values = cfg.experiment.p_values[:p_idx + 1]
            current_sign_agreements = sign_agreements
            current_partition_agreement_M = partition_agreement_M  # NEW
            current_partition_agreement_S = partition_agreement_S  # NEW
            current_dot_products = dot_products                    # NEW

            # Create current metrics dict with only completed p-values
            current_metrics_dict = {
                key: values[:p_idx + 1]
                for key, values in metrics_dict.items()
            }

            # Save as both single and taxa format for compatibility
            save_single_results(
                run_dir=run_dir,
                p_values=current_p_values,
                sign_agreements=current_sign_agreements,
                partition_agreement_M=current_partition_agreement_M,  # NEW
                partition_agreement_S=current_partition_agreement_S,  # NEW
                dot_products=current_dot_products,                    # NEW
                metrics_dict=current_metrics_dict
            )

            # Also save individual p-value results
            p_result = {
                "p": float(p),
                "sign_agreement": float(sign_agreement),
                "partition_agreement_M": float(partition_agr_M),  # NEW
                "partition_agreement_S": float(partition_agr_S),  # NEW
                "dot_product": float(dot_prod)                    # NEW
            }
            save_json(
                p_result,
                os.path.join(run_dir, f"p_{p:.0e}_n_{n_taxa}_L={seq_len}.json")
            )

            log_info('bootstrap', f"Saved incremental results for p={p:.4g}")

    return (fiedler_ref, sign_agreements, partition_agreement_M, partition_agreement_S, 
            dot_products, metrics_dict, reference_partition_quality, 
            sigma2_avg_M_list, sigma2_avg_S_list,
            partition_split_M_list, partition_split_S_list, result_source_list)

