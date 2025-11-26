"""Worker functions for parallel p-value processing."""
from typing import Dict, List, Tuple
import numpy as np

from utils.experiment_config import Config
from utils.random_entries import _subsample_matrix_entries, compute_fiedler_from_similarity, compute_fiedler_from_laplacian
from utils.utils import compute_laplacian
from utils.metrics import (
    compute_sign_agreement,
    compute_partition_agreement,
    compute_fiedler_dot_product,
    metric_composer,
    _normalize_vector
)
from utils.logging import log_info, log_warning, suppress_warnings


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


def process_p_value_worker(
    M: np.ndarray,
    fiedler_ref: np.ndarray,
    L_M: np.ndarray,
    p: float,
    cfg: Config
) -> Dict:
    """
    Worker function: processes one complete p-value with all bootstrap iterations.

    This function performs the entire bootstrap loop for a single p-value,
    computing all metrics and aggregating results.

    Args:
        M: Full similarity matrix (read-only, shared across workers)
        fiedler_ref: Reference Fiedler vector from M
        L_M: Laplacian of M (pre-computed, read-only)
        p: P-value to process
        cfg: Configuration object

    Returns:
        Dictionary containing:
            - 'p': p-value
            - 'sign_agreement': Legacy metric
            - 'partition_agreement_M': Partition agreement (M vs M)
            - 'partition_agreement_S': Partition agreement (M vs S_avg)
            - 'dot_product': Vector alignment metric
            - 'metrics': Dict of aggregated metrics (mean, median, std)
            - 'S_avg': Average subsampled matrix (for further analysis)
    """
    # Get config parameters
    empirical_rank_threshold = getattr(cfg, 'empirical_rank_threshold', None)
    coherence_k = getattr(cfg, 'coherence_k', 2)

    # Initialize metric storage
    metric_keys = [
        'operator_norm_error',
        'empirical_rank_M', 'empirical_rank_S', 'empirical_rank_L_M', 'empirical_rank_L_S',
        'spectral_gap_M', 'spectral_gap_S', 'spectral_gap_L_M', 'spectral_gap_L_S',
        'coherence_M', 'coherence_S', 'coherence_L_M', 'coherence_L_S',
        'min_separation_M', 'min_separation_S', 'min_separation_L_M', 'min_separation_L_S'
    ]

    # Compute M-based metrics once (constant across bootstrap reps)
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
        log_warning('worker', f"Failed to compute M-based metrics: {e}")
        M_constants = {key: float('nan') for key in metric_keys}

    # Check if p=1.0 (no bootstrap needed)
    p_is_one = p >= 0.9999

    if p_is_one:
        # Deterministic case: S=M, no bootstrap needed
        sign_agreement = 100.0
        partition_agr_M = 100.0
        partition_agr_S = 100.0
        dot_prod = 1.0
        S_avg = M.copy()

        # Metrics: S/L_S identical to M/L_M
        metric_values = {key: [M_constants.get(key, float('nan'))] for key in metric_keys}
        metric_values['operator_norm_error'] = [0.0]
    else:
        # Bootstrap loop
        aligned_vectors = []
        S_avg = None
        n_bootstrap_collected = 0
        metric_values = {key: [] for key in metric_keys}

        for i in range(cfg.bootstrap_reps):
            # Set bootstrap-specific seed
            bootstrap_seed = cfg.seed + i

            # Subsample matrix
            S = _subsample_matrix_entries(M, p, seed=bootstrap_seed)

            # Update running average of S (Welford's algorithm)
            if S_avg is None:
                S_avg = S.copy()
                n_bootstrap_collected = 1
            else:
                n_bootstrap_collected += 1
                S_avg += (S - S_avg) / n_bootstrap_collected

            # Compute Laplacian
            try:
                L_S = compute_laplacian(S)
            except Exception as e:
                log_warning('worker', f"Failed to compute Laplacian: {e}")
                L_S = None

            # Compute metrics
            if L_S is not None:
                try:
                    all_metrics = metric_composer(
                        M=M, S=S, L_M=L_M, L_S=L_S,
                        p=p,
                        empirical_rank_threshold=empirical_rank_threshold,
                        coherence_k=coherence_k
                    )
                except Exception as e:
                    log_warning('worker', f"Failed to compute metrics: {e}")
                    all_metrics = None
            else:
                all_metrics = None

            if all_metrics is None:
                all_metrics = {key: float('nan') for key in metric_keys}

            for key in metric_keys:
                metric_values[key].append(all_metrics.get(key, float('nan')))

            # Compute and align Fiedler vector (reuse Laplacian if available)
            if L_S is not None:
                f_est = compute_fiedler_from_laplacian(L_S)
            else:
                # Fallback: compute from similarity if Laplacian computation failed
                f_est = compute_fiedler_from_similarity(S)
            f_aligned = align_fiedler_by_dot_product(f_est, fiedler_ref)
            aligned_vectors.append(f_aligned)

        # Aggregate aligned vectors
        if len(aligned_vectors) > 0 and S_avg is not None:
            v_avg = np.mean(aligned_vectors, axis=0)

            # Normalize
            try:
                v_avg = _normalize_vector(v_avg)
            except ValueError as e:
                log_warning('worker', f"Averaged vector normalization failed: {e}")
                v_avg = np.zeros_like(v_avg)

            # Compute agreement metrics
            sign_agreement = compute_sign_agreement(fiedler_ref, v_avg)

            try:
                partition_agr_M = compute_partition_agreement(
                    fiedler_ref, v_avg, M, M,
                    num_gaps=getattr(cfg, 'num_gaps', 1),
                    min_split=getattr(cfg, 'min_split', 1)
                )
            except Exception as e:
                log_warning('worker', f"partition_agreement (M) failed: {e}")
                partition_agr_M = float('nan')

            try:
                partition_agr_S = compute_partition_agreement(
                    fiedler_ref, v_avg, M, S_avg,
                    num_gaps=getattr(cfg, 'num_gaps', 1),
                    min_split=getattr(cfg, 'min_split', 1)
                )
            except Exception as e:
                log_warning('worker', f"partition_agreement (S_avg) failed: {e}")
                partition_agr_S = float('nan')

            try:
                dot_prod = compute_fiedler_dot_product(fiedler_ref, v_avg)
            except Exception as e:
                log_warning('worker', f"dot_product failed: {e}")
                dot_prod = float('nan')
        else:
            log_warning('worker', f"No valid aligned vectors for p={p:.4g}")
            sign_agreement = 0.0
            partition_agr_M = 0.0
            partition_agr_S = 0.0
            dot_prod = 0.0

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

    # Constant vs varying metrics
    constant_metrics = [
        'empirical_rank_M', 'empirical_rank_L_M',
        'spectral_gap_M', 'spectral_gap_L_M',
        'coherence_M', 'coherence_L_M',
        'min_separation_M', 'min_separation_L_M'
    ]

    varying_metrics = [
        'operator_norm_error',
        'empirical_rank_S', 'empirical_rank_L_S',
        'spectral_gap_S', 'spectral_gap_L_S',
        'coherence_S', 'coherence_L_S',
        'min_separation_S', 'min_separation_L_S'
    ]

    aggregated_metrics = {}

    # Store constant metrics
    for key in constant_metrics:
        val = M_constants.get(key, float('nan'))
        aggregated_metrics[key] = (float(val), float(val), 0.0)

    # Aggregate varying metrics
    for key in varying_metrics:
        values = metric_values.get(key, [])
        aggregated_metrics[key] = aggregate_metric(values)

    return {
        'p': float(p),
        'sign_agreement': float(sign_agreement),
        'partition_agreement_M': float(partition_agr_M),
        'partition_agreement_S': float(partition_agr_S),
        'dot_product': float(dot_prod),
        'metrics': aggregated_metrics,
        'S_avg': S_avg
    }


def create_perfect_result(p: float, M_constants: Dict) -> Dict:
    """Create a result dict for p-values that would be 100% (guardrails triggered)."""
    metric_keys = list(M_constants.keys())
    aggregated_metrics = {}

    for key in metric_keys:
        if key == 'operator_norm_error':
            aggregated_metrics[key] = (0.0, 0.0, 0.0)
        else:
            val = M_constants.get(key, float('nan'))
            aggregated_metrics[key] = (float(val), float(val), 0.0)

    return {
        'p': float(p),
        'sign_agreement': 100.0,
        'partition_agreement_M': 100.0,
        'partition_agreement_S': 100.0,
        'dot_product': 1.0,
        'metrics': aggregated_metrics,
        'S_avg': None
    }


def create_below_threshold_result(p: float) -> Dict:
    """Create a result dict for p-values below threshold (low-side guardrails)."""
    return {
        'p': float(p),
        'sign_agreement': float('nan'),
        'partition_agreement_M': float('nan'),
        'partition_agreement_S': float('nan'),
        'dot_product': float('nan'),
        'metrics': {},
        'S_avg': None
    }
