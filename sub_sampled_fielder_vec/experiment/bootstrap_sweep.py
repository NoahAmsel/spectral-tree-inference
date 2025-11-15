"""Bootstrap sweep logic for parameter estimation."""
import os
from typing import Tuple, List, Dict

import numpy as np
import spectraltree

from utils.utils import generate_sequences, align_fiedler_vector, compute_laplacian
from utils.metrics import compute_sign_agreement, metric_composer
from utils.experiment_config import Config, progress_milestones
from utils.summaries import save_single_results, save_json
from utils.random_entries import _get_cached_similarity_matrix, _subsample_matrix_entries
from utils.logging import log_info, log_warning


def align_fiedler_by_dot_product(fiedler_vector: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
    """
    Align a Fiedler vector using magnitude-based dot product alignment.
    
    Algorithm:
    1. Normalize both vectors
    2. Compute dot product
    3. If dot product is negative, flip the sign
    
    Args:
        fiedler_vector: Fiedler vector to align (will be normalized)
        reference_vector: Reference vector for alignment (will be normalized)
        
    Returns:
        Aligned and normalized Fiedler vector
    """
    # Normalize both vectors
    v_norm = np.linalg.norm(fiedler_vector)
    u_norm = np.linalg.norm(reference_vector)
    
    if v_norm < 1e-12:
        log_warning('align', "Fiedler vector has zero or near-zero norm, returning as-is")
        return fiedler_vector
    
    if u_norm < 1e-12:
        raise ValueError("Reference vector has zero or near-zero norm")
    
    v_normalized = fiedler_vector / v_norm
    u_normalized = reference_vector / u_norm
    
    # Compute dot product
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


def sweep_for_params(
    cfg: Config,
    n_taxa: int,
    seq_len: int,
    run_dir: str,
    incremental_save: bool = False
) -> Tuple[np.ndarray, List[float], Dict[str, List[Tuple[float, float, float]]]]:
    """
    Run sweep for a specific (n_taxa, seq_len) combination.
    
    New bootstrap methodology:
    - For each p-value, collects aligned Fiedler vectors across bootstrap iterations
    - Averages the aligned vectors to create a single mean Fiedler vector
    - Computes sign agreement once between the mean vector and reference
    
    Returns:
        Tuple of (fiedler_ref, sign_agreements, metrics_dict)
        - fiedler_ref: Reference Fiedler vector from full matrix (p=1.0)
        - sign_agreements: List of single agreement values per p-value
        - metrics_dict: Aggregated metrics (mean, median, std) for each metric type
    """
    log_info('bootstrap', f"n={n_taxa}, L={seq_len} building tree and sequences…")
    tree = cfg.tree_model(n_taxa)
    seq_model = cfg.seq_model()
    observations = generate_sequences(
        n_taxa, seq_len, cfg.mutation_rate,
        tree_model=tree, seq_model=seq_model
    )

    log_info('bootstrap', "Computing full similarity + Fiedler…")
    # Get full similarity matrix M (will be cached)
    M = _get_cached_similarity_matrix(observations)
    
    # Use the same method as bootstrap to ensure consistency
    fiedler_ref = cfg.fiedler_method(observations, p=1.0, **cfg.fiedler_method_kwargs)

    # Compute Laplacian of M once (for metrics that need it)
    L_M = compute_laplacian(M)
    
    # Get config parameters for metrics
    empirical_rank_threshold = getattr(cfg, 'empirical_rank_threshold', None)
    coherence_k = getattr(cfg, 'coherence_k', 2)

    sign_agreements: List[float] = []  # Single agreement value per p-value
    
    # Initialize metric storage - all metrics for M, S, L_M, L_S
    metric_keys = [
        'operator_norm_error',
        'empirical_rank_M', 'empirical_rank_S', 'empirical_rank_L_M', 'empirical_rank_L_S',
        'spectral_gap_M', 'spectral_gap_S', 'spectral_gap_L_M', 'spectral_gap_L_S',
        'coherence_M', 'coherence_S', 'coherence_L_M', 'coherence_L_S',
        'min_separation_M', 'min_separation_S', 'min_separation_L_M', 'min_separation_L_S'
    ]
    metrics_dict: Dict[str, List[Tuple[float, float, float]]] = {
        key: [] for key in metric_keys
    }
    
    milestones = progress_milestones(cfg.bootstrap_reps, cfg.progress_prints)
    
    for p_idx, p in enumerate(cfg.p_values):
        log_info('bootstrap', f"Computing sign agreement for p={p} , n={n_taxa}, L={seq_len}")
        agreements_for_p: List[float] = []
        
        # Initialize metric lists for this p-value - one list per metric
        metric_values: Dict[str, List[float]] = {
            key: [] for key in metric_keys
        }
        
        # Compute M-based metrics once per p-value (they don't depend on bootstrap rep)
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
            }
        except Exception as e:
            log_warning('bootstrap', f"Failed to compute M-based metrics: {e}")
            M_constants = {key: float('nan') for key in metric_keys}
        
        # Check if p=1.0 (or very close): S=M and L_S=L_M, no need to bootstrap
        p_is_one = p >= 0.9999
        
        if p_is_one:
            # When p=1.0, S=M and L_S=L_M deterministically - no bootstrap needed
            # Sign agreement will be 100% (or very close)
            agreement = 100.0
            
            # Metrics: S/L_S are identical to M/L_M
            # Operator norm error is 0 since S=M
            for key in metric_keys:
                if key == 'operator_norm_error':
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
            
            for i in range(cfg.bootstrap_reps):
                if i in milestones:
                    log_info('bootstrap', f"Bootstrap {i+1}/{cfg.bootstrap_reps} for p={p:.4g}…")
                
                # Set bootstrap-specific seed for reproducibility
                bootstrap_seed = cfg.seed + i
                
                # Compute subsampled similarity matrix S
                S = _subsample_matrix_entries(M, p, seed=bootstrap_seed)
                
                # Compute Laplacian of S
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
                
                # If computation failed, store NaN for all metrics
                if all_metrics is None:
                    all_metrics = {key: float('nan') for key in metric_keys}
                
                # Store all metric values
                for key in metric_keys:
                    metric_values[key].append(all_metrics.get(key, float('nan')))
                
                # Call the fiedler method directly to get estimated Fiedler vector
                f_est = cfg.fiedler_method(
                    observations, p,
                    seed=bootstrap_seed,
                    **cfg.fiedler_method_kwargs
                )
                
                # Align using dot product and normalize
                f_aligned_normalized = align_fiedler_by_dot_product(f_est, fiedler_ref)
                aligned_vectors.append(f_aligned_normalized)
            
            # After bootstrap loop: Average the aligned vectors
            if len(aligned_vectors) > 0:
                v_avg = np.mean(aligned_vectors, axis=0)
                
                # Normalize the averaged vector
                v_avg_norm = np.linalg.norm(v_avg)
                if v_avg_norm > 1e-12:
                    v_avg = v_avg / v_avg_norm
                else:
                    log_warning('bootstrap', f"Averaged vector has zero norm for p={p:.4g}")
                
                # Compute single sign agreement between averaged vector and reference
                agreement = compute_sign_agreement(fiedler_ref, v_avg)
            else:
                log_warning('bootstrap', f"No valid aligned vectors for p={p:.4g}")
                agreement = 0.0
        
        # Store single sign agreement value
        sign_agreements.append(float(agreement))
        log_info('bootstrap', f"p={p:.4g}  sign_agreement={agreement:.2f}%")
        
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
            'min_separation_M', 'min_separation_L_M'
        ]
        
        # Metrics that vary per bootstrap (S and L_S based, plus operator_norm_error): aggregate
        varying_metrics = [
            'operator_norm_error',
            'empirical_rank_S', 'empirical_rank_L_S',
            'spectral_gap_S', 'spectral_gap_L_S',
            'coherence_S', 'coherence_L_S',
            'min_separation_S', 'min_separation_L_S'
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
            log_info('bootstrap', f"Guardrails triggered at p={p:.4g}: Two consecutive 100% agreements detected")
            log_info('bootstrap', f"Filling remaining {len(cfg.p_values) - p_idx - 1} p-values with 100% agreement")
            
            # Fill remaining p-values with 100% agreement
            for remaining_p_idx in range(p_idx + 1, len(cfg.p_values)):
                remaining_p = cfg.p_values[remaining_p_idx]
                
                # Fill sign agreement with 100%
                sign_agreements.append(100.0)
                
                # Handle metrics based on flag
                if cfg.compute_metrics_on_guardrails:
                    # Compute metrics for p=1.0 case (S=M, L_S=L_M)
                    # Operator norm error is 0 since S=M
                    for key in metric_keys:
                        if key == 'operator_norm_error':
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
                
                log_info('bootstrap', f"p={remaining_p:.4g}  [guardrails] sign_agreement=100.00%")
            
            # Break out of p-value loop
            break
        
        # Incremental save after each p-value is completed
        if incremental_save:
            # Save current progress
            current_p_values = cfg.p_values[:p_idx + 1]
            current_sign_agreements = sign_agreements
            
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
                metrics_dict=current_metrics_dict
            )
            
            # Also save individual p-value results
            p_result = {
                "p": float(p),
                "sign_agreement": float(agreement)
            }
            save_json(
                p_result,
                os.path.join(run_dir, f"p_{p:.0e}_n_{n_taxa}_L={seq_len}.json")
            )
            
            log_info('bootstrap', f"Saved incremental results for p={p:.4g}")
            
    return fiedler_ref, sign_agreements, metrics_dict

