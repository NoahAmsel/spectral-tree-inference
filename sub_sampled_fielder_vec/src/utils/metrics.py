"""Metrics for spectral matrix analysis."""
import numpy as np
import scipy.linalg
import warnings
from typing import Dict, Tuple
from ..core.utils import compute_laplacian
from .logging import suppress_warnings, log_warning, log_error
from ..core.metric_computer import MetricComputer

# Set up warning filters to catch RuntimeWarnings from numpy/scipy
warnings.filterwarnings('always', category=RuntimeWarning)

# Global instance for backward compatibility
_metric_computer = None

def compute_sign_agreement(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute percentage sign agreement between two vectors.
    
    Args:
        vec1: Reference vector
        vec2: Vector to compare (should be aligned to vec1)
        
    Returns:
        Percentage of entries with matching signs (0-100)
    """
    # Convert to sign vectors
    sign1 = np.sign(vec1)
    sign2 = np.sign(vec2)
    
    # Handle zero entries (keep their sign as 0)
    # Count matches
    matches = np.sum(sign1 == sign2)
    total = len(vec1)
    
    # Return percentage
    return 100.0 * matches / total if total > 0 else 0.0



# ============================================================================
# Internal helper functions for individual metrics
# ============================================================================

def _compute_operator_norm_error(M: np.ndarray, S: np.ndarray, p: float) -> float:
    """
    Compute operator (spectral) norm of scaled error matrix: E_scaled = S - M.
    
    The operator norm is the largest singular value of a matrix, which represents
    the maximum factor by which the matrix stretches any unit vector.
    
    Note: S is already scaled by 1/p during subsampling (see _subsample_matrix_entries),
    so the error matrix is simply S - M, not (1/p)S - M.
    
    Args:
        M: Full similarity matrix (original)
        S: Subsampled similarity matrix (estimated, already scaled by 1/p)
        p: Sampling probability (used for validation, not for scaling)
        
    Returns:
        Operator (spectral) norm of E_scaled = S - M (largest singular value)
    """
    if p <= 0:
        raise ValueError(f"Sampling probability p must be positive, got {p}")
    
    # S is already scaled by 1/p, so error is simply S - M
    E_scaled = S - M
    return float(np.linalg.norm(E_scaled, ord=2))


def _compute_matrix_coherence_from_svd(U: np.ndarray, k: int | None = None) -> float:
    """
    Compute matrix coherence from pre-computed left singular vectors.
    
    Coherence is defined as: max_i ||u_i||_∞^2 where u_i are the top-k left singular vectors.
    
    Args:
        U: Left singular vectors (columns are singular vectors)
        k: Number of top singular vectors to consider. If None, uses all available.
        
    Returns:
        Maximum coherence value across top-k singular vectors
    """
    if U.shape[1] == 0:
        return 0.0
    
    # Use all available vectors if k is None or larger than available
    if k is None or k > U.shape[1]:
        k = U.shape[1]
    
    # Compute coherence for each of the top-k singular vectors
    coherences = []
    for i in range(k):
        u = U[:, i]
        # Maximum absolute squared entry
        coherence_i = float(np.max(np.abs(u) ** 2))
        coherences.append(coherence_i)
    
    # Return maximum coherence
    return float(np.max(coherences)) if coherences else 0.0




# ============================================================================
# Optimized computation helpers
# ============================================================================

def _compute_svd_results(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SVD once and return all components.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Tuple of (U, singular_values, Vt)
    """
    # Suppress warnings and log in standardized format
    with suppress_warnings('metrics'):
        try:
            U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
        except (np.linalg.LinAlgError, ValueError) as e:
            log_error('metrics', f"SVD computation failed: {e}")
            raise
    return U, singular_values, Vt


def _compute_smallest_eigenvalues(matrix: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute smallest k eigenvalues efficiently.
    
    Args:
        matrix: Input symmetric matrix
        k: Number of smallest eigenvalues to compute
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted in ascending order
    """
    n = matrix.shape[0]
    k = min(k, n - 1)  # Can't compute more than n-1 eigenvalues
    
    # Suppress warnings and log in standardized format
    with suppress_warnings('metrics'):
        try:
            eigvals, eigvecs = scipy.linalg.eigh(matrix, subset_by_index=(0, k - 1))
        except (scipy.linalg.LinAlgError, ValueError) as e:
            log_error('metrics', f"Eigenvalue computation failed: {e}")
            raise
    return eigvals, eigvecs


# ============================================================================
# Metric Composer
# ============================================================================

def metric_composer(
    M: np.ndarray,
    S: np.ndarray,
    L_M: np.ndarray,
    L_S: np.ndarray,
    p: float,
    empirical_rank_threshold: float | None = None,
    coherence_k: int | None = None
) -> Dict[str, float]:
    """
    Metric Composer: Compute all metrics efficiently for M, S, L_M, and L_S.

    This function now delegates to MetricComputer for better organization.

    Returns:
        Dictionary with all metrics:
        - 'operator_norm_error': Operator (spectral) norm of S - M
        - For each matrix (M, S, L_M, L_S):
          - 'empirical_rank_{matrix}': Empirical rank
          - 'spectral_gap_{matrix}': Spectral gap
          - 'coherence_{matrix}': Matrix coherence
          - 'min_separation_{matrix}': Minimum separation
        - 'lambda2_L_M', 'lambda3_L_M': Raw eigenvalues for L_M
        - 'lambda2_L_S', 'lambda3_L_S': Raw eigenvalues for L_S
    """
    global _metric_computer

    # Create metric computer with specified parameters
    _metric_computer = MetricComputer(
        empirical_rank_threshold=empirical_rank_threshold,
        coherence_k=coherence_k if coherence_k is not None else 2
    )

    return _metric_computer.compute_all(M, S, L_M, L_S, p)


def estimate_operator_norm_diff(S: np.ndarray, M: np.ndarray, n_iter: int = 10) -> float:
    """
    Estimate ||S - M||_op via power iteration (avoids full SVD).

    Convenience wrapper around MetricComputer.estimate_operator_norm_diff.

    Args:
        S: Subsampled similarity matrix (already scaled by 1/p)
        M: Full similarity matrix
        n_iter: Number of power iterations (default 10)

    Returns:
        Estimated operator norm ||S - M||_op
    """
    return MetricComputer.estimate_operator_norm_diff(S, M, n_iter)


def compute_ipr(fiedler_vector: np.ndarray) -> float:
    """
    Compute Inverse Participation Ratio (IPR) of a vector.

    IPR = sum(v_i^4).  High IPR means vector is localized (sparse);
    low IPR means vector is delocalized (spread out).

    Args:
        fiedler_vector: Fiedler vector (should be normalized)

    Returns:
        IPR value (scalar)
    """
    return float(np.sum(fiedler_vector ** 4))


def compute_dk_ratio(
    operator_norm_diff: float,
    lambda2: float,
    lambda3: float,
    eps: float = 1e-12
) -> float:
    """
    Compute Davis-Kahan ratio: ||S - M||_op / (λ₂ - λ₃).

    If < 0.5, theory guarantees eigenvector recovery.
    If > 0.5, success is random luck.

    Args:
        operator_norm_diff: ||S - M||_op (from power iteration)
        lambda2: Second smallest eigenvalue of L_S
        lambda3: Third smallest eigenvalue of L_S
        eps: Small value to prevent division by zero

    Returns:
        DK ratio (scalar)
    """
    gap = lambda2 - lambda3
    if abs(gap) < eps:
        return float('inf')
    return float(operator_norm_diff / abs(gap))


# ============================================================================
# Partition-Based Metrics for STDR Quality Assessment
# ============================================================================

def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit length.

    Args:
        v: Input vector

    Returns:
        Normalized vector

    Raises:
        ValueError: If vector has zero or near-zero norm, with diagnostic info
    """
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        # Provide diagnostic information about the degenerate vector
        n_nan = np.sum(np.isnan(v))
        n_inf = np.sum(np.isinf(v))
        n_zero = np.sum(v == 0)
        diag_info = f"norm={norm:.2e}, nan_count={n_nan}, inf_count={n_inf}, zero_count={n_zero}/{len(v)}"
        raise ValueError(f"Vector has zero or near-zero norm: {diag_info}")
    return v / norm


def compute_partition_agreement(
    partition_ref: np.ndarray,
    fiedler_avg: np.ndarray,
    similarity_for_avg: np.ndarray,
    num_gaps: int = 1,
    min_split: int = 1
) -> Tuple[float, float, Tuple[int, int]]:
    """
    Compute partition agreement between reference partition and averaged Fiedler vector.
    
    OPTIMIZED: Takes pre-computed reference partition instead of recomputing it.
    
    This function is used TWICE to compute two agreement metrics:
    1. partition_agreement_M, sigma2_M, split_M = compute_partition_agreement(partition_ref, v_avg, M)
    2. partition_agreement_S, sigma2_S, split_S = compute_partition_agreement(partition_ref, v_avg, S_avg)

    Args:
        partition_ref: Pre-computed reference partition (from compute_reference_partition_and_quality)
        fiedler_avg: Averaged Fiedler vector from subsampled bootstraps
        similarity_for_avg: Similarity matrix to score test partition (M or S_avg)
        num_gaps: Number of gap-based thresholds to evaluate
        min_split: Minimum partition size

    Returns:
        Tuple of (agreement_percentage, sigma2_of_avg_partition, partition_split)
        - agreement: Percentage of taxa in matching partition (0-100)
        - sigma2: Quality score of the averaged partition
        - partition_split: Tuple of (n_small, n_large) where n_small <= n_large

    Raises:
        Exception: From partition_taxa if partitioning fails (propagates to caller)
    """
    from spectraltree.spectral_tree_reconstruction import partition_taxa, svd2

    # Compute partition from averaged Fiedler vector
    partition_avg = partition_taxa(fiedler_avg, similarity_for_avg, num_gaps, min_split)
    
    # Compute partition split sizes (sorted: smaller first)
    n_true = int(np.sum(partition_avg))
    n_false = len(partition_avg) - n_true
    partition_split = (min(n_true, n_false), max(n_true, n_false))
    
    # Compute σ₂ for this partition
    s_sliced = similarity_for_avg[partition_avg, :]
    s_sliced = s_sliced[:, ~partition_avg]
    sigma2 = float(svd2(s_sliced))

    # Compare partitions (handle both orientations: A|B ≡ B|A)
    matches_direct = np.sum(partition_ref == partition_avg)
    matches_flipped = np.sum(partition_ref != partition_avg)
    max_matches = max(matches_direct, matches_flipped)
    agreement = 100.0 * max_matches / len(partition_ref)

    return agreement, sigma2, partition_split


def compute_fiedler_dot_product(
    fiedler_full: np.ndarray,
    fiedler_avg: np.ndarray
) -> float:
    """
    Compute dot product between normalized Fiedler vectors.

    Args:
        fiedler_full: Reference Fiedler vector
        fiedler_avg: Averaged Fiedler vector

    Returns:
        Absolute dot product (0-1, higher = better alignment)

    Raises:
        ValueError: If either vector has zero norm or numerical issues occur
    """
    # Suppress numpy RuntimeWarnings (divide by zero, overflow, invalid value)
    # and instead raise a single informative ValueError if issues occur
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        try:
            # Normalize both vectors
            u_norm = _normalize_vector(fiedler_full)
            v_norm = _normalize_vector(fiedler_avg)

            # Compute dot product
            dot_prod = np.dot(u_norm, v_norm)

            # Check for numerical issues in the result
            if not np.isfinite(dot_prod):
                raise ValueError(f"Dot product is not finite: {dot_prod}")

            return float(np.abs(dot_prod))

        except ValueError as e:
            # Re-raise with context about which vector failed
            if "fiedler_full" in str(e) or "zero or near-zero norm" in str(e):
                # Determine which vector is problematic
                norm_full = np.linalg.norm(fiedler_full)
                norm_avg = np.linalg.norm(fiedler_avg)

                if norm_full < 1e-12:
                    raise ValueError(f"Reference Fiedler vector has degenerate norm: {str(e)}")
                elif norm_avg < 1e-12:
                    raise ValueError(f"Averaged Fiedler vector has degenerate norm: {str(e)}")
                else:
                    raise ValueError(f"Numerical issue in dot product computation: {str(e)}")
            else:
                raise


def compute_reference_partition_and_quality(
    fiedler_ref: np.ndarray,
    similarity: np.ndarray,
    num_gaps: int = 1,
    min_split: int = 1
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Compute reference partition and its quality score (σ₂) in one pass.
    
    This should be called ONCE per experiment, not per p-value, since
    the reference Fiedler vector and similarity matrix are constant.
    
    Args:
        fiedler_ref: Reference Fiedler vector from full data
        similarity: Full similarity matrix M
        num_gaps: Number of gap-based thresholds to evaluate
        min_split: Minimum partition size
    
    Returns:
        Tuple of (partition_boolean_array, sigma2_quality_score, partition_split)
        - partition: Boolean array indicating partition membership
        - sigma2: Second singular value of cross-partition submatrix (lower = better)
        - partition_split: Tuple of (n_small, n_large) where n_small <= n_large
    
    Raises:
        Exception: If partition_taxa fails
    """
    from spectraltree.spectral_tree_reconstruction import partition_taxa, svd2
    
    # Compute the reference partition
    partition = partition_taxa(fiedler_ref, similarity, num_gaps, min_split)
    
    # Compute partition split sizes (sorted: smaller first)
    n_true = int(np.sum(partition))
    n_false = len(partition) - n_true
    partition_split = (min(n_true, n_false), max(n_true, n_false))
    
    # Compute σ₂ quality score for this partition
    s_sliced = similarity[partition, :]
    s_sliced = s_sliced[:, ~partition]
    sigma2 = float(svd2(s_sliced))
    
    return partition, sigma2, partition_split
