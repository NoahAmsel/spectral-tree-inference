"""Metrics for spectral matrix analysis."""
import numpy as np
import scipy.linalg
import warnings
from typing import Dict, Tuple
from utils.utils import compute_laplacian
from .logging import suppress_warnings, log_warning, log_error
from .metric_computer import MetricComputer

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


def _compute_empirical_rank_from_svd(singular_values: np.ndarray, threshold: float | None = None) -> int:
    """
    Compute empirical rank from pre-computed singular values.
    
    Args:
        singular_values: Singular values from SVD
        threshold: Numerical threshold. If None, uses 1e-12 * max(singular_values)
        
    Returns:
        Number of singular values above threshold
    """
    # Set threshold if not provided
    if threshold is None:
        max_sv = np.max(singular_values)
        threshold = 1e-12 * max_sv if max_sv > 0 else 1e-12
    
    # Count singular values above threshold
    rank = int(np.sum(singular_values > threshold))
    return rank


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


def _compute_spectral_gap_from_eigenvalues(eigvals: np.ndarray) -> float:
    """
    Compute spectral gap from pre-computed eigenvalues.
    
    Spectral gap: difference between 2nd and 3rd smallest eigenvalues.
    
    Args:
        eigvals: Eigenvalues sorted in ascending order (smallest first)
        
    Returns:
        Spectral gap (lambda_2 - lambda_1) where lambda_1 is 2nd smallest, lambda_2 is 3rd smallest
    """
    if len(eigvals) < 3:
        return 0.0
    
    lambda_1 = float(eigvals[1])  # Second smallest (Fiedler eigenvalue for Laplacians)
    lambda_2 = float(eigvals[2])  # Third smallest
    gap = lambda_2 - lambda_1
    return gap


def _compute_minimum_separation_from_eigenvalues(eigvals: np.ndarray) -> float:
    """
    Compute minimum separation from pre-computed eigenvalues.
    
    Minimum separation: smallest gap between 2nd smallest eigenvalue and its neighbors.
    
    Args:
        eigvals: Eigenvalues sorted in ascending order (smallest first)
        
    Returns:
        Minimum separation around 2nd smallest eigenvalue
    """
    if len(eigvals) < 2:
        return 0.0
    
    lambda_0 = float(eigvals[0])  # Smallest eigenvalue
    lambda_1 = float(eigvals[1])  # Second smallest (Fiedler eigenvalue for Laplacians)
    
    if len(eigvals) >= 3:
        lambda_2 = float(eigvals[2])  # Third smallest
        gap_above = lambda_2 - lambda_1
    else:
        gap_above = float('inf')  # No eigenvalue above
    
    gap_below = lambda_1 - lambda_0
    
    # Return minimum of the two gaps
    min_separation = min(gap_below, gap_above)
    return min_separation


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
    """
    global _metric_computer

    # Create metric computer with specified parameters
    _metric_computer = MetricComputer(
        empirical_rank_threshold=empirical_rank_threshold,
        coherence_k=coherence_k if coherence_k is not None else 2
    )

    return _metric_computer.compute_all(M, S, L_M, L_S, p)


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
        ValueError: If vector has zero or near-zero norm
    """
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError(f"Vector has zero or near-zero norm: {norm}")
    return v / norm


def compute_partition_agreement(
    fiedler_full: np.ndarray,
    fiedler_avg: np.ndarray,
    similarity_for_full: np.ndarray,
    similarity_for_avg: np.ndarray,
    num_gaps: int = 1,
    min_split: int = 1
) -> float:
    """
    Compute partition agreement between two Fiedler vectors.

    This function is used TWICE to compute two agreement metrics:
    1. partition_agreement_M = compute_partition_agreement(v_full, v_avg, M, M)
    2. partition_agreement_S = compute_partition_agreement(v_full, v_avg, M, S_avg)

    Args:
        fiedler_full: Reference Fiedler vector from L(M)
        fiedler_avg: Averaged Fiedler vector from subsampled bootstraps
        similarity_for_full: Similarity matrix to score reference partition (typically M)
        similarity_for_avg: Similarity matrix to score test partition (M or S_avg)
        num_gaps: Number of gap-based thresholds to evaluate
        min_split: Minimum partition size

    Returns:
        Percentage of taxa in matching partition (0-100)

    Raises:
        Exception: From partition_taxa if partitioning fails (propagates to caller)
    """
    from spectraltree.spectral_tree_reconstruction import partition_taxa

    # Compute two partitions
    partition_full = partition_taxa(fiedler_full, similarity_for_full, num_gaps, min_split)
    partition_avg = partition_taxa(fiedler_avg, similarity_for_avg, num_gaps, min_split)

    # Compare (handle both orientations: A|B ≡ B|A)
    matches_direct = np.sum(partition_full == partition_avg)
    matches_flipped = np.sum(partition_full != partition_avg)
    max_matches = max(matches_direct, matches_flipped)

    return 100.0 * max_matches / len(fiedler_full)


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
        ValueError: If either vector has zero norm
    """
    u_norm = _normalize_vector(fiedler_full)
    v_norm = _normalize_vector(fiedler_avg)
    return float(np.abs(np.dot(u_norm, v_norm)))
