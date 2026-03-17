import spectraltree
import scipy.linalg
import numpy as np
import warnings
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from sklearn.decomposition import TruncatedSVD
from ..utils.logging import suppress_warnings, log_warning
from .fiedler_computer import FiedlerVectorComputer

# Global instance for backward compatibility
_fiedler_computer = FiedlerVectorComputer()

def compute_similarity_matrix(observations, p, seed=None):
    """
    Compute the similarity matrix for the observations.
    """
    return spectraltree.JC_similarity_matrix(observations)

def generate_sequences(num_taxa, sequence_length, mutation_rate, tree_model, seq_model):
    """
    Generate sequences for the given parameters and tree model.
    """
    # tree_model should be a tree object (e.g., from spectraltree.balanced_binary, unrooted_pure_kingman_tree, etc.)
    # seq_model should be a substitution model object (e.g., spectraltree.Jukes_Cantor())
    # This matches the usage in the tutorial: simulate_sequences(n, tree_model=..., seq_model=..., ...)
    char_matrix, meta = spectraltree.simulate_sequences(
        seq_len=sequence_length,
        tree_model=tree_model,
        seq_model=seq_model,
        mutation_rate=mutation_rate
    )
    return char_matrix  # Return only the character matrix, not the tuple

def compute_laplacian(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Compute unnormalized Laplacian matrix from similarity matrix.

    L = D - similarity_matrix, where D is diagonal degree matrix.

    Handles both dense and sparse similarity matrices. For sparse inputs,
    returns a sparse Laplacian. For dense inputs, returns a dense Laplacian.

    Args:
        similarity_matrix: Similarity matrix (dense or sparse)

    Returns:
        Laplacian matrix (same format as input)
    """
    from scipy.sparse import issparse, diags as sparse_diags

    if issparse(similarity_matrix):
        # Sparse computation
        degrees = np.array(similarity_matrix.sum(axis=0)).flatten()
        L = sparse_diags(degrees) - similarity_matrix
        return L
    else:
        # Dense computation (original logic)
        degrees = np.sum(similarity_matrix, axis=0)
        D = np.diag(degrees)
        L = D - similarity_matrix
        return L


def compute_fielder_vector(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Fiedler vector with deterministic sign convention.

    Convenience wrapper that computes Laplacian from similarity matrix
    and then computes the Fiedler vector.

    Args:
        similarity_matrix: Similarity matrix S

    Returns:
        Fiedler vector with consistent sign convention
    """
    laplacian = compute_laplacian(similarity_matrix)
    return _fiedler_computer.compute(laplacian)


def compute_fiedler_from_laplacian(laplacian: np.ndarray) -> np.ndarray:
    """
    Compute Fiedler vector from pre-computed Laplacian matrix.

    This function is useful when you've already computed the Laplacian
    for other purposes (e.g., metrics) and want to avoid recomputing it.

    Args:
        laplacian: Pre-computed Laplacian matrix L = D - S

    Returns:
        Fiedler vector with consistent sign convention

    Example:
        >>> S = build_similarity_matrix(observations)
        >>> L = compute_laplacian(S)
        >>> # Use L for metrics
        >>> metrics = compute_metrics(L)
        >>> # Reuse L for Fiedler vector (avoids recomputing Laplacian)
        >>> f = compute_fiedler_from_laplacian(L)
    """
    return _fiedler_computer.compute(laplacian)


def compute_fielder_for_sparse_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Fiedler vector for sparse similarity matrices.

    Convenience wrapper that computes Laplacian from sparse similarity matrix
    and uses sparse solver. The FiedlerVectorComputer will automatically detect
    the sparse format and use the appropriate solver.

    Args:
        similarity_matrix: Sparse similarity matrix S

    Returns:
        Fiedler vector with consistent sign convention
    """
    # Compute Laplacian (handles sparse matrices efficiently)
    from scipy.sparse import issparse, diags as sparse_diags

    if issparse(similarity_matrix):
        # Use sparse Laplacian computation
        degrees = np.array(similarity_matrix.sum(axis=0)).flatten()
        laplacian = sparse_diags(degrees) - similarity_matrix
    else:
        laplacian = compute_laplacian(similarity_matrix)

    return _fiedler_computer.compute(laplacian)

def align_fiedler_vector(fiedler_vector: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
    """
    Align a single Fiedler vector to have consistent sign orientation with a reference vector.
    
    CRITICAL: Alignment is based on sign vectors (binary ±1 vectors), not the actual vectors.
    This ensures proper alignment for sign agreement metrics.

    Args:
        fiedler_vector: Fiedler vector to align
        reference_vector: Reference vector for alignment

    Returns:
        Aligned Fiedler vector (np.ndarray)
        
    This function now delegates to FiedlerVectorComputer for better organization.
    """
    return _fiedler_computer.align_vector(fiedler_vector, reference_vector)