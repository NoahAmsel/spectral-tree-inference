import spectraltree
import scipy.linalg
import numpy as np
import warnings
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from sklearn.decomposition import TruncatedSVD
from .logging import suppress_warnings, log_warning
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
    
    Args:
        similarity_matrix: Similarity matrix
        
    Returns:
        Laplacian matrix
    """
    degrees = np.sum(similarity_matrix, axis=0)
    D = np.diag(degrees)
    L = D - similarity_matrix
    return L


def compute_fielder_vector(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Fiedler vector with deterministic sign convention.
    
    This function now delegates to FiedlerVectorComputer for better organization.
    """
    return _fiedler_computer.compute(similarity_matrix)

def compute_fielder_for_sparse_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Fiedler vector for the Laplacian of the given similarity matrix.
    This version is optimized for sparse matrices.
    
    This function now delegates to FiedlerVectorComputer for better organization.
    """
    # Force sparse computation
    is_sparse_format = isinstance(similarity_matrix, csr_matrix)
    return _fiedler_computer._compute_sparse(similarity_matrix, is_sparse_format)

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