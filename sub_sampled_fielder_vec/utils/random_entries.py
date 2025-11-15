import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import spectraltree
import scipy.linalg
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
import hashlib

from sklearn.decomposition import TruncatedSVD
from .utils import compute_fielder_vector
from .logging import log_info
from .similarity_builder import SimilarityMatrixBuilder

# Global builder instance for backward compatibility
_similarity_builder = SimilarityMatrixBuilder()

def _get_observations_hash(observations):
    """
    Generate a hash key for the observations to use as cache key.
    
    This function now delegates to SimilarityMatrixBuilder for better organization.
    """
    return _similarity_builder._get_observations_hash(observations)

def _get_cached_similarity_matrix(observations):
    """
    Get cached similarity matrix if available, otherwise compute and cache it.
    
    This function now delegates to SimilarityMatrixBuilder for better organization.
    """
    return _similarity_builder.build_full(observations)

def clear_similarity_cache():
    """
    Clear the similarity matrix cache. Useful for memory management.
    
    This function now delegates to SimilarityMatrixBuilder for better organization.
    """
    _similarity_builder.clear_cache()


def _subsample_matrix_entries(matrix, p, seed=None):
    """
    Subsample matrix entries with probability p and scale by 1/p.
    
    Args:
        matrix: Input matrix to subsample
        p: Sampling probability (0 < p <= 1)
        seed: Random seed for reproducibility
        
    Returns:
        Subsampled matrix with entries scaled by 1/p
        
    This function now delegates to SimilarityMatrixBuilder for better organization.
    """
    return _similarity_builder._subsample(matrix, p, seed)


def compute_fiedler_estimate(observations, p, roughness_factor=1.0, 
                            min_similarity=0.0, use_raw_hamming=False, 
                            seed=None):
    """
    Compute Fiedler vector estimate using random entry subsampling approach.
    
    Args:
        observations: Phylogenetic sequence data (n_taxa x seq_len)
        p: Sampling probability for matrix entries (0 < p <= 1)
        roughness_factor: Exponent for roughness adjustment
        min_similarity: Minimum similarity threshold
        use_raw_hamming: Not used, kept for API compatibility
        seed: Random seed for reproducibility
        
    Returns:
        Estimated Fiedler vector
    """
    similarity_matrix = compute_similarity_matrix(observations, p, roughness_factor, 
                                                min_similarity, use_raw_hamming, seed)
    return compute_fielder_vector(similarity_matrix)


def compute_similarity_matrix(observations, p, roughness_factor=1.0, 
                            min_similarity=0.0, use_raw_hamming=False, 
                            seed=None):
    """
    Compute similarity matrix using random entry subsampling approach.
    [Kept for backward compatibility - consider using compute_fiedler_estimate instead]
    
    This function now delegates to SimilarityMatrixBuilder for better organization.
    Note: roughness_factor and use_raw_hamming are kept for API compatibility but not used.
    """
    return _similarity_builder.build_subsampled(observations, p, seed, min_similarity)