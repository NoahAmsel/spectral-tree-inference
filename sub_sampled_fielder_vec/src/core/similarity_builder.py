"""Similarity matrix construction and subsampling."""
import hashlib
import numpy as np
import spectraltree

from ..utils.logging import log_info, suppress_warnings
from .sampling import get_sampler


class SimilarityMatrixBuilder:
    """
    Encapsulates similarity matrix construction and subsampling pipeline.
    
    Manages caching and provides clean methods for building full and subsampled matrices.
    Supports multiple sampling methods via the method parameter.
    """
    
    def __init__(self, method: str = "uniform", **method_kwargs):
        """
        Initialize the similarity matrix builder.
        
        Args:
            method: Sampling method ("uniform" or "leveraged")
            **method_kwargs: Method-specific parameters passed to sampler
        """
        self._cache = {}
        self.sampler = get_sampler(method, **method_kwargs)
    
    def build_full(self, observations: np.ndarray) -> np.ndarray:
        """
        Build full similarity matrix (with caching).
        
        Args:
            observations: Sequence observations (n_taxa x seq_len)
            
        Returns:
            Full similarity matrix
        """
        obs_hash = self._get_observations_hash(observations)
        
        if obs_hash not in self._cache:
            log_info('cache', "Computing and caching full similarity matrix...")
            # Wrap similarity computation to capture numerical warnings
            with suppress_warnings('similarity'):
                full_similarity = spectraltree.JC_similarity_matrix(observations)
            self._cache[obs_hash] = full_similarity
        
        return self._cache[obs_hash]
    
    def build_subsampled(self, observations: np.ndarray, p: float, seed: int = None,
                        min_similarity: float = 0.0) -> np.ndarray:
        """
        Build subsampled similarity matrix.

        Args:
            observations: Sequence observations (n_taxa x seq_len)
            p: Sampling probability (0 < p <= 1)
            seed: Random seed for reproducibility
            min_similarity: Minimum similarity threshold (currently unused)

        Returns:
            Subsampled similarity matrix with diagonal = 1.0
        """
        # Get full similarity matrix (cached)
        full_similarity = self.build_full(observations)

        # Use sampler to subsample/recover matrix
        subsampled = self.sampler.sample(full_similarity, p, seed)

        return subsampled
    

    def _get_observations_hash(self, observations: np.ndarray) -> str:
        """
        Generate a fast hash key for the observations to use as cache key.
        
        Uses array metadata + corner values + checksum for speed.
        Much faster than MD5 for large arrays (no full copy needed).
        
        Args:
            observations: Observation matrix
            
        Returns:
            Hash key string
        """
        # Fast hash using shape + dtype + corner values + checksum
        # This avoids creating a full copy with tobytes() for large arrays
        shape_tuple = observations.shape
        dtype_str = observations.dtype.str
        
        # Get corner values (if array is non-empty)
        if observations.size > 0:
            corner_0_0 = int(observations.flat[0]) if observations.size > 0 else 0
            corner_n_n = int(observations.flat[-1]) if observations.size > 0 else 0
            # Compute checksum (mod to prevent overflow)
            checksum = int(observations.sum()) % (2**31)
        else:
            corner_0_0 = 0
            corner_n_n = 0
            checksum = 0
        
        # Create hash tuple
        hash_tuple = (shape_tuple, dtype_str, corner_0_0, corner_n_n, checksum)
        
        # Use Python's built-in hash (fast and sufficient for cache keys)
        return str(hash(hash_tuple))
    
    def clear_cache(self):
        """Clear the similarity matrix cache."""
        self._cache.clear()
        log_info('cache', "Similarity matrix cache cleared")
    
    @property
    def cache_size(self) -> int:
        """Return the number of cached matrices."""
        return len(self._cache)

