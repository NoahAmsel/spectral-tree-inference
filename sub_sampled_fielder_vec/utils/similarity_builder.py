"""Similarity matrix construction and subsampling."""
import hashlib
import numpy as np
import spectraltree

from .logging import log_info


class SimilarityMatrixBuilder:
    """
    Encapsulates similarity matrix construction and subsampling pipeline.
    
    Manages caching and provides clean methods for building full and subsampled matrices.
    """
    
    def __init__(self):
        """Initialize the similarity matrix builder with empty cache."""
        self._cache = {}
    
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
            min_similarity: Minimum similarity threshold
            
        Returns:
            Subsampled similarity matrix
        """
        # Get full similarity matrix (cached)
        full_similarity = self.build_full(observations)
        
        # Subsample entries
        subsampled = self._subsample(full_similarity, p, seed)
        
        # Apply constraints (symmetry, diagonal, clipping, min_similarity)
        subsampled = self._apply_constraints(subsampled, min_similarity)
        
        return subsampled
    
    def _subsample(self, matrix: np.ndarray, p: float, seed: int = None) -> np.ndarray:
        """
        Subsample matrix entries with probability p and scale by 1/p.
        
        Args:
            matrix: Input matrix to subsample
            p: Sampling probability (0 < p <= 1)
            seed: Random seed for reproducibility
            
        Returns:
            Subsampled matrix with entries scaled by 1/p
        """
        # Special case: if p is 1.0 or very close to it, return the original matrix
        if p >= 0.9999:  # Use a small tolerance to handle floating-point precision
            return matrix.copy()
        
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        n_keep = int(p * matrix.size)
        indices = rng.choice(matrix.size, size=n_keep, replace=False)
        sampling_mask = np.zeros(matrix.shape, dtype=bool)
        sampling_mask.flat[indices] = True
        
        # Subsample and scale
        subsampled = np.zeros_like(matrix)
        subsampled[sampling_mask] = matrix[sampling_mask] / p
        
        return subsampled
    
    def _apply_constraints(self, matrix: np.ndarray, min_similarity: float = 0.0) -> np.ndarray:
        """
        Apply constraints to matrix: symmetry, diagonal, clipping, min_similarity.
        
        Args:
            matrix: Input matrix
            min_similarity: Minimum similarity threshold
            
        Returns:
            Constrained matrix
        """
        # Apply minimum similarity threshold
        matrix = np.maximum(matrix, min_similarity)
        
        # Ensure diagonal is 1 (self-similarity)
        np.fill_diagonal(matrix, 1.0)
        
        # Ensure symmetry
        matrix = self._symmetrize(matrix)
        
        return matrix
    
    def _symmetrize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Enforce matrix symmetry.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Symmetric matrix
        """
        return (matrix + matrix.T) / 2
    
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

