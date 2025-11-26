"""Similarity matrix construction and subsampling."""
import hashlib
import numpy as np
import spectraltree

from .logging import log_info, suppress_warnings


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

        # Subsample entries (preserves diagonal = 1.0)
        subsampled = self._subsample(full_similarity, p, seed)

        return subsampled
    
    def _subsample(self, matrix: np.ndarray, p: float, seed: int = None) -> np.ndarray:
        """
        Subsample matrix entries with SYMMETRIC masking and scale by 1/p.

        Key insight: For symmetric similarity matrices:
        1. Diagonal is always 1.0 (self-similarity) - preserved, not subsampled
        2. Off-diagonal edges (i,j) and (j,i) are sampled together for symmetry
        3. Sampled values are scaled by 1/p to remain unbiased estimators

        Strategy selection:
            - p >= 0.9999: Return copy
            - p < 0.05: Sparse mode (row-by-row with binomial sampling)
            - p >= 0.05: Dense mode (vectorized upper triangle sampling)

        Args:
            matrix: Symmetric similarity matrix to subsample
            p: Sampling probability (0 < p <= 1)
            seed: Random seed for reproducibility

        Returns:
            Subsampled symmetric matrix with diagonal = 1.0, off-diagonal scaled by 1/p
        """
        if p >= 0.9999:
            return matrix.copy()

        # Threshold to switch strategies (experimentally determined)
        SPARSE_THRESHOLD = 0.05

        if p < SPARSE_THRESHOLD:
            return self._subsample_sparse(matrix, p, seed)
        else:
            return self._subsample_dense(matrix, p, seed)

    def _subsample_sparse(self, matrix: np.ndarray, p: float, seed: int = None) -> np.ndarray:
        """
        Sparse subsampling: row-by-row with binomial counts.

        Optimized for low p (< 0.05) where most entries are zero.
        Uses binomial sampling to determine row counts, avoiding wasted work.

        Args:
            matrix: Symmetric similarity matrix
            p: Sampling probability (low p expected)
            seed: Random seed

        Returns:
            Subsampled matrix with diagonal preserved
        """
        # Initialize random generator
        if seed is not None:
            if hasattr(np.random, 'default_rng'):
                rng = np.random.default_rng(seed)
            else:
                rng = np.random.RandomState(seed)
        else:
            rng = np.random

        n = matrix.shape[0]
        subsampled = np.zeros_like(matrix)

        # Preserve diagonal (self-similarity = 1.0)
        np.fill_diagonal(subsampled, 1.0)

        # Process upper triangle row-by-row
        for i in range(n):
            row_len = n - 1 - i  # Number of entries in upper triangle for this row

            if row_len <= 0:
                continue

            # How many entries to sample in this row?
            if hasattr(rng, 'binomial'):
                n_keep = rng.binomial(row_len, p)
            else:
                n_keep = np.random.binomial(row_len, p)

            if n_keep == 0:
                continue

            # Which entries to sample?
            if hasattr(rng, 'choice'):
                valid_cols_rel = rng.choice(row_len, size=n_keep, replace=False)
            else:
                valid_cols_rel = np.random.choice(row_len, size=n_keep, replace=False)

            # Convert to absolute column indices
            valid_cols = valid_cols_rel + (i + 1)

            # Extract, scale, and mirror for symmetry
            vals = matrix[i, valid_cols] / p
            subsampled[i, valid_cols] = vals
            subsampled[valid_cols, i] = vals

        return subsampled

    def _subsample_dense(self, matrix: np.ndarray, p: float, seed: int = None) -> np.ndarray:
        """
        Dense subsampling: vectorized upper triangle sampling.

        Optimized for moderate to high p (>= 0.05) where vectorization wins.
        Generates boolean mask for entire upper triangle at once.

        Args:
            matrix: Symmetric similarity matrix
            p: Sampling probability (moderate to high p expected)
            seed: Random seed

        Returns:
            Subsampled matrix with diagonal preserved
        """
        # Initialize random generator
        if seed is not None:
            if hasattr(np.random, 'default_rng'):
                rng = np.random.default_rng(seed)
            else:
                rng = np.random.RandomState(seed)
        else:
            rng = np.random

        n = matrix.shape[0]
        subsampled = np.zeros_like(matrix)

        # Preserve diagonal (self-similarity = 1.0)
        np.fill_diagonal(subsampled, 1.0)

        # Generate random values for entire matrix
        if hasattr(rng, 'random'):
            mask_values = rng.random((n, n))
        else:
            mask_values = rng.random_sample((n, n))

        # Create upper triangle mask (i < j, excluding diagonal)
        upper_triangle_mask = np.triu(np.ones((n, n), dtype=bool), k=1)

        # Combine: sample only in upper triangle
        sampling_mask = upper_triangle_mask & (mask_values < p)

        # Extract indices where mask is True
        rows, cols = np.where(sampling_mask)

        # Scale and assign symmetrically
        vals = matrix[rows, cols] / p
        subsampled[rows, cols] = vals
        subsampled[cols, rows] = vals  # Mirror for symmetry

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

