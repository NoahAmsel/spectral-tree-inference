"""Uniform random subsampling implementation."""
import numpy as np

from ..base import BaseSampler


class UniformSampler(BaseSampler):
    """
    Uniform random subsampling of matrix entries.
    
    Preserves diagonal (self-similarity = 1.0) and maintains symmetry.
    Uses optimized sparse/dense strategies based on sampling probability.
    """
    
    def sample(self, matrix: np.ndarray, p: float, seed: int = None, **kwargs) -> np.ndarray:
        """
        Subsample matrix entries uniformly with probability p.
        
        Args:
            matrix: Symmetric similarity matrix to subsample
            p: Sampling probability (0 < p <= 1)
            seed: Random seed for reproducibility
            **kwargs: Ignored (for API compatibility)
            
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

