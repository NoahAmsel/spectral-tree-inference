"""Leveraged matrix completion sampler with two-phase sampling and IALM recovery."""
import numpy as np

from ..base import BaseSampler
from .leverage_scores import compute_leverage_scores, compute_sampling_probabilities
from .ialm_solver import ialm_solve

# Import from utils - from src/core/sampling/leveraged/ go up 3 levels to src/, then utils.logging
# This matches the pattern: src/core/utils.py uses ..utils.logging (up 1 level from core to src)
# Here we go up 3 levels: .. -> core/sampling, ... -> core, then need to go to src
# Using absolute import style that works with the package structure
import sys
from pathlib import Path
_src_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
from utils.logging import log_info


class LeveragedSampler(BaseSampler):
    """
    Leveraged matrix completion sampler.
    
    Implements Algorithm 1 from the paper:
    1. Phase 1: Uniform sampling to estimate leverage scores
    2. Phase 2: Non-uniform sampling based on leverage scores
    3. Phase 3: IALM recovery to get clean low-rank matrix
    """
    
    def __init__(
        self,
        theta: float = 0.3,
        target_rank: int = 2,
        ialm_max_iter: int = 100,
        ialm_tol: float = 1e-6
    ):
        """
        Initialize leveraged sampler.
        
        Args:
            theta: Phase 1 budget ratio (fraction of samples for uniform phase)
            target_rank: Rank r for SVD in leverage score computation
            ialm_max_iter: Maximum iterations for IALM solver
            ialm_tol: Convergence tolerance for IALM
        """
        self.theta = theta
        self.target_rank = target_rank
        self.ialm_max_iter = ialm_max_iter
        self.ialm_tol = ialm_tol
    
    def sample(self, matrix: np.ndarray, p: float, seed: int = None, **kwargs) -> np.ndarray:
        """
        Apply leveraged sampling and recovery pipeline.
        
        IMPORTANT: In leveraged sampling, `p` represents a **total budget fraction**,
        not a per-entry probability (unlike uniform sampling).
        
        - Uniform: `p` = probability per entry (stochastic, expected samples = p × n_upper)
        - Leveraged: `p` = budget fraction (deterministic, exact samples = p × n_upper)
        
        The total budget is split:
        - Phase 1: θ × total_budget (uniform sampling for leverage estimation)
        - Phase 2: (1-θ) × total_budget (leveraged sampling based on importance)
        
        Args:
            matrix: Full symmetric similarity matrix (n x n)
            p: Total sampling budget as fraction of available entries (0 < p <= 1)
               For n×n matrix: total_samples = p × n(n-1)/2 (upper triangle entries)
            seed: Random seed for reproducibility
            **kwargs: Ignored (for API compatibility)
            
        Returns:
            Recovered low-rank matrix L (n x n) with diagonal = 1.0
        """
        n = matrix.shape[0]
        
        # Total number of entries to sample (excluding diagonal)
        n_upper = n * (n - 1) // 2  # Upper triangle entries
        total_budget = int(p * n_upper)
        
        if total_budget == 0:
            # Return identity matrix if no budget
            result = np.eye(n, dtype=matrix.dtype)
            return result
        
        # Initialize random generator
        if seed is not None:
            if hasattr(np.random, 'default_rng'):
                rng = np.random.default_rng(seed)
            else:
                rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        # Phase 1: Uniform sampling to estimate leverage scores
        phase1_budget = int(self.theta * total_budget)
        Omega1 = self._uniform_sample_upper_triangle(n, phase1_budget, rng)
        
        # Compute leverage scores from Phase 1 observations
        log_info('leveraged', f"Phase 1: Computing leverage scores from {phase1_budget} uniform samples...")
        row_leverage, col_leverage = compute_leverage_scores(
            matrix, Omega1, self.target_rank
        )
        
        # Phase 2: Non-uniform sampling based on leverage scores
        phase2_budget = total_budget - phase1_budget
        if phase2_budget > 0:
            log_info('leveraged', f"Phase 2: Sampling {phase2_budget} entries based on leverage scores...")
            # Compute sampling probabilities
            p_matrix = compute_sampling_probabilities(
                row_leverage, col_leverage, self.target_rank, n
            )
            
            # Sample from upper triangle according to probabilities
            Omega2 = self._nonuniform_sample_upper_triangle(
                n, phase2_budget, p_matrix, rng
            )
            
            # Combine observation sets
            Omega = Omega1 | Omega2
        else:
            Omega = Omega1
        
        # Phase 3: IALM recovery
        log_info('leveraged', f"Phase 3: IALM recovery from {np.sum(Omega)} observed entries...")
        
        # Compute lambda parameter: λ = 1 / (24 * sqrt(n * log(n)))
        lambda_param = 1.0 / (24 * np.sqrt(n * np.log(n))) if n > 1 else 0.1
        
        # Run IALM solver
        L, S = ialm_solve(
            matrix,
            Omega,
            lambda_param,
            max_iter=self.ialm_max_iter,
            tol=self.ialm_tol
        )
        
        # Ensure diagonal is 1.0 (self-similarity)
        np.fill_diagonal(L, 1.0)
        
        # Ensure symmetry (IALM might introduce small asymmetry due to numerical errors)
        L = (L + L.T) / 2
        
        log_info('leveraged', f"Recovery complete. Rank estimate: {np.linalg.matrix_rank(L, tol=1e-6)}")
        
        return L
    
    def _uniform_sample_upper_triangle(
        self,
        n: int,
        budget: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Uniformly sample entries from upper triangle.
        
        Args:
            n: Matrix dimension
            budget: Number of entries to sample
            rng: Random number generator
            
        Returns:
            Boolean mask (n x n) with True for sampled entries
        """
        # Get all upper triangle indices (excluding diagonal)
        upper_indices = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_indices.append((i, j))
        
        # Sample without replacement
        if budget >= len(upper_indices):
            # Sample all entries
            sampled_indices = upper_indices
        else:
            if hasattr(rng, 'choice'):
                sampled_idx = rng.choice(len(upper_indices), size=budget, replace=False)
            else:
                sampled_idx = np.random.choice(len(upper_indices), size=budget, replace=False)
            sampled_indices = [upper_indices[i] for i in sampled_idx]
        
        # Create boolean mask
        Omega = np.zeros((n, n), dtype=bool)
        for i, j in sampled_indices:
            Omega[i, j] = True
            Omega[j, i] = True  # Symmetric
        
        return Omega
    
    def _nonuniform_sample_upper_triangle(
        self,
        n: int,
        budget: int,
        p_matrix: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample entries from upper triangle according to probability distribution.
        
        Args:
            n: Matrix dimension
            budget: Number of entries to sample
            p_matrix: Probability matrix (n x n) - only upper triangle is used
            rng: Random number generator
            
        Returns:
            Boolean mask (n x n) with True for sampled entries
        """
        # Extract upper triangle probabilities
        upper_triangle_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        p_upper = p_matrix * upper_triangle_mask
        
        # Flatten to 1D for sampling
        p_flat = p_upper[upper_triangle_mask]
        indices_flat = np.where(upper_triangle_mask)
        
        # Normalize probabilities
        p_flat = p_flat / np.sum(p_flat) if np.sum(p_flat) > 0 else p_flat
        
        # Sample according to probabilities (with replacement if budget > available)
        n_upper = len(p_flat)
        if budget >= n_upper:
            # Sample all entries
            sampled_idx = np.arange(n_upper)
        else:
            if hasattr(rng, 'choice'):
                sampled_idx = rng.choice(n_upper, size=budget, replace=False, p=p_flat)
            else:
                sampled_idx = np.random.choice(n_upper, size=budget, replace=False, p=p_flat)
        
        # Create boolean mask
        Omega = np.zeros((n, n), dtype=bool)
        for idx in sampled_idx:
            i, j = indices_flat[0][idx], indices_flat[1][idx]
            Omega[i, j] = True
            Omega[j, i] = True  # Symmetric
        
        return Omega

