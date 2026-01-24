"""Leveraged matrix completion sampler with two-phase sampling and IALM recovery."""
import numpy as np

from ..base import BaseSampler
from .compute_leverage_scores import compute_leverage_scores
from .compute_sampling_probabilities import compute_sampling_probabilities
from .ialm_solve import ialm_solve
from .get_last_result import get_last_result
from .uniform_sampling import uniform_sample_upper_triangle
from .nonuniform_sampling import nonuniform_sample_upper_triangle
from .rng_utils import get_rng

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
        ialm_tol: float = 1e-6,
        ialm_bypass_threshold: float = 0.1,
        force_leveraged: bool = False
    ):
        """
        Initialize leveraged sampler.

        Args:
            theta: Phase 1 budget ratio (fraction of samples for uniform phase)
            target_rank: Rank r for SVD in leverage score computation
            ialm_max_iter: Maximum iterations for IALM solver
            ialm_tol: Convergence tolerance for IALM
            ialm_bypass_threshold: Skip IALM when p >= this threshold (default: 0.1)
            force_leveraged: If True, always use leveraged sampling even when Phase 1
                           budget is theoretically insufficient. Useful for experimentation.
        """
        self.theta = theta
        self.target_rank = target_rank
        self.ialm_max_iter = ialm_max_iter
        self.ialm_tol = ialm_tol
        self.ialm_bypass_threshold = ialm_bypass_threshold
        self.force_leveraged = force_leveraged
        # Track which p-values have been logged (to log once per p, not per sample)
        self._logged_p_values = set()
        # Store per-sample diagnostic metrics
        self.last_sample_metrics = None
    
    def reset_logging(self):
        """Reset logged p-values to allow re-logging in new experiments."""
        self._logged_p_values = set()
    
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
        rng = get_rng(seed)

        # Compute theoretical minimum samples for Phase 1 (from matrix completion theory)
        # Paper requirement: m₁ ≥ C·n·r·log(n) for reliable leverage score estimation
        # For symmetric rank-r matrices: 4·n·r·log(n) is conservative
        theoretical_min_phase1 = int(4 * n * self.target_rank * np.log(n)) if n > 1 else 10

        # Phase 1: Uniform sampling to estimate leverage scores
        phase1_budget_naive = int(self.theta * total_budget)
        
        if self.force_leveraged:
            # Force leveraged sampling: use naive budget even if below theoretical minimum
            phase1_budget = phase1_budget_naive
            # Ensure at least 1 sample for Phase 1
            if phase1_budget < 1:
                phase1_budget = 1
        else:
            # Use max(theta*total, min_required) to ensure meaningful leverage scores
            phase1_budget = max(phase1_budget_naive, theoretical_min_phase1)

        # Check if total budget is sufficient for leveraged sampling
        if phase1_budget >= total_budget:
            if self.force_leveraged:
                # Even when forced, we need some budget for Phase 2
                # Use 90% for Phase 1, 10% for Phase 2
                phase1_budget = max(1, int(0.9 * total_budget))
                log_info('leveraged',
                    f"p={p:.4f}: Force leveraged mode - using {phase1_budget:,}/{total_budget:,} for Phase 1 "
                    f"(theoretical min: {theoretical_min_phase1:,})"
                )
            else:
                # Phase 1 would consume entire budget - fall back to uniform sampling
                # This happens when p is too small for leveraged sampling to be beneficial
                log_info('leveraged',
                    f"p={p:.4f}: Phase 1 requires {phase1_budget:,} samples but total budget is {total_budget:,}. "
                    f"Falling back to uniform sampling (leveraged sampling requires p ≥ {theoretical_min_phase1/n_upper:.6f})"
                )
                # Use uniform sampler logic instead (uniform_sample_upper_triangle imported at module level)
                Omega = uniform_sample_upper_triangle(n, total_budget, rng)
                L = np.zeros_like(matrix)
                L[Omega] = matrix[Omega]
                L = (L + L.T) / 2
                np.fill_diagonal(L, 1.0)
                return L

        Omega1 = uniform_sample_upper_triangle(n, phase1_budget, rng)
        phase1_actual = np.sum(Omega1) // 2  # Divide by 2 since symmetric mask counts both (i,j) and (j,i)
        
        # Compute leverage scores from Phase 1 observations (also returns singular values)
        row_leverage, col_leverage, phase1_singular_values = compute_leverage_scores(
            matrix, Omega1, self.target_rank
        )
        
        # Phase 2: Non-uniform sampling based on leverage scores
        phase2_budget = total_budget - phase1_budget
        phase2_actual = 0
        if phase2_budget > 0:
            # Compute sampling probabilities
            p_matrix = compute_sampling_probabilities(
                row_leverage, col_leverage, self.target_rank, n
            )
            
            # Sample from upper triangle according to probabilities
            # Restrict to Omega_1^c (complement of Phase 1 observations)
            Omega2 = nonuniform_sample_upper_triangle(
                n, phase2_budget, p_matrix, rng, Omega_1=Omega1
            )
            
            # Omega = Omega_1 ∪ Omega_2 (disjoint by construction: Omega_2 ⊆ Omega_1^c)
            Omega = Omega1 | Omega2
            phase2_actual = np.sum(Omega2) // 2  # All entries in Omega_2 are new
        else:
            Omega = Omega1

        # Compute Phase 1 sufficiency ratio (already computed theoretical_min above)
        phase1_sufficiency = phase1_actual / theoretical_min_phase1 if theoretical_min_phase1 > 0 else 0.0

        # Leverage score diagnostics (for symmetric matrices, μ_i = ν_i)
        leverage_scores = row_leverage  # Use row scores (col scores identical for symmetric)
        leverage_max = float(np.max(leverage_scores))
        leverage_std = float(np.std(leverage_scores))
        leverage_sum = float(np.sum(leverage_scores))  # Should equal n
        leverage_symmetry_error = float(np.linalg.norm(row_leverage - col_leverage))  # Should be ~0

        # Store diagnostic metrics (before bypass or IALM)
        self.last_sample_metrics = {
            # Sampling budget
            'phase1_budget': phase1_budget,
            'phase1_actual': phase1_actual,
            'phase2_budget': phase2_budget,
            'phase2_actual': phase2_actual,
            'theoretical_min_phase1': theoretical_min_phase1,
            'phase1_sufficiency': phase1_sufficiency,

            # Phase 1 SVD (first r singular values)
            'phase1_singular_values': phase1_singular_values.tolist(),

            # Leverage score stats
            'leverage_max': leverage_max,
            'leverage_std': leverage_std,
            'leverage_sum': leverage_sum,
            'leverage_symmetry_error': leverage_symmetry_error,

            # IALM execution (will be updated after IALM if not bypassed)
            'ialm_bypassed': False,
            'ialm_iterations': 0,
            'ialm_converged': False,
        }

        # Check if we should bypass IALM (data is dense enough)
        if p >= self.ialm_bypass_threshold:
            # Dense sampling - skip matrix completion, use sparse matrix directly
            L = np.zeros_like(matrix)
            L[Omega] = matrix[Omega]
            L = (L + L.T) / 2  # Ensure symmetry
            np.fill_diagonal(L, 1.0)

            # Log bypass decision
            total_sampled = np.sum(Omega) // 2
            log_info('leveraged',
                f"p={p:.4f}: Bypassing IALM (p ≥ {self.ialm_bypass_threshold:.2f}), "
                f"using {total_sampled:,} samples directly"
            )

            # Update metrics for bypass case
            self.last_sample_metrics['ialm_bypassed'] = True
            self.last_sample_metrics['ialm_iterations'] = 0
            self.last_sample_metrics['ialm_converged'] = True  # Trivially converged

            return L

        # Phase 3: IALM recovery
        # Compute adaptive lambda parameter from paper's Theorem 3.2
        # Theory: λ = 1/(4√mn) × √(mn/|Ω|) = 1/(4√|Ω|)
        # For symmetric n×n with |Ω| = p·n²/2:
        #   λ = 1/(4√(p·n²/2)) = 1/(2n√(2p))
        #
        # Simplified: λ = 1 / (n · √(2p))
        # This ensures λ→0 as p→1 (less denoising for dense sampling)
        # and λ increases as p→0 (more aggressive denoising for sparse sampling)
        total_samples = np.sum(Omega) // 2  # Actual sampled entries (symmetric)
        if total_samples > 0:
            lambda_param = 1.0 / (n * np.sqrt(2 * p)) if p > 0 else 0.1
        else:
            lambda_param = 0.1  # Fallback for edge case
        
        # Run IALM solver
        L, S = ialm_solve(
            matrix,
            Omega,
            lambda_param,
            max_iter=self.ialm_max_iter,
            tol=self.ialm_tol
        )
        
        # Get diagnostic info from the solver
        result_info = get_last_result()

        # Update IALM metrics
        self.last_sample_metrics['ialm_bypassed'] = False
        self.last_sample_metrics['ialm_iterations'] = result_info.iterations if result_info else self.ialm_max_iter
        self.last_sample_metrics['ialm_converged'] = result_info.converged if result_info else False

        # Ensure diagonal is 1.0 (self-similarity)
        np.fill_diagonal(L, 1.0)
        
        # Ensure symmetry (IALM might introduce small asymmetry due to numerical errors)
        L = (L + L.T) / 2
        
        # Compute rank estimate
        rank_estimate = np.linalg.matrix_rank(L, tol=1e-6)
        
        # Log once per p-value (not per bootstrap replicate)
        p_key = round(p, 6)  # Round to avoid floating point comparison issues
        if p_key not in self._logged_p_values:
            self._logged_p_values.add(p_key)
            
            # Compute total unique sampled entries
            total_sampled = np.sum(Omega) // 2  # Divide by 2 since symmetric
            
            # Build informative log message with phase breakdown
            status = "✓" if result_info and result_info.converged else "⚠ max_iter"
            iters = result_info.iterations if result_info else self.ialm_max_iter
            
            log_msg = (
                f"p={p:.4f}: {total_sampled:,} samples "
                f"(Phase1: {phase1_actual:,} uniform, Phase2: {phase2_actual:,} leveraged) → "
                f"IALM {status} ({iters} iters), rank≈{rank_estimate}"
            )
            
            if result_info and result_info.had_numerical_issues:
                log_msg += " [numerical issues]"
            
            log_info('leveraged', log_msg, force=True)
        
        return L
