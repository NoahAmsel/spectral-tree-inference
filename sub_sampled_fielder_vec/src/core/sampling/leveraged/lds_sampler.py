"""LDS (Leveraged Debiased Sampler) for spectral preservation.

This sampler implements the LDS approach for phylogenetic tree reconstruction via
Spectral Top-Down Recovery (STDR). Unlike the IALM-based LeveragedSampler which
performs iterative matrix completion, LDSSampler uses a single-shot debiased
estimator that directly preserves spectral structure.

Key Differences from LeveragedSampler (IALM):
--------------------------------------------
1. **No Iterative Optimization**: Single-shot debiased estimator vs 100+ IALM iterations
2. **Regularization Floor**: Prevents zero-probability lockout for missed taxa
3. **Probability Capping**: Ensures unbiased estimator when p_ij > 1.0
4. **Sparse Output**: Returns scipy.sparse.csr_matrix for efficiency
5. **Performance**: O(n²) vs O(n³ × iterations) → 10-100x speedup

Theoretical Foundation:
----------------------
- **LDS Theorem**: With m = O(nr log n) samples using leverage scores, the debiased
  estimator X̂ satisfies ‖X̂ - X‖ = O(√(n log n / m)) with high probability
- **Davis-Kahan Bound**: ‖v̂ - v‖ ≤ ‖X̂ - X‖ / gap where v is Fiedler vector, gap is spectral gap
- **Conclusion**: For STDR, we don't need exact matrix entries, just eigenvector stability

When to Use LDS vs IALM:
------------------------
- **LDS**: Large trees (n > 1000), need speed, spectral gap is large
- **IALM**: Small trees, need exact matrix reconstruction, very noisy data

Reference:
---------
Based on Huang, Liu, Du, Tao - "Leveraged Matrix Completion With Noise" (2018)
Davis, Kahan - "The Rotation of Eigenvectors by a Perturbation" (1970)
"""
import numpy as np
import time
from scipy.sparse import csr_matrix

from ..base import BaseSampler
from .compute_leverage_scores import compute_leverage_scores
from .compute_sampling_probabilities import compute_sampling_probabilities
from .compute_debiased_estimator import compute_debiased_estimator
from .uniform_sampling import uniform_sample_upper_triangle
from .nonuniform_sampling import nonuniform_sample_upper_triangle
from .rng_utils import get_rng

# Import logging utilities
import sys
from pathlib import Path
_src_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
from utils.logging import log_info


class InsufficientBudgetError(Exception):
    """Raised when LDS sampling budget is insufficient and fallback is disabled."""
    pass


class LDSSampler(BaseSampler):
    """
    LDS (Leveraged Debiased Sampler) for spectral preservation in STDR.

    Implements a three-phase sampling pipeline:
    1. **Phase 1 (Uniform)**: Sample θ×budget entries uniformly to estimate leverage scores
    2. **Phase 2 (Leveraged)**: Sample (1-θ)×budget entries based on leverage importance
    3. **Phase 3 (Debiasing)**: Construct debiased estimator X̂ = X/p (no IALM iterations)

    The debiased estimator satisfies E[X̂] = X, which by Davis-Kahan theorem ensures
    eigenvectors of L̂ = D̂ - X̂ are close to eigenvectors of L = D - X.

    Key Features:
    - Regularization floor prevents zero-probability lockout
    - Probability capping ensures unbiased estimator
    - Sparse matrix output for memory efficiency
    - Phase 1 sufficiency logging (warns if below theoretical minimum)
    - 10-100x faster than IALM-based LeveragedSampler
    """

    def __init__(
        self,
        theta: float = 0.3,
        target_rank: int = 2,
        tau_floor_multiplier: float = 1.0,
        force_lds: bool = False,
        allow_uniform_fallback: bool = True,
        prob_formula: str = "additive"
    ):
        """
        Initialize LDS sampler.

        Args:
            theta: Phase 1 budget ratio (fraction of samples for uniform phase)
                  Default 0.3 = 30% for Phase 1, 70% for Phase 2
            target_rank: Rank r for SVD in leverage score computation
                        Default 2 (sufficient for Fiedler vector)
            tau_floor_multiplier: Multiplier for regularization floor
                                 Default 1.0 (τ_floor = multiplier × mean(μ))
            force_lds: Retained for API compatibility (no longer changes behavior)
            allow_uniform_fallback: Retained for API compatibility (no longer changes behavior)
            prob_formula: Formula for combining row/col leverage into p_ij.
                         'additive': μ_i+μ_j (HLDT paper default)
                         'multiplicative': μ_i×μ_j (concentrates on high-leverage pairs)
                         'max': max(μ_i,μ_j)
        """
        self.theta = theta
        self.target_rank = target_rank
        self.tau_floor_multiplier = tau_floor_multiplier
        self.force_lds = force_lds
        self.allow_uniform_fallback = allow_uniform_fallback
        self.prob_formula = prob_formula

        # Track which p-values have been logged (to log once per p, not per sample)
        self._logged_p_values = set()

        # Store per-sample diagnostic metrics
        self.last_sample_metrics = None

    def reset_logging(self):
        """Reset logged p-values to allow re-logging in new experiments."""
        self._logged_p_values = set()

    def sample(self, matrix: np.ndarray, p: float, seed: int = None, **kwargs) -> np.ndarray:
        """
        Apply LDS sampling and debiasing pipeline.

        IMPORTANT: In LDS sampling, `p` represents a **total budget fraction**,
        not a per-entry probability (unlike uniform sampling).

        - Uniform: `p` = probability per entry (stochastic, expected samples = p × n_upper)
        - LDS: `p` = budget fraction (deterministic, exact samples = p × n_upper)

        The total budget is split:
        - Phase 1: θ × total_budget (uniform sampling for leverage estimation)
        - Phase 2: (1-θ) × total_budget (leveraged sampling based on importance)

        Args:
            matrix: Full symmetric similarity matrix (n x n)
            p: Total sampling budget as fraction of available entries (0 < p <= 1)
               For n×n matrix: total_samples = p × n(n-1)/2 (upper triangle entries)
            seed: Random seed for reproducibility
            **kwargs: Ignored (for API compatibility with BaseSampler)

        Returns:
            Debiased estimator X̂ as sparse matrix (scipy.sparse.csr_matrix or dense)
            - Symmetric: X̂ = X̂ᵀ
            - Diagonal = 1.0: X̂_ii = 1.0 for all i
            - Sparse: nnz(X̂) = |Ω| + n (sampled entries + diagonal)
            - Unbiased: E[X̂] = X

        Raises:
            InsufficientBudgetError: If p is too small for LDS and allow_uniform_fallback=False
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

        # Compute theoretical minimum samples for Phase 1 (DIAGNOSTIC ONLY — never overrides theta)
        # Paper requirement: m₁ ≥ C·n·r·log²(n) for reliable leverage score estimation
        theoretical_min_phase1 = int(4 * n * self.target_rank * np.log(n) ** 2) if n > 1 else 10

        # =============================================================================
        # PHASE 1: Uniform Sampling to Estimate Leverage Scores
        # =============================================================================

        # Pure-theta split: theta controls Phase 1/Phase 2 budget directly.
        # theoretical_min_phase1 is logged for diagnostics but never overrides theta.
        phase1_budget = max(1, int(self.theta * total_budget))

        # Sample uniformly from upper triangle
        Omega1 = uniform_sample_upper_triangle(n, phase1_budget, rng)
        # Exclude diagonal (self-similarity doesn't need sampling)
        np.fill_diagonal(Omega1, False)
        phase1_actual = np.sum(Omega1) // 2  # Divide by 2 since symmetric mask counts both (i,j) and (j,i)

        # Compute leverage scores from Phase 1 observations WITH regularization
        # Returns: (row_raw, col_raw, row_reg, col_reg, singular_values, tau_floor)
        row_leverage_raw, col_leverage_raw, row_leverage_reg, col_leverage_reg, phase1_singular_values, tau_floor = compute_leverage_scores(
            matrix, Omega1, self.target_rank, apply_regularization=True, seed=seed
        )

        # Apply tau_floor_multiplier scaling
        if self.tau_floor_multiplier != 1.0:
            tau_floor_scaled = tau_floor * self.tau_floor_multiplier
            row_leverage_reg = row_leverage_raw + tau_floor_scaled
            col_leverage_reg = col_leverage_raw + tau_floor_scaled
        else:
            tau_floor_scaled = tau_floor

        # =============================================================================
        # Phase 1 Sufficiency Check & Logging
        # =============================================================================

        phase1_sufficiency = phase1_actual / theoretical_min_phase1 if theoretical_min_phase1 > 0 else 0.0

        # Log warning if Phase 1 budget is below theoretical minimum (but don't block)
        if phase1_actual < theoretical_min_phase1:
            if p not in self._logged_p_values:
                log_info('bootstrap',
                    f"  ⚠ LDS Phase 1 Quality: p={p:.4f} has only {phase1_actual:,}/{theoretical_min_phase1:,} "
                    f"samples ({phase1_sufficiency:.1%} of theoretical minimum)",
                    force=True
                )
                log_info('bootstrap',
                    f"    → Leverage estimates will be noisy, but estimator remains unbiased",
                    force=True
                )
                self._logged_p_values.add(p)  # Mark as logged to avoid spam across bootstraps

        # =============================================================================
        # PHASE 2: Non-Uniform Sampling Based on Leverage Scores
        # =============================================================================

        phase2_budget = total_budget - phase1_budget
        phase2_actual = 0
        p_matrix = None
        phase2_sampled_indices = []

        if phase2_budget > 0:
            # Compute sampling probabilities using REGULARIZED leverage scores
            p_matrix = compute_sampling_probabilities(
                row_leverage_reg, col_leverage_reg, self.target_rank, n,
                prob_formula=self.prob_formula
            )

            # **CRITICAL**: Cap probabilities at 1.0
            # This ensures E[X̂] = X (unbiased property) even if theoretical formula gives p > 1
            p_matrix = np.minimum(p_matrix, 1.0)

            # Sample from upper triangle according to probabilities
            # Restrict to Omega_1^c (complement of Phase 1 observations)
            Omega2 = nonuniform_sample_upper_triangle(
                n, phase2_budget, p_matrix, rng, Omega_1=Omega1
            )

            # Exclude diagonal
            np.fill_diagonal(Omega2, False)

            # Extract Phase 2 sampled indices (i,j) where i < j from upper triangle
            phase2_rows, phase2_cols = np.where(np.triu(Omega2, k=1))
            phase2_sampled_indices = list(zip(phase2_rows.tolist(), phase2_cols.tolist()))

            # Omega = Omega_1 ∪ Omega_2 (disjoint by construction: Omega_2 ⊆ Omega_1^c)
            Omega = Omega1 | Omega2
            phase2_actual = np.sum(Omega2) // 2  # All entries in Omega_2 are new

            # **CRITICAL**: Compute effective inclusion probability for debiasing
            # π_{ij} = p_0 + (1 - p_0) · p2_{ij}
            # p_matrix is a normalized PMF (sums to 1).
            # Actual Phase 2 inclusion probability = weight × budget / restricted_sum
            p_0 = phase1_actual / n_upper if n_upper > 0 else 0.0
            p_restricted = np.triu(p_matrix, k=1).copy()
            p_restricted[np.triu(Omega1, k=1)] = 0.0  # exclude Phase 1 entries
            p_restricted_sum = p_restricted.sum()
            if p_restricted_sum > 0:
                p2 = np.minimum((p_restricted / p_restricted_sum) * phase2_budget, 1.0)
            else:
                p2 = np.zeros_like(p_matrix)
            p2 = p2 + p2.T  # symmetrize
            pi_matrix = np.minimum(p_0 + (1.0 - p_0) * p2, 1.0)
        else:
            Omega = Omega1
            # Phase 1 only: effective probability is just p_0 (no Phase 2)
            p_0 = phase1_actual / n_upper if n_upper > 0 else 1.0
            pi_matrix = np.ones((n, n)) * p_0

        # Final check: ensure diagonal is excluded
        np.fill_diagonal(Omega, False)

        # =============================================================================
        # PHASE 3: Debiased Estimator (NO IALM ITERATIONS!)
        # =============================================================================

        # This is the key difference from LeveragedSampler:
        # - LeveragedSampler: 100+ IALM iterations (O(n³ × iters))
        # - LDSSampler: Single-shot debiasing (O(n²))

        start_time = time.time()
        X_hat_sparse = compute_debiased_estimator(matrix, Omega, pi_matrix)
        debiasing_time = time.time() - start_time

        # Compute matrix sparsity for diagnostics
        matrix_sparsity = X_hat_sparse.nnz / (n * n)

        # =============================================================================
        # Diagnostic Metrics
        # =============================================================================

        # Leverage score diagnostics (for symmetric matrices, μ_i = ν_i)
        leverage_scores_raw = row_leverage_raw
        leverage_scores_reg = row_leverage_reg
        leverage_max_raw = float(np.max(leverage_scores_raw))
        leverage_std_raw = float(np.std(leverage_scores_raw))
        leverage_sum_raw = float(np.sum(leverage_scores_raw))  # Should equal n

        # Compute spectral gap (ratio s[1]/s[2]) for Davis-Kahan bound diagnostics
        sv = phase1_singular_values
        spectral_gap = float(sv[1] / sv[2]) if len(sv) >= 3 and sv[2] > 1e-12 else float('inf')

        # Store diagnostic metrics
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

            # Leverage score stats (raw and regularized)
            'leverage_max': leverage_max_raw,
            'leverage_std': leverage_std_raw,
            'leverage_sum': leverage_sum_raw,
            'leverage_scores_raw': leverage_scores_raw,
            'leverage_scores_regularized': leverage_scores_reg,
            'tau_floor': tau_floor_scaled,

            # Phase 2 sampling details
            'phase2_sampling_probs': p_matrix,  # Leverage-based probabilities p_{ij}
            'phase2_sampled_indices': phase2_sampled_indices,

            # Debiasing probabilities (effective inclusion probabilities)
            'phase1_uniform_prob': p_0,  # Uniform probability from Phase 1
            'effective_debiasing_probs': pi_matrix,  # π_{ij} = p_0 + (1-p_0)·p2_{ij}

            # Spectral gap (s[1]/s[2]) — relevant for Davis-Kahan bound
            # inf when target_rank < 3 (set target_rank=3 for a meaningful value)
            'spectral_gap': spectral_gap,

            # Debiasing performance
            'debiasing_time': debiasing_time,
            'matrix_sparsity': matrix_sparsity,

            # No IALM metrics (not applicable to LDS)
            'ialm_bypassed': True,
            'ialm_iterations': 0,
            'ialm_converged': True,  # Trivially converged (no iterations)
            'fallback_to_uniform': False,

            # Probability formula used for Phase 2 sampling
            'prob_formula': self.prob_formula,
        }

        # Log successful LDS sampling setup (only log once per p-value)
        if p not in self._logged_p_values:
            satisfied_pct = phase1_actual / theoretical_min_phase1 if theoretical_min_phase1 > 0 else float('inf')
            log_info('bootstrap',
                f"  p={p:.4f}: Phase1={phase1_actual:,}/{phase1_budget:,}, Phase2={phase2_actual:,}/{phase2_budget:,}, "
                f"theoretical_min={theoretical_min_phase1:,} ({satisfied_pct:.1%} satisfied)"
            )
            log_info('bootstrap',
                f"    Leverage scores: max={leverage_max_raw:.3f}, std={leverage_std_raw:.3f}, "
                f"sum={leverage_sum_raw:.1f}, τ_floor={tau_floor_scaled:.3f}, formula={self.prob_formula}"
            )
            log_info('bootstrap',
                f"    Debiased estimator: {X_hat_sparse.nnz:,} non-zeros ({matrix_sparsity:.1%} dense), "
                f"computed in {debiasing_time:.3f}s"
            )
            self._logged_p_values.add(p)

        # Convert sparse to dense if matrix is small or dense enough
        # This provides compatibility with FiedlerVectorComputer which handles both
        if n < 1000 or matrix_sparsity > 0.5:
            return X_hat_sparse.toarray()
        else:
            return X_hat_sparse
