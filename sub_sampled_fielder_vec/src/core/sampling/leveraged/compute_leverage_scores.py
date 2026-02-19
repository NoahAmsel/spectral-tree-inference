"""Compute leverage scores from rank-r SVD of observed entries."""
import numpy as np
from typing import Tuple
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def compute_leverage_scores(
    X: np.ndarray,
    Omega: np.ndarray,
    r: int,
    apply_regularization: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute row and column leverage scores from rank-r SVD of observed entries.

    Phase 1 of leveraged sampling: Estimate importance of each row/column
    by computing leverage scores from a low-rank approximation.

    HLDT Extension - Regularization Floor:
    --------------------------------------
    When apply_regularization=True, adds a regularization floor τ_floor to prevent
    "zero-score lockout" where taxa missed in Phase 1 get zero sampling probability
    in Phase 2.

    Mathematical Justification:
        Without regularization: If taxon i missed in Phase 1 → μᵢ = 0 → pᵢ = 0 → never sampled
        With regularization: μ'ᵢ = μᵢ + τ_floor where τ_floor = mean(μ) ensures pᵢ > 0

    This is critical for STDR where missing an entire taxon can cause catastrophic
    tree reconstruction failure.

    Args:
        X: Full matrix (n x n) - only entries in Omega are used
        Omega: Boolean mask (n x n) indicating observed entries
        r: Target rank for SVD (typically 2 for Fiedler vector)
        apply_regularization: If True, add regularization floor to leverage scores
                             (default False for backward compatibility with LeveragedSampler)

    Returns:
        Tuple of (row_raw, col_raw, row_reg, col_reg, singular_values, tau_floor):
        - row_raw: (n,) array with raw μᵢ = (n/r) * ||U[i, :]||²
        - col_raw: (n,) array with raw νⱼ = (n/r) * ||V[j, :]||²
        - row_reg: (n,) array with regularized μ'ᵢ = μᵢ + τ_floor
        - col_reg: (n,) array with regularized ν'ⱼ = νⱼ + τ_floor
        - singular_values: (r,) array with top r singular values from Phase 1 SVD
        - tau_floor: Float, regularization floor value (0.0 if not applied)

    Note:
        When apply_regularization=False (default), row_reg = row_raw and col_reg = col_raw
        (no regularization applied, returned for API consistency).
    """
    n = X.shape[0]

    # Compute uniform sampling probability for Inverse Probability Weighting (IPW)
    # This corrects the spectral bias from zero-filling at low sampling rates
    n_upper = n * (n - 1) // 2  # Upper triangle entries
    n_observed = np.sum(Omega) // 2  # Divide by 2 since symmetric
    p_uniform = n_observed / n_upper if n_upper > 0 else 1.0

    # Extract observed entries with Inverse Probability Weighting: P_Omega(X) / p
    # This ensures E[X_observed] = X (unbiased spectral estimator)
    # Without IPW: E[X_observed] = p·X (biased, singular vectors localize on few entries)
    X_observed = np.zeros_like(X)
    if p_uniform > 0:
        X_observed[Omega] = X[Omega] / p_uniform  # IPW scaling

    # Convert to sparse matrix for efficient SVD (at low p, matrix is 99%+ zeros)
    X_sparse = csr_matrix(X_observed)
    
    # Compute rank-r SVD
    # For large matrices, use TruncatedSVD for efficiency
    if n > 1000:
        # Use sklearn's TruncatedSVD (supports sparse matrices efficiently)
        svd_model = TruncatedSVD(n_components=r, random_state=42)

        # CRITICAL FIX: fit_transform() returns U*Σ, NOT orthonormal U
        # Without normalization, leverage scores are scaled by σ² → billion-scale values
        U_sigma = svd_model.fit_transform(X_sparse)  # Shape: (n, r), contains U*Σ
        s = svd_model.singular_values_  # Shape: (r,)

        # Normalize columns by singular values to recover orthonormal U
        # Handle near-zero singular values for numerical stability
        s_safe = s.copy()
        s_safe[s_safe < 1e-12] = 1.0
        U = U_sigma / s_safe[None, :]  # Broadcasting: (n, r) / (1, r) → (n, r)

        # V is already orthonormal (svd_model.components_ returns Vt directly)
        Vt = svd_model.components_
        V = Vt.T
    else:
        # For small matrices, use dense SVD (scipy.linalg.svd doesn't support sparse)
        U, s, Vt = svd(X_observed, full_matrices=False)
        # Truncate to rank r
        U = U[:, :r]
        s = s[:r]
        Vt = Vt[:r, :]
        V = Vt.T
    
    # Compute row leverage scores: μ_i = (n/r) * ||U[i, :]||²
    # U is (n x r), so we compute row-wise L2 norms
    row_norms_sq = np.sum(U ** 2, axis=1)  # (n,)
    row_leverage = (n / r) * row_norms_sq
    
    # Compute column leverage scores: ν_j = (n/r) * ||V[j, :]||²
    # V is (n x r), so we compute row-wise L2 norms
    col_norms_sq = np.sum(V ** 2, axis=1)  # (n,)
    col_leverage = (n / r) * col_norms_sq

    # Store raw leverage scores
    row_leverage_raw = row_leverage
    col_leverage_raw = col_leverage

    # Apply regularization floor if requested
    if apply_regularization:
        # Compute regularization floor as mean of estimated leverage scores
        # Theory: leverage scores sum to n, so mean should be 1.0 in perfect estimation
        # In practice, use empirical mean to account for estimation error from Phase 1
        tau_floor = np.mean(row_leverage_raw)

        # Edge case: If mean is very small (Phase 1 severely underpowered),
        # use fallback of 1.0 (theoretical expected value)
        if tau_floor < 1e-12:
            tau_floor = 1.0

        # Add floor to all leverage scores
        # This ensures all taxa have non-zero sampling probability in Phase 2
        row_leverage_reg = row_leverage_raw + tau_floor
        col_leverage_reg = col_leverage_raw + tau_floor
    else:
        # No regularization: return raw scores as "regularized" for API consistency
        tau_floor = 0.0
        row_leverage_reg = row_leverage_raw
        col_leverage_reg = col_leverage_raw

    # Return both raw and regularized scores for diagnostic purposes
    # Also return singular values for Phase 1 quality tracking
    return row_leverage_raw, col_leverage_raw, row_leverage_reg, col_leverage_reg, s[:r], tau_floor
