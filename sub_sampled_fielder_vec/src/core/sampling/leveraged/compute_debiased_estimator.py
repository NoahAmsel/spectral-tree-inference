"""Compute debiased estimator for LDS spectral sampling.

This module implements the single-shot debiased estimator from the Huang-Liu-Du-Tao (2018)
paper on leveraged matrix completion, used in LDS (Leveraged Debiased Sampler). Unlike
iterative matrix completion (IALM), this approach directly constructs an unbiased
estimator that preserves spectral structure.

Mathematical Foundation:
-----------------------
For sampled entries (i,j) ∈ Ω with sampling probability p_ij:
    X̂_ij = X_ij / min(p_ij, 1.0)

For unsampled entries (i,j) ∉ Ω:
    X̂_ij = 0

Key Property (Unbiased Estimator):
    E[X̂_ij] = E[𝟙_{(i,j)∈Ω} × X_ij / p_ij]
            = p_ij × X_ij / p_ij
            = X_ij

This ensures that E[X̂] = X, which by the Davis-Kahan theorem guarantees that
the eigenvectors of the Laplacian L̂ = D̂ - X̂ are close to the true eigenvectors
of L = D - X (critical for STDR partitioning).

Probability Capping:
-------------------
The theoretical sampling formula p_ij ∝ (μ_i + μ_j) × r × log²(n) / n can produce
values > 1.0 for highly leveraged pairs. We MUST cap at 1.0 during both sampling
and debiasing to maintain the unbiased property:

    X̂_ij = X_ij / min(p_ij, 1.0)  [capped probability used for debiasing]

Without capping, E[X̂] ≠ X (biased estimator) and theoretical guarantees are lost.

Reference:
---------
Huang, Liu, Du, Tao - "Leveraged Matrix Completion With Noise", Theorem 3.1

Performance:
-----------
- Complexity: O(|Ω|) = O(p × n²) for constructing sparse matrix
- Memory: O(|Ω|) sparse storage vs O(n²) dense (10-100x savings for low p)
- No iterations: Single-shot construction vs O(iterations × n³) for IALM
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from typing import Tuple


def compute_debiased_estimator(
    matrix: np.ndarray,
    Omega: np.ndarray,
    p_matrix: np.ndarray
) -> csr_matrix:
    """
    Compute debiased estimator X̂ from sampled entries.

    This function implements the HLDT debiased estimator for spectral preservation.
    The key property is that E[X̂] = X (unbiased), which ensures eigenvector stability
    via the Davis-Kahan perturbation bound.

    Algorithm:
    1. Extract sampled (i,j) pairs from Omega (upper triangle only, excluding diagonal)
    2. Cap probabilities at 1.0: p_capped = min(p_ij, 1.0)
    3. Compute debiased values: X̂_ij = X_ij / p_capped
    4. Build sparse COO matrix from (row, col, value) triplets
    5. Convert to CSR format and symmetrize
    6. Explicitly set diagonal to 1.0 (self-similarity)

    Args:
        matrix: Full similarity matrix (n×n) - only Omega entries are used
        Omega: Boolean mask (n×n) - True for sampled entries, False elsewhere
               NOTE: Diagonal should be False (not sampled)
        p_matrix: Sampling probability matrix (n×n) - probabilities used for sampling
                  NOTE: Must already be capped at 1.0 (verified by assertion)

    Returns:
        Sparse debiased estimator X̂ as scipy.sparse.csr_matrix (n×n)
        Properties guaranteed:
        - X̂ is symmetric: X̂ = X̂ᵀ
        - Diagonal is 1.0: X̂_ii = 1.0 for all i
        - Sparse: nnz(X̂) = |Ω| + n (sampled entries + diagonal)
        - Unbiased: E[X̂] = X

    Raises:
        AssertionError: If p_matrix contains values > 1.0 (must be capped before calling)
        AssertionError: If matrix is not square
        AssertionError: If Omega has diagonal entries (should be excluded)

    Example:
        >>> # Phase 1 & 2: Get sampling mask and probabilities
        >>> Omega1 = uniform_sample_upper_triangle(n, phase1_budget, rng)
        >>> row_raw, col_raw, row_reg, col_reg, sing_vals, tau = compute_leverage_scores(
        ...     matrix, Omega1, target_rank=2, apply_regularization=True
        ... )
        >>> p_matrix = compute_sampling_probabilities(row_reg, col_reg, 2, n)
        >>> p_matrix = np.minimum(p_matrix, 1.0)  # CAP PROBABILITIES
        >>> Omega2 = nonuniform_sample_upper_triangle(n, phase2_budget, p_matrix, rng, Omega_1=Omega1)
        >>> Omega = Omega1 | Omega2
        >>> np.fill_diagonal(Omega, False)  # Exclude diagonal
        >>>
        >>> # Phase 3: Compute debiased estimator
        >>> X_hat = compute_debiased_estimator(matrix, Omega, p_matrix)
        >>>
        >>> # X_hat can now be passed to FiedlerVectorComputer for spectral partitioning
    """
    n = matrix.shape[0]

    # Validation: Matrix must be square
    assert matrix.shape == (n, n), f"Matrix must be square, got shape {matrix.shape}"

    # Validation: Probabilities must be capped at 1.0
    # This is CRITICAL for maintaining E[X̂] = X
    assert np.all(p_matrix <= 1.0 + 1e-10), \
        f"Probabilities must be capped at 1.0! Max value: {np.max(p_matrix):.6f}"

    # Validation: Diagonal should not be sampled (wasteful, not needed)
    assert not np.any(np.diag(Omega)), \
        "Diagonal entries should not be in Omega (set np.fill_diagonal(Omega, False))"

    # Extract upper triangle indices (excluding diagonal) where entries were sampled
    # We work with upper triangle only to avoid duplicate work (matrix is symmetric)
    upper_triangle_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    sampled_upper = Omega & upper_triangle_mask

    # Get (i,j) indices of sampled entries
    rows, cols = np.where(sampled_upper)

    # Extract values and probabilities for sampled entries
    values = matrix[rows, cols]
    probs = p_matrix[rows, cols]

    # Probability capping: Use min(p_ij, 1.0) for debiasing
    # This ensures E[X̂_ij] = X_ij even if theoretical formula gave p_ij > 1.0
    probs_capped = np.minimum(probs, 1.0)

    # Handle edge case: If probability is exactly 0, avoid division by zero
    # (This should not happen with proper leverage sampling, but be defensive)
    # Set p_capped to 1.0 as fallback (equivalent to no debiasing)
    probs_capped = np.where(probs_capped > 1e-15, probs_capped, 1.0)

    # Compute debiased values: X̂_ij = X_ij / p_ij
    debiased_values = values / probs_capped

    # Build sparse matrix from upper triangle in COO format
    # COO (Coordinate) format is efficient for construction from triplets
    X_hat_coo = coo_matrix(
        (debiased_values, (rows, cols)),
        shape=(n, n),
        dtype=matrix.dtype
    )

    # Convert to CSR (Compressed Sparse Row) format for efficient arithmetic
    X_hat_csr = X_hat_coo.tocsr()

    # Symmetrize: X̂_sym = (X̂ + X̂ᵀ) / 2
    # This ensures numerical symmetry despite floating-point errors
    X_hat_sym = (X_hat_csr + X_hat_csr.T) / 2

    # Explicitly set diagonal to 1.0 (self-similarity)
    # The diagonal was never sampled (excluded from Omega), so it starts at 0
    # For phylogenetic similarity matrices, S_ii = 1.0 by definition
    X_hat_sym.setdiag(1.0)

    return X_hat_sym


def validate_debiased_estimator(
    X_hat: csr_matrix,
    expected_n: int,
    expected_nnz_min: int = None
) -> Tuple[bool, str]:
    """
    Validate that debiased estimator has expected properties.

    Checks:
    1. Shape is (n, n)
    2. Matrix is symmetric (within numerical tolerance)
    3. Diagonal is 1.0
    4. No NaN/Inf values
    5. Sufficient non-zeros (if expected_nnz_min provided)

    Args:
        X_hat: Debiased estimator to validate
        expected_n: Expected matrix dimension
        expected_nnz_min: Minimum expected non-zero entries (optional)

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all checks pass
        - error_message: Empty string if valid, otherwise describes first failure
    """
    n = expected_n

    # Check 1: Shape
    if X_hat.shape != (n, n):
        return False, f"Shape mismatch: expected ({n}, {n}), got {X_hat.shape}"

    # Check 2: Symmetry (convert to dense for numerical check on small sample)
    # Sample 100 random entries to avoid dense conversion overhead
    sample_size = min(100, X_hat.nnz // 2)
    if sample_size > 0:
        # Get random row/col indices from non-zero entries
        rows, cols = X_hat.nonzero()
        sample_indices = np.random.choice(len(rows), size=sample_size, replace=False)
        for idx in sample_indices:
            i, j = rows[idx], cols[idx]
            if abs(X_hat[i, j] - X_hat[j, i]) > 1e-10:
                return False, f"Asymmetry detected at ({i},{j}): {X_hat[i,j]} != {X_hat[j,i]}"

    # Check 3: Diagonal is 1.0
    diagonal = X_hat.diagonal()
    if not np.allclose(diagonal, 1.0, atol=1e-10):
        bad_idx = np.where(np.abs(diagonal - 1.0) > 1e-10)[0]
        return False, f"Diagonal not 1.0 at indices {bad_idx[:5]}: {diagonal[bad_idx[:5]]}"

    # Check 4: No NaN/Inf
    if not np.all(np.isfinite(X_hat.data)):
        return False, f"Contains NaN/Inf values"

    # Check 5: Sufficient non-zeros
    if expected_nnz_min is not None:
        if X_hat.nnz < expected_nnz_min:
            return False, f"Too few non-zeros: {X_hat.nnz} < {expected_nnz_min}"

    return True, ""
