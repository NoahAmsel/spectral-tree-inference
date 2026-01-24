"""Inexact Augmented Lagrange Multiplier (IALM) method for robust matrix completion."""
import numpy as np
from typing import Tuple
from .soft_threshold import soft_threshold
from .singular_value_threshold import singular_value_threshold
from .ialm_result import IALMResult


def ialm_solve(
    X: np.ndarray,
    Omega: np.ndarray,
    lambda_param: float,
    max_iter: int = 100,
    tol: float = 1e-6,
    mu: float = None,
    rho: float = 1.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inexact Augmented Lagrange Multiplier (IALM) method for robust matrix completion.
    
    Solves: min_{L,S} ||L||_* + λ||S||_1  s.t.  P_Ω(L + S) = P_Ω(X)
    
    Args:
        X: Observed matrix (n x n) - only entries in Omega are meaningful
        Omega: Boolean mask (n x n) indicating observed entries
        lambda_param: Regularization parameter λ = 1/(24*sqrt(n*log(n)))
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        mu: Initial penalty parameter (auto if None)
        rho: Penalty parameter growth factor
        
    Returns:
        Tuple of (L, S) where:
        - L: Recovered low-rank matrix
        - S: Sparse noise matrix
    """
    n = X.shape[0]
    
    # Initialize
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)  # Lagrange multiplier
    
    # Track numerical issues across all iterations
    had_numerical_issues = False
    
    # Set initial penalty parameter
    if mu is None:
        # Default: use norm of observed entries
        X_obs = X * Omega
        norm_val = np.linalg.norm(X_obs, ord='fro')
        if norm_val < 1e-10:
            mu = 1.0  # Fallback for near-zero matrix
        else:
            mu = 1.25 / norm_val
    
    # Projection operator: P_Omega keeps entries in Omega, zeros elsewhere
    def P_Omega(M):
        result = np.zeros_like(M)
        result[Omega] = M[Omega]
        return result
    
    # Main IALM loop
    converged = False
    final_iter = max_iter
    prev_constraint_violation = float('inf')  # Track for adaptive penalty update

    for k in range(max_iter):
        # Store previous values for convergence check
        L_prev = L.copy()
        S_prev = S.copy()

        # IALM Update Step (corrected formulation)
        # The algorithm minimizes: ||L||_* + λ||S||_1 subject to P_Ω(L + S) = P_Ω(X)
        #
        # Key insight: On Ω, constraint forces L+S=X (use observations)
        #              On Ω^c, no constraint (use current estimates)
        #
        # Correct L-update: SVT applied to matrix that equals:
        #   - On Ω: X - S_k - Y_k/μ_k (use observations minus current S and Lagrange term)
        #   - On Ω^c: L_k + S_k - S_k - Y_k/μ_k = L_k - Y_k/μ_k (maintain current estimate)
        #
        # Build the argument for SVT:
        Z = L - Y / mu  # Start with current estimate minus Lagrange term
        Z = P_Omega(X - S - Y / mu) + (1 - Omega) * Z  # Observed: use X-S-Y/μ, unobserved: L-Y/μ

        L, issue = singular_value_threshold(Z, 1.0 / mu)
        if issue:
            had_numerical_issues = True

        # Update S: S_{k+1} = soft_thresh on observed entries only
        # Correct S-update: Apply soft-threshold to:
        #   - On Ω: X - L_{k+1} - Y_k/μ_k (residual after L update)
        #   - On Ω^c: 0 (S is sparse, only non-zero on observed entries)
        S_arg = P_Omega(X - L - Y / mu)  # Only compute on observed entries
        S = soft_threshold(S_arg, lambda_param / mu)  # Soft-threshold, will preserve sparsity
        
        # Update Lagrange multiplier: Y_{k+1} = Y_k + μ_k * (X - L_{k+1} - S_{k+1})
        # Constraint: P_Omega(L + S) = P_Omega(X)
        residual = P_Omega(X - L - S)
        Y = Y + mu * residual
        
        # Check convergence
        # Convergence: ||L - L_prev||_F / ||L||_F < tol and ||S - S_prev||_F / ||S||_F < tol
        L_norm = np.linalg.norm(L, ord='fro')
        S_norm = np.linalg.norm(S, ord='fro')
        
        L_rel_change = np.linalg.norm(L - L_prev, ord='fro') / (L_norm + 1e-10)
        S_rel_change = np.linalg.norm(S - S_prev, ord='fro') / (S_norm + 1e-10)
        
        # Also check constraint violation
        constraint_violation = np.linalg.norm(residual, ord='fro')
        
        if L_rel_change < tol and S_rel_change < tol and constraint_violation < tol:
            converged = True
            final_iter = k + 1
            break
        
        # Check for numerical divergence - early exit
        if np.any(~np.isfinite(L)) or np.any(~np.isfinite(S)):
            had_numerical_issues = True
            L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
            final_iter = k + 1
            break

        # Adaptive penalty parameter update
        # Theory: Increase μ only when making progress on constraint violation
        # This prevents numerical overflow and oscillation in ill-conditioned cases
        #
        # Standard IALM: μ_{k+1} = ρ·μ_k (always increase)
        # Adaptive: μ_{k+1} = ρ·μ_k if ||residual|| < 0.25·||prev_residual|| (good progress)
        #           μ_{k+1} = μ_k otherwise (stalled - don't make problem harder)
        #
        # Benefit: Prevents μ → ∞ when convergence stalls, improves stability
        if constraint_violation < 0.25 * prev_constraint_violation:
            # Good progress on constraint - increase penalty
            mu = rho * mu
        # else: Keep current μ (don't increase if not making progress)

        prev_constraint_violation = constraint_violation
    
    # Store result info for logging (accessible via module-level tracking)
    ialm_solve._last_result = IALMResult(L, S, converged, final_iter, had_numerical_issues)
    
    return L, S
