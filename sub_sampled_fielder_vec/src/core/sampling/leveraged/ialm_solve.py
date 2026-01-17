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
    
    for k in range(max_iter):
        # Store previous values for convergence check
        L_prev = L.copy()
        S_prev = S.copy()
        
        # Update L: L_{k+1} = SVT(X - S_k - Y_k/μ_k, 1/μ_k)
        # But we only observe entries in Omega, so:
        # We need to fill in unobserved entries with current estimate
        X_estimate = L + S + Y / mu
        X_estimate = P_Omega(X) + (1 - Omega) * X_estimate  # Observed: use X, unobserved: use estimate
        
        L, issue = singular_value_threshold(X_estimate - S - Y / mu, 1.0 / mu)
        if issue:
            had_numerical_issues = True
        
        # Update S: S_{k+1} = soft_thresh(X - L_{k+1} - Y_k/μ_k, λ/μ_k)
        # Again, only update observed entries
        S_update = soft_threshold(X_estimate - L - Y / mu, lambda_param / mu)
        S = P_Omega(S_update)  # Only keep observed entries
        
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
        
        # Update penalty parameter
        mu = rho * mu
    
    # Store result info for logging (accessible via module-level tracking)
    ialm_solve._last_result = IALMResult(L, S, converged, final_iter, had_numerical_issues)
    
    return L, S
