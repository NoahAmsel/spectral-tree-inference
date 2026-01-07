"""IALM solver for nuclear norm minimization with sparse noise."""
import numpy as np
from typing import Tuple
from scipy.linalg import svd


def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """
    Element-wise soft thresholding operator.
    
    S_tau(x) = sign(x) * max(|x| - tau, 0)
    
    Used for L1-norm proximal operator: prox_{tau||·||_1}(X)
    
    Args:
        X: Input matrix
        tau: Threshold parameter
        
    Returns:
        Soft-thresholded matrix
    """
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def singular_value_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """
    Singular value thresholding (SVT) operator.
    
    Proximal operator for nuclear norm: prox_{tau||·||_*}(X)
    
    Algorithm:
        1. Compute SVD: X = U Σ V^T
        2. Soft-threshold singular values: Σ_tau = max(Σ - tau, 0)
        3. Reconstruct: X_tau = U Σ_tau V^T
    
    Args:
        X: Input matrix (n x n)
        tau: Threshold parameter
        
    Returns:
        Matrix with thresholded singular values
    """
    U, s, Vt = svd(X, full_matrices=False)
    # Soft-threshold singular values
    s_thresh = np.maximum(s - tau, 0)
    # Reconstruct
    return U @ np.diag(s_thresh) @ Vt


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
    
    Algorithm (from paper):
        Initialize: L_0 = 0, S_0 = 0, Y_0 = 0, μ_0 > 0
        For k = 0, 1, 2, ...:
            L_{k+1} = SVT(X - S_k - Y_k/μ_k, 1/μ_k)
            S_{k+1} = soft_thresh(X - L_{k+1} - Y_k/μ_k, λ/μ_k)
            Y_{k+1} = Y_k + μ_k * (X - L_{k+1} - S_{k+1})
            μ_{k+1} = ρ * μ_k  (if not converged)
    
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
    
    # Set initial penalty parameter
    if mu is None:
        # Default: use norm of observed entries
        X_obs = X * Omega
        mu = 1.25 / np.linalg.norm(X_obs, ord='fro')
    
    # Projection operator: P_Omega keeps entries in Omega, zeros elsewhere
    def P_Omega(M):
        result = np.zeros_like(M)
        result[Omega] = M[Omega]
        return result
    
    # Main IALM loop
    for k in range(max_iter):
        # Store previous values for convergence check
        L_prev = L.copy()
        S_prev = S.copy()
        
        # Update L: L_{k+1} = SVT(X - S_k - Y_k/μ_k, 1/μ_k)
        # But we only observe entries in Omega, so:
        # We need to fill in unobserved entries with current estimate
        X_estimate = L + S + Y / mu
        X_estimate = P_Omega(X) + (1 - Omega) * X_estimate  # Observed: use X, unobserved: use estimate
        
        L = singular_value_threshold(X_estimate - S - Y / mu, 1.0 / mu)
        
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
            break
        
        # Update penalty parameter
        mu = rho * mu
    
    return L, S

