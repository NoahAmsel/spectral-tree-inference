"""Metric computation class for spectral matrix analysis."""
import numpy as np
import scipy.linalg
from typing import Dict, Tuple

from .logging import suppress_warnings, log_warning, log_error


class MetricComputer:
    """
    Encapsulates all metric computation logic with efficient caching.
    
    Keeps all existing math logic unchanged, just provides better organization.
    """
    
    def __init__(self, empirical_rank_threshold: float = None, coherence_k: int = 2):
        """
        Initialize the metric computer.
        
        Args:
            empirical_rank_threshold: Threshold for rank computation (None = auto)
            coherence_k: Number of top singular vectors for coherence
        """
        self.empirical_rank_threshold = empirical_rank_threshold
        self.coherence_k = coherence_k
    
    def compute_all(self, M: np.ndarray, S: np.ndarray, L_M: np.ndarray, 
                   L_S: np.ndarray, p: float) -> Dict[str, float]:
        """
        Compute all metrics efficiently for M, S, L_M, and L_S.
        
        This function optimizes computation by:
        - Computing SVD once per matrix and reusing for rank and coherence
        - Computing eigenvalues once per matrix and reusing for gap and min_separation
        
        Args:
            M: Full similarity matrix
            S: Subsampled similarity matrix
            L_M: Laplacian of M
            L_S: Laplacian of S
            p: Sampling probability
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # 1. Operator norm error (comparison metric)
        metrics['operator_norm_error'] = self.compute_comparison_metrics(M, S, p)
        
        # 2. Compute metrics for each matrix
        matrices = [
            ('M', M),
            ('S', S),
            ('L_M', L_M),
            ('L_S', L_S)
        ]
        
        for name, matrix in matrices:
            matrix_metrics = self.compute_matrix_metrics(matrix, name)
            metrics.update(matrix_metrics)
        
        return metrics
    
    def compute_matrix_metrics(self, matrix: np.ndarray, name: str) -> Dict[str, float]:
        """
        Compute all metrics for a single matrix using optimized eigenvalue decomposition.

        For symmetric matrices, uses eigenvalue decomposition instead of SVD
        (2-3x faster) and computes numerical rank using norm formula.

        Args:
            matrix: Input symmetric matrix
            name: Matrix name (for metric keys)

        Returns:
            Dictionary of metrics for this matrix
        """
        metrics = {}

        # Compute largest eigenvalues once for coherence and numerical rank
        try:
            eigvecs, singular_values = self._compute_largest_eigenvalues(matrix, k=self.coherence_k)

            # Numerical rank using norm formula: NumRank = ||S||²_F / ||S||²_2
            try:
                metrics[f'empirical_rank_{name}'] = self._compute_numerical_rank(
                    matrix,
                    largest_singular_value=singular_values[0]
                )
            except Exception as e:
                log_warning('metrics', f"Failed to compute numerical rank for {name}: {e}")
                metrics[f'empirical_rank_{name}'] = 0.0

            # Coherence from eigenvectors (same as left singular vectors for symmetric matrices)
            try:
                metrics[f'coherence_{name}'] = self._compute_coherence(eigvecs)
            except Exception as e:
                log_warning('metrics', f"Failed to compute coherence for {name}: {e}")
                metrics[f'coherence_{name}'] = float('nan')
        except Exception as e:
            # If eigenvalue computation fails, set both metrics to NaN/0
            log_warning('metrics', f"Eigenvalue computation failed for {name}: {e}")
            metrics[f'empirical_rank_{name}'] = 0.0
            metrics[f'coherence_{name}'] = float('nan')

        # Compute smallest eigenvalues for spectral gap and min_separation
        try:
            eigvals, _ = self._compute_smallest_eigenvalues(matrix, k=3)

            # Spectral gap from eigenvalues
            try:
                metrics[f'spectral_gap_{name}'] = self._compute_spectral_gap(eigvals)
            except Exception as e:
                log_warning('metrics', f"Failed to compute spectral gap for {name}: {e}")
                metrics[f'spectral_gap_{name}'] = float('nan')

            # Minimum separation from eigenvalues
            try:
                metrics[f'min_separation_{name}'] = self._compute_min_separation(eigvals)
            except Exception as e:
                log_warning('metrics', f"Failed to compute minimum separation for {name}: {e}")
                metrics[f'min_separation_{name}'] = float('nan')
        except Exception as e:
            # If eigenvalue computation fails, set both metrics to NaN
            log_warning('metrics', f"Eigenvalue computation failed for {name}: {e}")
            metrics[f'spectral_gap_{name}'] = float('nan')
            metrics[f'min_separation_{name}'] = float('nan')

        return metrics
    
    def compute_comparison_metrics(self, M: np.ndarray, S: np.ndarray, p: float) -> float:
        """
        Compute comparison metrics between M and S (Operator norm error).
        
        Args:
            M: Full similarity matrix
            S: Subsampled similarity matrix
            p: Sampling probability
            
        Returns:
            Operator norm error
        """
        try:
            return self._compute_operator_norm_error(M, S, p)
        except Exception as e:
            log_warning('metrics', f"Failed to compute operator norm error: {e}")
            return float('nan')
    
    # ============================================================================
    # Internal computation methods (keep existing math logic)
    # ============================================================================
    
    def _compute_largest_eigenvalues(self, matrix: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute largest k eigenvalues/eigenvectors for symmetric matrix.

        For symmetric matrices, eigenvalue decomposition is 2-3x faster than SVD
        and gives the same information: singular values = |eigenvalues|.

        Args:
            matrix: Symmetric matrix (M, S, L_M, or L_S)
            k: Number of largest eigenvalues to compute

        Returns:
            eigenvectors: Left singular vectors (n x k matrix)
            singular_values: Singular values (|eigenvalues|) in descending order
        """
        n = matrix.shape[0]
        k = min(k, n - 1)  # Can't compute more than n-1 eigenvalues

        with suppress_warnings('metrics'):
            try:
                # Compute LARGEST k eigenvalues using subset_by_index
                # subset_by_index=(n-k, n-1) gives last k eigenvalues (largest in magnitude)
                eigvals, eigvecs = scipy.linalg.eigh(matrix, subset_by_index=(n-k, n-1))

                # Sort by descending absolute value (like singular values)
                # This handles potential negative eigenvalues correctly
                idx = np.argsort(np.abs(eigvals))[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]

                # Convert eigenvalues to singular values (absolute value)
                singular_values = np.abs(eigvals)

            except (scipy.linalg.LinAlgError, ValueError) as e:
                log_error('metrics', f"Eigenvalue computation failed: {e}")
                raise

        return eigvecs, singular_values
    
    def _compute_smallest_eigenvalues(self, matrix: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Compute smallest k eigenvalues efficiently."""
        n = matrix.shape[0]
        k = min(k, n - 1)  # Can't compute more than n-1 eigenvalues
        
        with suppress_warnings('metrics'):
            try:
                eigvals, eigvecs = scipy.linalg.eigh(matrix, subset_by_index=(0, k - 1))
            except (scipy.linalg.LinAlgError, ValueError) as e:
                log_error('metrics', f"Eigenvalue computation failed: {e}")
                raise
        return eigvals, eigvecs
    
    def _compute_operator_norm_error(self, M: np.ndarray, S: np.ndarray, p: float) -> float:
        """
        Compute operator (spectral) norm of scaled error matrix: E_scaled = S - M.
        
        The operator norm is the largest singular value of a matrix, which represents
        the maximum factor by which the matrix stretches any unit vector.
        """
        if p <= 0:
            raise ValueError(f"Sampling probability p must be positive, got {p}")
        
        # S is already scaled by 1/p, so error is simply S - M
        E_scaled = S - M
        return float(np.linalg.norm(E_scaled, ord=2))
    
    def _compute_numerical_rank(self, matrix: np.ndarray, largest_singular_value: float) -> float:
        """
        Compute numerical rank using the formula: NumRank(S) = ||S||²_F / ||S||²_2.

        This is more robust than counting singular values above a threshold and
        can be computed efficiently without full SVD.

        Args:
            matrix: Input matrix
            largest_singular_value: ||S||_2 (spectral norm, from eigenvalue decomposition)

        Returns:
            Numerical rank (continuous value, not integer)
        """
        # Handle zero or near-zero matrices
        if largest_singular_value < 1e-14:
            return 0.0

        # Frobenius norm: O(n²) - single pass over all elements
        frobenius_norm = float(np.linalg.norm(matrix, 'fro'))

        # Spectral norm: already have it as largest_singular_value
        spectral_norm = float(largest_singular_value)

        # NumRank = ||S||²_F / ||S||²_2
        numerical_rank = (frobenius_norm ** 2) / (spectral_norm ** 2)

        return numerical_rank
    
    def _compute_coherence(self, U: np.ndarray) -> float:
        """Compute matrix coherence from pre-computed left singular vectors."""
        if U.shape[1] == 0:
            return 0.0
        
        # Use all available vectors if coherence_k is None or larger than available
        k = self.coherence_k if self.coherence_k is not None else U.shape[1]
        k = min(k, U.shape[1])
        
        # Compute coherence for each of the top-k singular vectors
        coherences = []
        for i in range(k):
            u = U[:, i]
            # Maximum absolute squared entry
            coherence_i = float(np.max(np.abs(u) ** 2))
            coherences.append(coherence_i)
        
        # Return maximum coherence
        return float(np.max(coherences)) if coherences else 0.0
    
    def _compute_spectral_gap(self, eigvals: np.ndarray) -> float:
        """Compute spectral gap from pre-computed eigenvalues."""
        if len(eigvals) < 3:
            return 0.0
        
        lambda_1 = float(eigvals[1])  # Second smallest (Fiedler eigenvalue for Laplacians)
        lambda_2 = float(eigvals[2])  # Third smallest
        gap = lambda_2 - lambda_1
        return gap
    
    def _compute_min_separation(self, eigvals: np.ndarray) -> float:
        """Compute minimum separation from pre-computed eigenvalues."""
        if len(eigvals) < 2:
            return 0.0
        
        lambda_0 = float(eigvals[0])  # Smallest eigenvalue
        lambda_1 = float(eigvals[1])  # Second smallest (Fiedler eigenvalue for Laplacians)
        
        if len(eigvals) >= 3:
            lambda_2 = float(eigvals[2])  # Third smallest
            gap_above = lambda_2 - lambda_1
        else:
            gap_above = float('inf')  # No eigenvalue above
        
        gap_below = lambda_1 - lambda_0
        
        # Return minimum of the two gaps
        min_separation = min(gap_below, gap_above)
        return min_separation

