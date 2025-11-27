"""Fiedler vector computation class."""
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import eigsh

from .logging import suppress_warnings, log_warning, log_info


class FiedlerVectorComputer:
    """
    Computes Fiedler vectors from Laplacian matrices using optimized dense/sparse eigensolvers.

    The Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
    of the graph Laplacian. This class automatically selects the appropriate solver
    (dense or sparse) based on matrix size and density.

    Optimizations:
    - Efficient density estimation (avoids O(N²) counting for large matrices)
    - Optimized sparse solver parameters (sigma=-1e-3, ncv=20)
    - Connectivity checking (warns if graph is disconnected)
    - Consistent sign convention enforcement
    """
    
    def __init__(self, sparsity_threshold: float = 0.9, size_threshold: int = 1024):
        """
        Initialize the Fiedler vector computer.

        Args:
            sparsity_threshold: Minimum sparsity for sparse methods (default 0.9 = 90% zeros)
            size_threshold: Matrix size below which dense methods are preferred (default 1024)
        """
        self.sparsity_threshold = sparsity_threshold
        self.size_threshold = size_threshold
        self.last_method_used = None
    
    def compute(self, laplacian: np.ndarray, sampling_prob: float = None) -> np.ndarray:
        """
        Compute Fiedler vector from Laplacian matrix.

        Automatically selects dense or sparse solver based on matrix size and density.

        Args:
            laplacian: Laplacian matrix L = D - S (where S is similarity matrix)
            sampling_prob: Optional density hint to avoid O(N²) counting (e.g., if L came from subsampled S with probability p)

        Returns:
            Fiedler vector with consistent sign convention
        """
        n = laplacian.shape[0]

        # Determine whether to use dense or sparse solver
        # Selection logic: Use DENSE if N < size_threshold OR density > 0.1 (sparsity < 0.9)

        if issparse(laplacian):
            # Already in sparse format - use sparse solver
            self.last_method_used = 'sparse'
            return self._solve_sparse_laplacian(laplacian)

        # For dense matrices, decide based on size and density
        if n < self.size_threshold:
            # Small matrix: use dense solver
            self.last_method_used = 'dense'
            return self._solve_dense_laplacian(laplacian)

        # For large matrices, estimate density to decide
        # Optimization: Avoid O(N²) density computation for large matrices
        if sampling_prob is not None:
            # If sampling probability is known, use it as density estimate
            density = sampling_prob
        elif n < 1000:
            # For medium matrices, counting nonzeros is acceptable
            density = np.count_nonzero(laplacian) / laplacian.size
        else:
            # For large matrices, estimate from sample (much faster than full count)
            sample_size = min(100, n)
            sample_indices = np.random.choice(n, sample_size, replace=False)
            sample_density = np.count_nonzero(laplacian[sample_indices]) / (sample_size * n)
            density = sample_density

        # Select solver based on density
        use_dense = density > (1.0 - self.sparsity_threshold)

        if use_dense:
            self.last_method_used = 'dense'
            return self._solve_dense_laplacian(laplacian)
        else:
            self.last_method_used = 'sparse'
            # Convert to sparse format for efficiency
            laplacian_sparse = csr_matrix(laplacian)
            return self._solve_sparse_laplacian(laplacian_sparse)

    def _solve_dense_laplacian(self, laplacian: np.ndarray) -> np.ndarray:
        """
        Solve eigenvalue problem L*v = λ*v using dense solver.

        Uses scipy.linalg.eigh to compute the 2 smallest eigenvalues/eigenvectors.
        Optimal for small or dense matrices.

        Args:
            laplacian: Laplacian matrix L

        Returns:
            Fiedler vector (eigenvector corresponding to 2nd smallest eigenvalue)
        """
        # Suppress numpy/scipy warnings and log in standardized format
        with suppress_warnings('fiedler'):
            # Compute the two smallest eigenvalues and corresponding eigenvectors
            eigvals, eigvecs = scipy.linalg.eigh(laplacian, subset_by_index=(0, 1))

        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler_vector = eigvecs[:, 1]

        # Enforce a consistent sign convention
        return self._apply_sign_convention(fiedler_vector)

    def _solve_sparse_laplacian(self, laplacian) -> np.ndarray:
        """
        Solve eigenvalue problem L*v = λ*v using sparse solver.

        Uses scipy.sparse.linalg.eigsh (ARPACK) with optimized parameters:
        - sigma=-1e-3 for stable shift-invert near zero eigenvalues
        - ncv=20 for efficient Krylov subspace
        - Connectivity check to detect disconnected graphs

        Optimal for large sparse matrices.

        Args:
            laplacian: Sparse Laplacian matrix L

        Returns:
            Fiedler vector (eigenvector corresponding to 2nd smallest eigenvalue)
        """
        n = laplacian.shape[0]
        k = 2  # We need 2 smallest eigenvalues

        # Optimization #4: Reduce ncv from 50 to min(n, 20)
        # Computing 50 Lanczos vectors for just 2 eigenvalues is unnecessary overhead
        ncv = min(n - 1, 20)  # Must satisfy ncv > k and ncv <= n-1

        # Reasonable maxiter based on matrix size
        maxiter = max(n, 1000)

        # Suppress warnings
        with suppress_warnings('fiedler'):
            try:
                # Optimization #2: Change sigma from 1e-10 to -1e-3 for stability
                # A negative shift avoids singularity issues near zero better than a tiny positive one
                eigvals, eigvecs = eigsh(
                    laplacian,
                    k=k,
                    sigma=-1e-3,  # Negative shift for stability
                    ncv=ncv,
                    which='LM',
                    maxiter=maxiter,
                    tol=1e-8  # Slightly relaxed tolerance
                )
            except Exception as e:
                log_warning('fiedler', f"Sparse eigsh with sigma=-1e-3 failed: {str(e)}, trying SM mode...")
                try:
                    # Second attempt: Standard smallest magnitude approach
                    eigvals, eigvecs = eigsh(
                        laplacian,
                        k=k,
                        which='SM',
                        ncv=ncv,
                        maxiter=maxiter,
                        tol=1e-8
                    )
                except Exception as e2:
                    # Final fallback: Use dense computation
                    log_warning('fiedler', f"Sparse computation failed: {str(e2)}, falling back to dense method")
                    # Convert to dense and use dense method
                    laplacian_dense = laplacian.toarray() if hasattr(laplacian, 'toarray') else laplacian
                    return self._solve_dense_laplacian(laplacian_dense)

        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # (eigsh does not guarantee order, so sort)
        idx = np.argsort(eigvals)

        # Optimization #3: Connectivity check
        # If the 2nd eigenvalue is ≈ 0 (< 1e-9), this indicates disconnected graph components
        if eigvals[idx[1]] < 1e-9:
            log_warning('fiedler',
                f"Second eigenvalue ({eigvals[idx[1]]:.2e}) is near zero - graph may have disconnected components. "
                f"Fiedler vector may not represent a valid cut.")

        fiedler_vector = eigvecs[:, idx[1]]

        # Enforce a consistent sign convention
        return self._apply_sign_convention(fiedler_vector)

    def _apply_sign_convention(self, fiedler_vector: np.ndarray) -> np.ndarray:
        """
        Enforce a consistent sign convention on the Fiedler vector.
        
        Choose sign so that the first non-zero element is positive.
        
        Args:
            fiedler_vector: Fiedler vector
            
        Returns:
            Fiedler vector with consistent sign
        """
        first_nonzero_idx = np.argmax(np.abs(fiedler_vector) > 1e-12)
        if fiedler_vector[first_nonzero_idx] < 0:
            return -fiedler_vector
        return fiedler_vector
    
    def align_vector(self, fiedler_vector: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
        """
        Align a Fiedler vector to have consistent sign orientation with a reference vector.
        
        Alignment is based on sign vectors (binary ±1 vectors), not the actual vectors.
        
        Args:
            fiedler_vector: Fiedler vector to align
            reference_vector: Reference vector for alignment
            
        Returns:
            Aligned Fiedler vector
        """
        # Check for zero or near-zero vectors using a more numerically stable method
        # Use max absolute value as a proxy for checking if vector is non-zero
        ref_max_abs = np.max(np.abs(reference_vector))
        if ref_max_abs < 1e-12:
            raise ValueError("Reference vector has zero or near-zero norm")
        
        fiedler_max_abs = np.max(np.abs(fiedler_vector))
        if fiedler_max_abs < 1e-12:
            log_warning('align', "Fiedler vector has zero or near-zero norm, skipping alignment")
            return fiedler_vector
        
        # Convert to sign vectors (binary ±1 vectors)
        # This is numerically stable as we're just checking signs
        ref_sign = np.sign(reference_vector)
        fiedler_sign = np.sign(fiedler_vector)
        
        # Handle zero entries in sign vectors (should be rare but possible)
        # For exactly zero entries, keep them as 0
        ref_sign[ref_sign == 0] = 1  # Default to positive for zero entries
        fiedler_sign[fiedler_sign == 0] = 1  # Default to positive for zero entries
        
        # Compute dot product of sign vectors
        # Since sign vectors only contain ±1, this is numerically stable
        # Use int64 to prevent overflow for very large vectors
        sign_dot_product = np.dot(fiedler_sign.astype(np.int64), ref_sign.astype(np.int64))
        
        # Flip sign if needed (negative dot product means opposite signs)
        if sign_dot_product < 0:
            return -fiedler_vector
        else:
            return fiedler_vector

