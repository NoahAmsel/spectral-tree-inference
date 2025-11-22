"""Fiedler vector computation class."""
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from .logging import suppress_warnings, log_warning


class FiedlerVectorComputer:
    """
    Encapsulates Fiedler vector computation with automatic sparse/dense selection.
    
    Keeps all existing math logic unchanged, just provides a clean class interface.
    """
    
    def __init__(self, sparsity_threshold: float = 0.5):
        """
        Initialize the Fiedler vector computer.
        
        Args:
            sparsity_threshold: Threshold for switching to sparse methods (default 0.5)
        """
        self.sparsity_threshold = sparsity_threshold
        self.last_method_used = None
    
    def compute(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the Fiedler vector with deterministic sign convention.
        
        Automatically selects dense or sparse computation based on sparsity.
        
        Args:
            similarity_matrix: Similarity matrix (n x n)
            
        Returns:
            Fiedler vector with consistent sign convention
        """
        # Detect sparsity
        if isinstance(similarity_matrix, csr_matrix):
            sparsity = 1.0 - similarity_matrix.nnz / (similarity_matrix.shape[0] ** 2)
            is_sparse_format = True
        else:
            sparsity = 1.0 - np.count_nonzero(similarity_matrix) / similarity_matrix.size
            is_sparse_format = False
        
        # Choose method based on sparsity
        if sparsity > self.sparsity_threshold and similarity_matrix.shape[0] > 100:
            self.last_method_used = 'sparse'
            return self._compute_sparse(similarity_matrix, is_sparse_format)
        else:
            self.last_method_used = 'dense'
            return self._compute_dense(similarity_matrix)
    
    def _compute_dense(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Compute Fiedler vector using dense methods.
        
        Args:
            similarity_matrix: Similarity matrix
            
        Returns:
            Fiedler vector
        """
        # Compute the unnormalized Laplacian
        laplacian = self._compute_laplacian(similarity_matrix)
        
        # Suppress numpy/scipy warnings and log in standardized format
        with suppress_warnings('fiedler'):
            # Compute the two smallest eigenvalues and corresponding eigenvectors
            eigvals, eigvecs = scipy.linalg.eigh(laplacian, subset_by_index=(0, 1))
        
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler_vector = eigvecs[:, 1]
        
        # Enforce a consistent sign convention
        return self._apply_sign_convention(fiedler_vector)
    
    def _compute_sparse(self, similarity_matrix: np.ndarray, is_sparse_format: bool = False) -> np.ndarray:
        """
        Compute Fiedler vector using sparse methods (optimized for sparse matrices).
        
        Args:
            similarity_matrix: Similarity matrix
            is_sparse_format: Whether the matrix is already in sparse format
            
        Returns:
            Fiedler vector
        """
        # Convert to sparse matrix if not already
        if not is_sparse_format:
            similarity_matrix = csr_matrix(similarity_matrix)
        
        # Compute the unnormalized Laplacian: L = D - A
        degrees = np.array(similarity_matrix.sum(axis=0)).flatten()
        laplacian = diags(degrees) - similarity_matrix
        
        # Suppress warnings
        with suppress_warnings('fiedler'):
            try:
                # First attempt: Use sigma=0 to find eigenvalues near zero (more stable for Laplacian)
                # Increase maxiter and adjust tolerance for better convergence
                eigvals, eigvecs = eigsh(laplacian, k=2,  sigma=1e-10, ncv=50, which='LM', 
                                        maxiter=laplacian.shape[0] * 10, tol=1e-6)
            except Exception as e:
                log_warning('fiedler', f"Sparse eigsh with sigma=1e-10 failed: {str(e)}, trying without sigma...")
                try:
                    # Second attempt: Standard approach with increased iterations
                    eigvals, eigvecs = eigsh(laplacian, k=2, which='SM', 
                                            maxiter=laplacian.shape[0] * 10, tol=1e-6)
                except Exception as e2:
                    # Final fallback: Use dense computation
                    log_warning('fiedler', f"Sparse computation failed: {str(e2)}, falling back to dense method")
                    return self._compute_dense(similarity_matrix)
        
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # (eigsh does not guarantee order, so sort)
        idx = np.argsort(eigvals)
        fiedler_vector = eigvecs[:, idx[1]]
        
        # Enforce a consistent sign convention
        return self._apply_sign_convention(fiedler_vector)
    
    def _compute_laplacian(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Compute unnormalized Laplacian matrix from similarity matrix.
        
        L = D - similarity_matrix, where D is diagonal degree matrix.
        
        Args:
            similarity_matrix: Similarity matrix
            
        Returns:
            Laplacian matrix
        """
        degrees = np.sum(similarity_matrix, axis=0)
        D = np.diag(degrees)
        L = D - similarity_matrix
        return L
    
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

