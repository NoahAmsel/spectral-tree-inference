import numpy as np

# ----------------------------
# DEBUG FUNCTIONS
# ----------------------------

def debug_matrix(matrix, name, max_elements=5):
    """Print debug info about a matrix."""
    print(f"\n=== DEBUG: {name} ===")
    print(f"Shape: {matrix.shape}")
    print(f"Dtype: {matrix.dtype}")
    print(f"Min/Max: {np.min(matrix):.6f} / {np.max(matrix):.6f}")
    print(f"Diagonal (first {max_elements}): {np.diag(matrix)[:max_elements]}")
    print(f"First row (first {max_elements}): {matrix[0, :max_elements]}")
    print(f"Is symmetric: {np.allclose(matrix, matrix.T)}")
    print(f"Matrix hash (for identity check): {hash(matrix.tobytes())}")
    
def debug_vector(vector, name, max_elements=10):
    """Print debug info about a vector."""
    print(f"\n=== DEBUG: {name} ===")
    print(f"Shape: {vector.shape}")
    print(f"Norm: {np.linalg.norm(vector):.6f}")
    print(f"Min/Max: {np.min(vector):.6f} / {np.max(vector):.6f}")
    print(f"First {max_elements} elements: {vector[:max_elements]}")
    print(f"Last {max_elements} elements: {vector[-max_elements:]}")
    print(f"Signs (first {max_elements}): {np.sign(vector)[:max_elements]}")
    print(f"Vector hash: {hash(vector.tobytes())}")

def debug_alignment(fiedler_vector, reference_vector, aligned_vector):
    """Debug the alignment process."""
    print(f"\n=== DEBUG: ALIGNMENT PROCESS ===")
    
    # Normalize vectors for comparison
    ref_norm = reference_vector / np.linalg.norm(reference_vector)
    fiedler_norm = fiedler_vector / np.linalg.norm(fiedler_vector)
    aligned_norm = aligned_vector / np.linalg.norm(aligned_vector)
    
    # Compute correlations
    corr_before = np.dot(ref_norm, fiedler_norm)
    corr_after = np.dot(ref_norm, aligned_norm)
    
    print(f"Correlation before alignment: {corr_before:.6f}")
    print(f"Correlation after alignment: {corr_after:.6f}")
    print(f"Sign was flipped: {np.allclose(aligned_vector, -fiedler_vector)}")
    print(f"Vectors are identical: {np.allclose(aligned_vector, fiedler_vector)}")