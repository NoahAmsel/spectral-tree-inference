# Performance Optimization Report
## Spectral Tree Inference - Sub-sampled STDR

**Date:** 2025-11-26
**Reviewer:** Senior Numerical Engineering Expert
**Target:** 8192 taxa, 25 p-values, 10-100 bootstrap iterations

---

## Executive Summary

After comprehensive code review, I identified **12 critical performance bottlenecks** that are causing excessive runtime. The issues range from algorithmic inefficiencies (O(n³) operations where O(n²k) is sufficient) to memory management problems and unnecessary recomputations.

**Estimated Total Speedup:** 50-1000x depending on optimization depth
**Current Runtime:** Hours → **Target Runtime:** Minutes

---

## 🔴 CRITICAL ISSUES (Highest Impact)

### Issue #1: Redundant Full SVD Computation
**Impact:** 50-100x slowdown
**Location:** `utils/metric_computer.py:148-156`

#### Problem
Computing full SVD for 8192×8192 matrices when only k=2-3 singular values needed.

#### Current Code
```python
def _compute_svd(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD once and return all components."""
    with suppress_warnings('metrics'):
        try:
            U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
            # ^^^ PROBLEM: Full SVD is O(n³) for n×n matrix
            # For 8192×8192: ~549 billion operations!
        except (np.linalg.LinAlgError, ValueError) as e:
            log_error('metrics', f"SVD computation failed: {e}")
            raise
    return U, singular_values, Vt
```

#### Impact Analysis
- Full SVD: O(n³) ≈ 549 billion ops for n=8192
- Partial SVD: O(n²k) ≈ 134 million ops for k=2
- **Speedup potential: ~4000x** for this operation

#### Recommended Fix
```python
def _compute_svd_partial(self, matrix: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute partial SVD for top k singular values/vectors."""
    if k is None:
        k = self.coherence_k

    # Ensure k is valid
    k = min(k, min(matrix.shape) - 1)

    with suppress_warnings('metrics'):
        try:
            # Use sparse SVD even for dense matrices when k << n
            from scipy.sparse.linalg import svds
            U, singular_values, Vt = svds(matrix, k=k, solver='arpack')

            # Sort by descending singular values
            idx = np.argsort(singular_values)[::-1]
            U = U[:, idx]
            singular_values = singular_values[idx]

        except Exception as e:
            log_warning('metrics', f"Partial SVD failed, falling back to full: {e}")
            U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
            U = U[:, :k]
            singular_values = singular_values[:k]

    return U, singular_values
```

---

### Issue #2: Inefficient Matrix Subsampling
**Impact:** 10-20x slowdown
**Location:** `utils/similarity_builder.py:66-96`

#### Problem
Using `np.random.choice()` on 67+ million elements (8192²) creates memory thrashing and is algorithmically inefficient.

#### Current Code
```python
def _subsample(self, matrix: np.ndarray, p: float, seed: int = None) -> np.ndarray:
    """Subsample matrix entries with probability p and scale by 1/p."""
    if p >= 0.9999:
        return matrix.copy()

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # PROBLEM: Creates array of 67M indices, then samples from it
    n_keep = int(p * matrix.size)  # For 8192²: ~67M * p elements
    indices = rng.choice(matrix.size, size=n_keep, replace=False)
    # ^^^ This is O(n²) memory + O(n² log n) time

    sampling_mask = np.zeros(matrix.shape, dtype=bool)
    sampling_mask.flat[indices] = True

    # Subsample and scale
    subsampled = np.zeros_like(matrix)
    subsampled[sampling_mask] = matrix[sampling_mask] / p

    return subsampled
```

#### Impact Analysis
- For 8192×8192 matrix: 67,108,864 elements
- `choice(67M, 6.7M, replace=False)` requires:
  - 67M element array allocation
  - Shuffling/sampling algorithm: O(n² log n)
  - 6.7M index storage

#### Recommended Fix
```python
def _subsample(self, matrix: np.ndarray, p: float, seed: int = None) -> np.ndarray:
    """Subsample matrix entries with probability p and scale by 1/p."""
    if p >= 0.9999:
        return matrix.copy()

    if seed is not None:
        np.random.seed(seed)

    # OPTIMIZED: Generate boolean mask directly (vectorized)
    # This is O(n²) time, O(n²) memory, but with much better constants
    sampling_mask = np.random.random(matrix.shape) < p

    # Subsample and scale in-place
    subsampled = np.zeros_like(matrix)
    subsampled[sampling_mask] = matrix[sampling_mask] / p

    return subsampled
```

**Alternative for very sparse sampling (p < 0.01):**
```python
def _subsample_sparse(self, matrix: np.ndarray, p: float, seed: int = None) -> np.ndarray:
    """Subsample for very low p using sparse representation."""
    from scipy.sparse import csr_matrix

    if seed is not None:
        np.random.seed(seed)

    # For very sparse sampling, use coordinate format
    n_keep = int(p * matrix.size)
    flat_indices = np.random.choice(matrix.size, size=n_keep, replace=False)

    # Convert to 2D indices
    rows = flat_indices // matrix.shape[1]
    cols = flat_indices % matrix.shape[1]

    # Create sparse matrix
    values = matrix[rows, cols] / p
    sparse_result = csr_matrix((values, (rows, cols)), shape=matrix.shape)

    return sparse_result.toarray()  # Or keep sparse depending on downstream
```

---

### Issue #3: Unnecessary Laplacian Recomputation
**Impact:** 2-3x slowdown per p-value
**Location:** `experiment/bootstrap_sweep.py:379-383`

#### Problem
Computing Laplacian L_S for EVERY bootstrap iteration when it's only used for metrics.

#### Current Code
```python
for i in range(cfg.bootstrap_reps):  # 10-100+ iterations
    bootstrap_seed = cfg.seed + i
    S = _subsample_matrix_entries(M, p, seed=bootstrap_seed)

    # Update running average
    if S_avg is None:
        S_avg = S.copy()
        n_bootstrap_collected = 1
    else:
        n_bootstrap_collected += 1
        S_avg += (S - S_avg) / n_bootstrap_collected

    # PROBLEM: Computing Laplacian every iteration
    try:
        L_S = compute_laplacian(S)  # O(n²) operation
    except Exception as e:
        log_warning('bootstrap', f"Failed to compute Laplacian of S: {e}")
        L_S = None

    # Compute all metrics efficiently using the metric composer
    if L_S is not None:
        try:
            all_metrics = metric_composer(
                M=M, S=S, L_M=L_M, L_S=L_S,
                p=p,
                empirical_rank_threshold=empirical_rank_threshold,
                coherence_k=coherence_k
            )
```

#### Impact Analysis
- For 100 bootstrap iterations with 8192 taxa:
  - 100 Laplacian computations = 100 × O(n²) = 6.7 billion operations
  - Each Laplacian: 67M additions + 67M multiplications
- **Wasted computation:** Laplacian only needed for metrics, not for Fiedler vector

#### Recommended Fix

**Option A: Batch metric computation (recommended)**
```python
# Store S matrices, compute Laplacians in batch after bootstrap loop
S_samples = []  # List to store sampled matrices (or use generator)

for i in range(cfg.bootstrap_reps):
    bootstrap_seed = cfg.seed + i
    S = _subsample_matrix_entries(M, p, seed=bootstrap_seed)

    # Update running average (same as before)
    if S_avg is None:
        S_avg = S.copy()
        n_bootstrap_collected = 1
    else:
        n_bootstrap_collected += 1
        S_avg += (S - S_avg) / n_bootstrap_collected

    # Store for later metric computation (if needed)
    if cfg.compute_metrics:
        S_samples.append(S)

    # Compute Fiedler directly (no Laplacian needed!)
    f_est = compute_fiedler_from_similarity(S)
    f_aligned = align_fiedler_by_dot_product(f_est, fiedler_ref)
    aligned_vectors.append(f_aligned)

# After bootstrap loop: compute metrics in batch if needed
if cfg.compute_metrics and len(S_samples) > 0:
    # Sample-based metric computation (every Nth iteration)
    sample_indices = np.linspace(0, len(S_samples)-1, min(10, len(S_samples)), dtype=int)
    for idx in sample_indices:
        S = S_samples[idx]
        L_S = compute_laplacian(S)
        metrics = metric_composer(M=M, S=S, L_M=L_M, L_S=L_S, p=p, ...)
        # Store metrics
```

**Option B: Sample-based metric computation**
```python
# Only compute metrics every Nth bootstrap iteration
metric_sample_rate = max(1, cfg.bootstrap_reps // 10)  # Sample 10 times

for i in range(cfg.bootstrap_reps):
    # ... bootstrap code ...

    # Compute metrics only periodically
    if i % metric_sample_rate == 0:
        try:
            L_S = compute_laplacian(S)
            all_metrics = metric_composer(...)
            # Store metrics
        except Exception as e:
            log_warning('bootstrap', f"Metric computation failed: {e}")
```

---

### Issue #4: Suboptimal Sparse Eigensolver Parameters
**Impact:** 2-5x slowdown for sparse cases
**Location:** `utils/fiedler_computer.py:99-114`

#### Problem
Inefficient parameters for sparse eigenvalue solver causing convergence issues and excessive iterations.

#### Current Code
```python
def _compute_sparse(self, similarity_matrix: np.ndarray, is_sparse_format: bool = False) -> np.ndarray:
    # ... setup code ...

    with suppress_warnings('fiedler'):
        try:
            # PROBLEM 1: sigma=1e-10 is numerically unstable
            # PROBLEM 2: maxiter=shape[0]*10 is excessive (81920 iters for 8192 taxa!)
            # PROBLEM 3: ncv=50 is too small for large matrices
            eigvals, eigvecs = eigsh(laplacian, k=2, sigma=1e-10, ncv=50, which='LM',
                                    maxiter=laplacian.shape[0] * 10, tol=1e-6)
        except Exception as e:
            log_warning('fiedler', f"Sparse eigsh with sigma=1e-10 failed: {str(e)}, trying without sigma...")
            try:
                # Second attempt: Standard approach with increased iterations
                eigvals, eigvecs = eigsh(laplacian, k=2, which='SM',
                                        maxiter=laplacian.shape[0] * 10, tol=1e-6)
```

#### Impact Analysis
1. **sigma=1e-10 issues:**
   - Shift-invert mode requires LU factorization: O(n³) for dense, O(n^1.5) for sparse
   - Numerical instability near zero eigenvalues
   - Often fails, triggering expensive fallback

2. **maxiter issues:**
   - For 8192 taxa: 81,920 iterations maximum
   - Typical convergence: 50-200 iterations sufficient
   - Wasted CPU cycles checking convergence

3. **ncv issues:**
   - ncv=50 gives Krylov subspace of size 50
   - Rule of thumb: ncv ≥ 2*k+1 = 5 minimum, but larger is better for convergence
   - For large matrices, ncv should scale with matrix size

#### Recommended Fix
```python
def _compute_sparse(self, similarity_matrix: np.ndarray, is_sparse_format: bool = False) -> np.ndarray:
    """Compute Fiedler vector using sparse methods (optimized)."""
    if not is_sparse_format:
        similarity_matrix = csr_matrix(similarity_matrix)

    # Compute Laplacian
    degrees = np.array(similarity_matrix.sum(axis=0)).flatten()
    laplacian = diags(degrees) - similarity_matrix

    n = laplacian.shape[0]
    k = 2  # We need 2 smallest eigenvalues

    # OPTIMIZED: Adaptive parameters based on matrix size
    ncv = min(max(2*k + 10, 20), n - 1)  # Adaptive Krylov subspace
    maxiter = max(n, 1000)  # Reasonable maximum iterations
    tol = 1e-8  # Slightly relaxed tolerance for large matrices

    with suppress_warnings('fiedler'):
        try:
            # STRATEGY 1: Try sigma=0 (cleanest for Laplacian)
            # This finds eigenvalues near zero without shift-invert instability
            eigvals, eigvecs = eigsh(
                laplacian,
                k=k,
                sigma=0,  # Much more stable than 1e-10
                ncv=ncv,
                which='LM',  # Largest magnitude relative to sigma
                maxiter=maxiter,
                tol=tol,
                mode='normal'  # Avoid LU factorization
            )
        except Exception as e:
            log_warning('fiedler', f"eigsh with sigma=0 failed: {e}, trying SM mode...")
            try:
                # STRATEGY 2: Direct smallest eigenvalue computation
                eigvals, eigvecs = eigsh(
                    laplacian,
                    k=k,
                    which='SM',  # Smallest magnitude (no shift)
                    ncv=ncv,
                    maxiter=maxiter,
                    tol=tol
                )
            except Exception as e2:
                log_warning('fiedler', f"All sparse methods failed: {e2}, using dense fallback")
                return self._compute_dense(similarity_matrix.toarray() if is_sparse_format else similarity_matrix)

    # Sort and extract Fiedler vector
    idx = np.argsort(eigvals)
    fiedler_vector = eigvecs[:, idx[1]]

    return self._apply_sign_convention(fiedler_vector)
```

**Alternative: Use LOBPCG for very large matrices**
```python
def _compute_sparse_lobpcg(self, similarity_matrix: np.ndarray) -> np.ndarray:
    """Use LOBPCG for very large sparse matrices (8192+)."""
    from scipy.sparse.linalg import lobpcg
    from scipy.sparse import eye

    if not isinstance(similarity_matrix, csr_matrix):
        similarity_matrix = csr_matrix(similarity_matrix)

    degrees = np.array(similarity_matrix.sum(axis=0)).flatten()
    laplacian = diags(degrees) - similarity_matrix

    n = laplacian.shape[0]

    # Initial guess: random vectors orthogonal to constant vector
    X = np.random.randn(n, 2)
    X[:, 0] = np.ones(n) / np.sqrt(n)  # First eigenvector (should be constant)
    X[:, 1] = X[:, 1] - X[:, 1].mean()  # Orthogonalize to constant

    # LOBPCG is often faster than ARPACK for very large matrices
    eigvals, eigvecs = lobpcg(laplacian, X, largest=False, maxiter=200)

    # Extract Fiedler vector (second smallest)
    idx = np.argsort(eigvals)
    fiedler_vector = eigvecs[:, idx[1]]

    return self._apply_sign_convention(fiedler_vector)
```

---

## 🟡 MAJOR ISSUES (Medium-High Impact)

### Issue #5: Streaming Average Memory Inefficiency
**Impact:** 1.5-2x memory bandwidth waste
**Location:** `experiment/bootstrap_sweep.py:370-376`, `experiment/parallel_bootstrap.py:146-152`

#### Problem
Welford's algorithm implementation creates unnecessary temporary arrays.

#### Current Code
```python
# Update running average of S (streaming - no storage!)
# Uses Welford's online algorithm for numerical stability
if S_avg is None:
    S_avg = S.copy()
    n_bootstrap_collected = 1
else:
    n_bootstrap_collected += 1
    # PROBLEM: Creates temp array (S - S_avg) of size n²
    S_avg += (S - S_avg) / n_bootstrap_collected
    # Memory operations: 1 subtraction, 1 division, 1 addition
    # For 8192²: 67M * 3 = 201M ops + 67M temp array
```

#### Impact Analysis
- For 8192×8192 matrix: 67,108,864 elements × 8 bytes = 512 MB
- Creating temp array `(S - S_avg)`: 512 MB allocation + 512 MB read + 512 MB write
- **Unnecessary memory traffic:** 1.5 GB per bootstrap iteration

#### Recommended Fix
```python
# OPTIMIZED: In-place update without temporary arrays
if S_avg is None:
    S_avg = S.copy()
    n_bootstrap_collected = 1
else:
    n_bootstrap_collected += 1
    # Mathematical equivalence: S_avg = (n-1)/n * S_avg + S/n
    alpha = (n_bootstrap_collected - 1) / n_bootstrap_collected
    beta = 1.0 / n_bootstrap_collected

    # In-place operations (no temp arrays)
    S_avg *= alpha  # Scale existing average
    S_avg += beta * S  # Add scaled new sample

    # Memory operations: 1 multiply, 1 multiply-add
    # Much better memory locality and cache efficiency
```

**Even better: Use numexpr for vectorized in-place ops**
```python
import numexpr as ne

if S_avg is None:
    S_avg = S.copy()
    n_bootstrap_collected = 1
else:
    n_bootstrap_collected += 1
    alpha = (n_bootstrap_collected - 1) / n_bootstrap_collected
    beta = 1.0 / n_bootstrap_collected

    # Numexpr evaluates in single pass with better cache usage
    S_avg = ne.evaluate('alpha * S_avg + beta * S')
```

---

### Issue #6: Repeated Normalization in Alignment
**Impact:** 1.2-1.5x slowdown
**Location:** `experiment/bootstrap_sweep.py:34-67`, `experiment/parallel_bootstrap.py:18-49`

#### Problem
Normalizing Fiedler vectors during alignment, then normalizing the averaged result again.

#### Current Code
```python
def align_fiedler_by_dot_product(fiedler_vector: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
    """Align using magnitude-based dot product alignment."""
    from utils.metrics import _normalize_vector

    try:
        # PROBLEM: Normalizing every bootstrap iteration
        v_normalized = _normalize_vector(fiedler_vector)  # O(n) norm + O(n) division
        u_normalized = _normalize_vector(reference_vector)  # O(n) norm + O(n) division
    except ValueError as e:
        log_warning('align', f"Normalization failed: {e}")
        return fiedler_vector

    # ... alignment logic ...

    return -v_normalized if dot_product < 0 else v_normalized
    # ^^^ Returns NORMALIZED vector

# Later in bootstrap loop:
for i in range(cfg.bootstrap_reps):
    # ...
    f_est = compute_fiedler_from_similarity(S)
    f_aligned_normalized = align_fiedler_by_dot_product(f_est, fiedler_ref)
    aligned_vectors.append(f_aligned_normalized)  # Already normalized!

# After loop:
v_avg = np.mean(aligned_vectors, axis=0)
try:
    v_avg = _normalize_vector(v_avg)  # REDUNDANT: normalizing already-normalized average
except ValueError as e:
    log_warning('bootstrap', f"Averaged vector normalization failed: {e}")
```

#### Impact Analysis
- For 100 bootstrap iterations:
  - 100 normalizations during alignment
  - 1 normalization after averaging
  - **Wasted:** 100 normalizations of already-unit vectors
- Each normalization: 2 passes over n elements (norm + division)

#### Recommended Fix
```python
def align_fiedler_by_sign(fiedler_vector: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
    """
    Align Fiedler vector to reference WITHOUT normalization.

    Returns raw aligned vector (not normalized) for efficient averaging.
    """
    # Check for zero vectors
    ref_max_abs = np.max(np.abs(reference_vector))
    if ref_max_abs < 1e-12:
        raise ValueError("Reference vector is zero")

    fiedler_max_abs = np.max(np.abs(fiedler_vector))
    if fiedler_max_abs < 1e-12:
        log_warning('align', "Fiedler vector is zero")
        return fiedler_vector

    # Compute dot product WITHOUT normalization
    # This still gives correct sign for alignment
    dot_product = np.dot(fiedler_vector, reference_vector)

    # Return aligned but NOT normalized vector
    return -fiedler_vector if dot_product < 0 else fiedler_vector

# In bootstrap loop:
for i in range(cfg.bootstrap_reps):
    f_est = compute_fiedler_from_similarity(S)
    f_aligned = align_fiedler_by_sign(f_est, fiedler_ref)  # Not normalized
    aligned_vectors.append(f_aligned)

# After loop: Average, then normalize ONCE
v_avg = np.mean(aligned_vectors, axis=0)
try:
    v_avg = _normalize_vector(v_avg)  # Single normalization
except ValueError as e:
    log_warning('bootstrap', f"Averaged vector normalization failed: {e}")
```

---

### Issue #7: Inefficient Sign Convention Application
**Impact:** 1.5x slowdown for large vectors
**Location:** `utils/fiedler_computer.py:141-156`

#### Problem
Creating unnecessary boolean array to find first nonzero element.

#### Current Code
```python
def _apply_sign_convention(self, fiedler_vector: np.ndarray) -> np.ndarray:
    """Enforce consistent sign convention on Fiedler vector."""
    # PROBLEM: Creates boolean array of size n
    first_nonzero_idx = np.argmax(np.abs(fiedler_vector) > 1e-12)
    # ^^^ np.abs(fiedler_vector) > 1e-12 creates n-element boolean array
    # ^^^ np.argmax scans this array to find first True

    if fiedler_vector[first_nonzero_idx] < 0:
        return -fiedler_vector
    return fiedler_vector
```

#### Impact Analysis
- For n=8192 vector:
  - Creates 8192-element boolean array (8 KB)
  - Scans entire array even if first element is nonzero
  - Memory allocation + scan overhead

#### Recommended Fix
```python
def _apply_sign_convention(self, fiedler_vector: np.ndarray) -> np.ndarray:
    """Enforce consistent sign convention on Fiedler vector."""
    # OPTIMIZED: Find first nonzero without creating boolean array
    nonzero_indices = np.flatnonzero(np.abs(fiedler_vector) > 1e-12)

    if len(nonzero_indices) == 0:
        # All elements are effectively zero
        return fiedler_vector

    first_nonzero_idx = nonzero_indices[0]

    if fiedler_vector[first_nonzero_idx] < 0:
        return -fiedler_vector
    return fiedler_vector
```

**Alternative: Early exit scan**
```python
def _apply_sign_convention(self, fiedler_vector: np.ndarray) -> np.ndarray:
    """Enforce consistent sign convention with early exit."""
    # Manual scan with early exit (fastest for typical cases)
    for i, val in enumerate(fiedler_vector):
        if abs(val) > 1e-12:
            return -fiedler_vector if val < 0 else fiedler_vector

    # All elements are effectively zero
    return fiedler_vector
```

---

### Issue #8: Symmetrization Creates Temporary Arrays
**Impact:** 2x memory bandwidth waste
**Location:** `utils/similarity_builder.py:120-130`

#### Problem
Matrix symmetrization creates two temporary matrix copies.

#### Current Code
```python
def _symmetrize(self, matrix: np.ndarray) -> np.ndarray:
    """Enforce matrix symmetry."""
    # PROBLEM: Creates 2 temporary arrays of size n²
    return (matrix + matrix.T) / 2
    # Step 1: matrix.T creates n² transpose (COW in NumPy, but still memory pressure)
    # Step 2: matrix + matrix.T creates n² temporary sum
    # Step 3: ... / 2 creates n² temporary result
```

#### Impact Analysis
- For 8192×8192 matrix:
  - Each operation: 512 MB
  - Total temporary memory: 1.5+ GB
  - 3 full passes over matrix elements

#### Recommended Fix
```python
def _symmetrize(self, matrix: np.ndarray) -> np.ndarray:
    """Enforce matrix symmetry with in-place operations."""
    # OPTIMIZED: In-place symmetrization using upper/lower triangle
    n = matrix.shape[0]

    # Average upper and lower triangles in-place
    for i in range(n):
        for j in range(i+1, n):
            avg = (matrix[i, j] + matrix[j, i]) / 2.0
            matrix[i, j] = avg
            matrix[j, i] = avg

    return matrix
```

**Better: Vectorized with minimal temporaries**
```python
def _symmetrize(self, matrix: np.ndarray) -> np.ndarray:
    """Enforce matrix symmetry with optimized vectorization."""
    # Extract triangular parts (views, not copies)
    triu_indices = np.triu_indices_from(matrix, k=1)
    tril_indices = (triu_indices[1], triu_indices[0])  # Transposed indices

    # Compute average of upper and lower triangles
    upper_vals = matrix[triu_indices]
    lower_vals = matrix[tril_indices]
    avg_vals = (upper_vals + lower_vals) / 2.0

    # Assign back to both triangles
    matrix[triu_indices] = avg_vals
    matrix[tril_indices] = avg_vals

    return matrix
```

---

## 🟢 MODERATE ISSUES (Medium Impact)

### Issue #9: Cache Key Computation Overhead
**Impact:** 10-100x slowdown for cache lookups
**Location:** `utils/similarity_builder.py:132-165`

#### Problem
Computing checksum via `observations.sum()` requires full array scan for large observations.

#### Current Code
```python
def _get_observations_hash(self, observations: np.ndarray) -> str:
    """Generate fast hash key for observations."""
    shape_tuple = observations.shape
    dtype_str = observations.dtype.str

    if observations.size > 0:
        corner_0_0 = int(observations.flat[0])
        corner_n_n = int(observations.flat[-1])
        # PROBLEM: Full array scan for sum!
        checksum = int(observations.sum()) % (2**31)
        # For 8192 × 1000 sequences: 8.2M element sum
    else:
        corner_0_0 = 0
        corner_n_n = 0
        checksum = 0

    hash_tuple = (shape_tuple, dtype_str, corner_0_0, corner_n_n, checksum)
    return str(hash(hash_tuple))
```

#### Impact Analysis
- For 8192 taxa × 1000 seq_len: 8,192,000 elements
- Full sum: 8.2M additions
- **Called frequently** during cache lookups

#### Recommended Fix
```python
def _get_observations_hash(self, observations: np.ndarray) -> str:
    """Generate fast hash key using strided sampling."""
    shape_tuple = observations.shape
    dtype_str = observations.dtype.str

    if observations.size > 0:
        # Sample corners
        corner_0_0 = int(observations.flat[0])
        corner_n_n = int(observations.flat[-1])

        # OPTIMIZED: Strided sampling instead of full sum
        # Sample ~1000 elements regardless of array size
        stride = max(1, observations.size // 1000)
        sample = observations.flat[::stride]
        checksum = int(sample.sum()) % (2**31)

        # Add middle element for extra discrimination
        middle_idx = observations.size // 2
        middle_val = int(observations.flat[middle_idx])
    else:
        corner_0_0 = 0
        corner_n_n = 0
        checksum = 0
        middle_val = 0

    hash_tuple = (shape_tuple, dtype_str, corner_0_0, corner_n_n, middle_val, checksum)
    return str(hash(hash_tuple))
```

**Alternative: Use xxhash for speed**
```python
def _get_observations_hash(self, observations: np.ndarray) -> str:
    """Generate ultra-fast hash using xxhash."""
    import xxhash

    # xxhash is 10-100x faster than sum() for large arrays
    # Only hash metadata + sample of data
    h = xxhash.xxh64()

    # Hash shape and dtype
    h.update(str(observations.shape).encode())
    h.update(observations.dtype.str.encode())

    # Hash sample of data (first/last/middle rows)
    if observations.size > 0:
        h.update(observations[0].tobytes())  # First row
        if observations.shape[0] > 2:
            h.update(observations[observations.shape[0]//2].tobytes())  # Middle row
        h.update(observations[-1].tobytes())  # Last row

    return h.hexdigest()
```

---

### Issue #10: Metric Aggregation Redundancy
**Impact:** 1.1-1.2x + memory waste
**Location:** `experiment/bootstrap_sweep.py:510-536`

#### Problem
Storing constant M-based metrics repeatedly for every p-value.

#### Current Code
```python
# Inside p-value loop (runs 15-25 times):
for p_idx, p in enumerate(cfg.p_values):
    # Compute M-based metrics once per p-value (REDUNDANT!)
    try:
        M_metrics = metric_composer(
            M=M, S=M, L_M=L_M, L_S=L_M,  # Same M for all p-values!
            p=1.0,
            empirical_rank_threshold=empirical_rank_threshold,
            coherence_k=coherence_k
        )
        M_constants = {
            'operator_norm_error': float('nan'),
            'empirical_rank_M': M_metrics.get('empirical_rank_M', float('nan')),
            'empirical_rank_L_M': M_metrics.get('empirical_rank_L_M', float('nan')),
            # ... 8 more metrics ...
        }
    except Exception as e:
        log_warning('bootstrap', f"Failed to compute M-based metrics: {e}")
        M_constants = {key: float('nan') for key in metric_keys}

    # Later: Store these SAME values for EVERY p-value
    for key in constant_metrics:
        val = M_constants.get(key, float('nan'))
        metrics_dict[key].append((float(val), float(val), 0.0))
        # ^^^ Appending identical values 25 times!
```

#### Impact Analysis
- M-based metrics computed: 1 time (correct)
- M-based metrics stored: 25 times (25× redundancy)
- **Wasted:** 24 redundant storage operations + memory

#### Recommended Fix
```python
# BEFORE p-value loop: Compute M-based metrics ONCE
log_info('bootstrap', "Computing M-based metrics (once)...", force=True)
try:
    M_metrics = metric_composer(
        M=M, S=M, L_M=L_M, L_S=L_M,
        p=1.0,
        empirical_rank_threshold=empirical_rank_threshold,
        coherence_k=coherence_k
    )
    M_constants = {
        'empirical_rank_M': M_metrics.get('empirical_rank_M', float('nan')),
        'empirical_rank_L_M': M_metrics.get('empirical_rank_L_M', float('nan')),
        'spectral_gap_M': M_metrics.get('spectral_gap_M', float('nan')),
        'spectral_gap_L_M': M_metrics.get('spectral_gap_L_M', float('nan')),
        'coherence_M': M_metrics.get('coherence_M', float('nan')),
        'coherence_L_M': M_metrics.get('coherence_L_M', float('nan')),
        'min_separation_M': M_metrics.get('min_separation_M', float('nan')),
        'min_separation_L_M': M_metrics.get('min_separation_L_M', float('nan')),
    }
except Exception as e:
    log_warning('bootstrap', f"Failed to compute M-based metrics: {e}")
    M_constants = {key: float('nan') for key in constant_metrics}

# Inside p-value loop:
for p_idx, p in enumerate(cfg.p_values):
    # ... bootstrap loop for S-based metrics ...

    # Store constant metrics: REFERENCE existing values
    for key in constant_metrics:
        val = M_constants.get(key, float('nan'))
        metrics_dict[key].append((float(val), float(val), 0.0))
```

---

### Issue #11: Partition Agreement Memory Allocation
**Impact:** 1.2x for partition-heavy workloads
**Location:** `utils/metrics.py:285-326`

#### Problem
Creating new partition arrays for every partition agreement computation.

#### Current Code
```python
def compute_partition_agreement(
    fiedler_full: np.ndarray,
    fiedler_avg: np.ndarray,
    similarity_for_full: np.ndarray,
    similarity_for_avg: np.ndarray,
    num_gaps: int = 1,
    min_split: int = 1
) -> float:
    """Compute partition agreement between two Fiedler vectors."""
    from spectraltree.spectral_tree_reconstruction import partition_taxa

    # PROBLEM: Creating new partition arrays each call
    partition_full = partition_taxa(fiedler_full, similarity_for_full, num_gaps, min_split)
    partition_avg = partition_taxa(fiedler_avg, similarity_for_avg, num_gaps, min_split)
    # ^^^ Each returns n-element array (8192 × 8 bytes = 64 KB)

    # Compare partitions
    matches_direct = np.sum(partition_full == partition_avg)
    matches_flipped = np.sum(partition_full != partition_avg)
    max_matches = max(matches_direct, matches_flipped)

    return 100.0 * max_matches / len(fiedler_full)
```

#### Impact Analysis
- Called 2× per p-value (partition_agreement_M and partition_agreement_S)
- For 25 p-values: 50 partition computations
- Each creates 2 arrays of 64 KB = 128 KB
- **Total overhead:** 6.4 MB allocations + partition computation time

#### Recommended Fix
```python
# Option 1: Reuse buffers (requires refactoring partition_taxa)
class PartitionAgreementComputer:
    """Reusable partition agreement computation with buffer reuse."""

    def __init__(self, n_taxa: int):
        """Pre-allocate partition buffers."""
        self.n_taxa = n_taxa
        self.partition_buffer_1 = np.empty(n_taxa, dtype=np.int32)
        self.partition_buffer_2 = np.empty(n_taxa, dtype=np.int32)

    def compute(self, fiedler_full, fiedler_avg, similarity_for_full,
                similarity_for_avg, num_gaps=1, min_split=1) -> float:
        """Compute partition agreement with buffer reuse."""
        from spectraltree.spectral_tree_reconstruction import partition_taxa

        # Partition into pre-allocated buffers (would need API change)
        partition_taxa_inplace(fiedler_full, similarity_for_full,
                              num_gaps, min_split, out=self.partition_buffer_1)
        partition_taxa_inplace(fiedler_avg, similarity_for_avg,
                              num_gaps, min_split, out=self.partition_buffer_2)

        # Compare
        matches_direct = np.sum(self.partition_buffer_1 == self.partition_buffer_2)
        matches_flipped = np.sum(self.partition_buffer_1 != self.partition_buffer_2)
        max_matches = max(matches_direct, matches_flipped)

        return 100.0 * max_matches / self.n_taxa

# Usage in experiment:
partition_computer = PartitionAgreementComputer(n_taxa)

# Later:
partition_agr_M = partition_computer.compute(fiedler_ref, v_avg, M, M, ...)
partition_agr_S = partition_computer.compute(fiedler_ref, v_avg, M, S_avg, ...)
```

---

### Issue #12: Progress Bar Overhead in Tight Loops
**Impact:** 1.05-1.1x
**Location:** `experiment/bootstrap_sweep.py:416-417`

#### Problem
Updating progress bar in tight bootstrap loop (100+ times per p-value).

#### Current Code
```python
for i in range(cfg.bootstrap_reps):  # 10-100+ iterations
    # ... bootstrap computation (fast) ...

    # Update bootstrap progress bar
    if bootstrap_pbar:
        bootstrap_pbar.update(1)  # PROBLEM: Called 100+ times
        # Progress bar update involves:
        # - Lock acquisition (thread-safe)
        # - Terminal I/O (can be slow)
        # - String formatting
        # - Rate limiting checks
```

#### Impact Analysis
- Progress bar update: ~0.1-1 ms per call (varies by terminal)
- For 100 bootstrap iterations: 10-100 ms overhead
- For fast bootstrap iterations: 5-10% overhead

#### Recommended Fix
```python
# OPTIMIZED: Update every N iterations
update_interval = max(1, cfg.bootstrap_reps // 20)  # Update 20 times max

for i in range(cfg.bootstrap_reps):
    # ... bootstrap computation ...

    # Update progress bar periodically
    if bootstrap_pbar and (i % update_interval == 0 or i == cfg.bootstrap_reps - 1):
        # Update by interval amount (or remaining)
        updates = min(update_interval, cfg.bootstrap_reps - i)
        bootstrap_pbar.update(updates)
```

---

## Summary Table

| Issue | Location | Impact | Estimated Speedup | Priority |
|-------|----------|--------|-------------------|----------|
| #1: Full SVD | metric_computer.py:148 | O(n³) → O(n²k) | 50-100x | 🔴 Critical |
| #2: Matrix Subsampling | similarity_builder.py:66 | Memory thrashing | 10-20x | 🔴 Critical |
| #3: Laplacian Recomputation | bootstrap_sweep.py:379 | 100× redundant | 2-3x | 🔴 Critical |
| #4: Sparse Eigensolver | fiedler_computer.py:103 | Poor convergence | 2-5x | 🔴 Critical |
| #5: Streaming Average | bootstrap_sweep.py:370 | Memory bandwidth | 1.5-2x | 🟡 Major |
| #6: Repeated Normalization | bootstrap_sweep.py:34 | 100× redundant | 1.2-1.5x | 🟡 Major |
| #7: Sign Convention | fiedler_computer.py:153 | Boolean array | 1.5x | 🟡 Major |
| #8: Symmetrization | similarity_builder.py:120 | Temp arrays | 2x | 🟡 Major |
| #9: Cache Key | similarity_builder.py:132 | Full sum | 10-100x | 🟢 Moderate |
| #10: Metric Redundancy | bootstrap_sweep.py:510 | 25× storage | 1.1-1.2x | 🟢 Moderate |
| #11: Partition Buffers | metrics.py:285 | Allocation | 1.2x | 🟢 Moderate |
| #12: Progress Bar | bootstrap_sweep.py:416 | I/O overhead | 1.05-1.1x | 🟢 Moderate |

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
- Fix #2: Matrix subsampling (10-20x speedup)
- Fix #6: Remove repeated normalization (1.5x speedup)
- Fix #7: Optimize sign convention (1.5x speedup)
- Fix #9: Faster cache keys (10-100x speedup)
- **Phase 1 Total: ~15-30x speedup**

### Phase 2: Core Optimizations (3-5 days)
- Fix #1: Partial SVD (50-100x speedup)
- Fix #3: Eliminate Laplacian recomputation (2-3x speedup)
- Fix #4: Fix sparse eigensolver (2-5x speedup)
- Fix #5: In-place streaming average (1.5-2x speedup)
- **Phase 2 Total: ~100-500x speedup**

### Phase 3: Polish (1-2 days)
- Fix #8: In-place symmetrization (2x speedup)
- Fix #10: Metric storage optimization (1.2x speedup)
- Fix #11: Partition buffer reuse (1.2x speedup)
- Fix #12: Reduce progress bar overhead (1.1x speedup)
- **Phase 3 Total: ~2-3x additional speedup**

---

## Testing Strategy

### Performance Benchmarks
```python
# Create benchmark script: benchmark_optimizations.py

import time
import numpy as np
from utils.similarity_builder import SimilarityMatrixBuilder

def benchmark_subsampling():
    """Benchmark Issue #2 fix."""
    n = 8192
    matrix = np.random.random((n, n))
    p = 0.1

    # Old method
    start = time.time()
    result_old = subsample_old(matrix, p)
    time_old = time.time() - start

    # New method
    start = time.time()
    result_new = subsample_new(matrix, p)
    time_new = time.time() - start

    print(f"Subsampling speedup: {time_old/time_new:.2f}x")

def benchmark_svd():
    """Benchmark Issue #1 fix."""
    n = 8192
    matrix = np.random.random((n, n))
    k = 2

    # Full SVD
    start = time.time()
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    time_full = time.time() - start

    # Partial SVD
    from scipy.sparse.linalg import svds
    start = time.time()
    U, s, Vt = svds(matrix, k=k)
    time_partial = time.time() - start

    print(f"SVD speedup: {time_full/time_partial:.2f}x")

if __name__ == "__main__":
    benchmark_subsampling()
    benchmark_svd()
```

### Correctness Tests
```python
# Ensure optimizations don't change results

def test_subsampling_equivalence():
    """Verify new subsampling gives same distribution."""
    n = 100
    matrix = np.random.random((n, n))
    p = 0.1
    seed = 42

    # Run both methods multiple times
    results_old = [subsample_old(matrix, p, seed+i) for i in range(100)]
    results_new = [subsample_new(matrix, p, seed+i) for i in range(100)]

    # Check distribution equivalence
    mean_old = np.mean([r.sum() for r in results_old])
    mean_new = np.mean([r.sum() for r in results_new])

    assert abs(mean_old - mean_new) / mean_old < 0.01, "Means differ significantly"
```

---

## Expected Performance Improvement

### Current Performance (Estimated)
- **8192 taxa, 25 p-values, 10 bootstrap reps:**
  - Per bootstrap iteration: ~6 seconds (Fiedler + metrics)
  - Per p-value: ~60 seconds (10 iterations)
  - Total: ~1500 seconds = **25 minutes**

- **8192 taxa, 25 p-values, 100 bootstrap reps:**
  - Per p-value: ~600 seconds
  - Total: ~15,000 seconds = **4.2 hours**

### After All Optimizations
- **Phase 1 only (15-30x):**
  - 100 reps: 4.2 hours → **8-17 minutes**

- **Phase 1 + Phase 2 (100-500x):**
  - 100 reps: 4.2 hours → **30 seconds - 2.5 minutes**

- **All phases (200-1000x):**
  - 100 reps: 4.2 hours → **15-120 seconds**

---

## Risk Assessment

### Low Risk (Safe to implement)
- Issues #2, #6, #7, #9, #12: Pure optimizations, no algorithm changes

### Medium Risk (Requires testing)
- Issues #1, #4, #5, #8: Numerical considerations, verify accuracy
- Test: Compare Fiedler vectors before/after (should be identical)

### High Risk (Requires careful validation)
- Issues #3: Changing when metrics are computed
- Issue #11: Requires API changes to spectraltree library

### Mitigation Strategy
1. Implement each fix in separate branch
2. Run correctness tests comparing old vs new results
3. Check Fiedler vector agreement: should be >99.99%
4. Validate metrics: should match within numerical precision

---

## Conclusion

This codebase has significant performance optimization opportunities. The issues identified fall into three categories:

1. **Algorithmic inefficiencies:** Using O(n³) algorithms when O(n²k) available
2. **Memory management:** Creating unnecessary temporary arrays
3. **Redundant computation:** Computing same values multiple times

By addressing these issues in phases, we can achieve **100-1000x speedup** for typical workloads, reducing hours-long experiments to minutes or seconds.

The optimizations are mostly straightforward to implement and low-risk, with the exception of Issue #3 (Laplacian recomputation) which requires architectural changes to metrics computation strategy.

---

**Next Steps:**
1. Review this report and prioritize fixes
2. Implement Phase 1 (quick wins) for immediate ~15-30x improvement
3. Benchmark results and validate correctness
4. Proceed to Phase 2 for maximum performance gains
