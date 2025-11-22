# Plan: Partition-Based Metrics for STDR Fiedler Vector Quality Assessment

## Executive Summary

**Current metric**: Sign agreement at threshold=0 (simple sign comparison)

**Problem**: Doesn't reflect how STDR actually partitions data (uses gap-based thresholding + SVD quality scoring)

**Solution**:
1. **Primary metrics**: Use `partition_taxa` to compute 3 partitions, then measure 2 agreement scores
2. **Secondary metric**: Compute dot product u·v between normalized Fiedler vectors (alignment score)

**Key insight**: We want to know if preprocessing (subsampling + averaging Fiedler vectors) produces equivalent results to using full data directly in STDR.

---

## Scientific Question

**What we're testing**: Does averaged Fiedler vector from subsampled data produce the same partition as the full Fiedler vector when both are evaluated using STDR's actual partitioning algorithm?

**Context**:
- M = similarity matrix from observed sequences (the "data" STDR sees)
- True tree = hidden, unknowable
- STDR's job = recover tree structure from M
- Our preprocessing = subsample M → compute multiple Fiedler vectors → average them

**Two evaluation scenarios**:
1. **Ideal**: Both reference and averaged Fiedler use M for partitioning (tests vector quality in isolation)
2. **Realistic**: Reference uses M, averaged Fiedler uses S_avg (reflects real-world usage where we only have subsampled data)

---

## Metrics Design

### 3 Partitions Computed

For each p-value, we compute **3 partitions total**:

1. **Reference partition** (baseline):
   ```python
   partition_ref = partition_taxa(fiedler_full, M, num_gaps, min_split)
   ```
   - Input: Fiedler vector from full Laplacian L(M)
   - Scoring matrix: M (full similarity)
   - Represents: Best possible partition from complete data

2. **Test partition vs M** (ideal scenario):
   ```python
   partition_avg_vs_M = partition_taxa(fiedler_avg, M, num_gaps, min_split)
   ```
   - Input: Averaged Fiedler vector from subsampled bootstraps
   - Scoring matrix: M (full similarity)
   - Represents: How well does averaged Fiedler work with clean STDR?

3. **Test partition vs S_avg** (realistic scenario):
   ```python
   partition_avg_vs_S = partition_taxa(fiedler_avg, S_avg, num_gaps, min_split)
   ```
   - Input: Averaged Fiedler vector from subsampled bootstraps
   - Scoring matrix: S_avg (averaged subsampled similarity)
   - Represents: Real-world performance where we only have subsampled data

### 2 Agreement Scores

1. **`partition_agreement_M`**: Compare partition 1 vs partition 2
   - Measures: Vector quality in isolation (both use M)
   - Range: 0-100%
   - Interpretation: "If we had full M, would averaged Fiedler give same partition as full Fiedler?"

2. **`partition_agreement_S`**: Compare partition 1 vs partition 3
   - Measures: Real-world performance (reference uses M, test uses S_avg)
   - Range: 0-100%
   - Interpretation: "In realistic scenario, how close do we get to the ideal partition?"

### Additional Metrics

3. **`dot_product`**: Alignment between normalized Fiedler vectors
   - Range: 0-1
   - Cheap continuous measure of vector similarity

4. **`sign_agreement`**: Legacy metric (kept for comparison)
   - Range: 0-100%
   - Simple threshold-at-zero comparison

---

## Implementation Plan

### Phase 1: Metric Functions and Bootstrap Logic

#### Part A: Helper Functions (`utils/metrics.py`)

**Add to end of file**:

```python
def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit length.

    Args:
        v: Input vector

    Returns:
        Normalized vector

    Raises:
        ValueError: If vector has zero or near-zero norm
    """
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError(f"Vector has zero or near-zero norm: {norm}")
    return v / norm


def compute_partition_agreement(
    fiedler_full: np.ndarray,
    fiedler_avg: np.ndarray,
    similarity_for_full: np.ndarray,
    similarity_for_avg: np.ndarray,
    num_gaps: int = 1,
    min_split: int = 1
) -> float:
    """
    Compute partition agreement between two Fiedler vectors.

    This function is used TWICE to compute the two agreement metrics:
    1. partition_agreement_M = compute_partition_agreement(v_full, v_avg, M, M)
    2. partition_agreement_S = compute_partition_agreement(v_full, v_avg, M, S_avg)

    Args:
        fiedler_full: Reference Fiedler vector from L(M)
        fiedler_avg: Averaged Fiedler vector from subsampled bootstraps
        similarity_for_full: Similarity matrix to score reference partition (typically M)
        similarity_for_avg: Similarity matrix to score test partition (M or S_avg)
        num_gaps: Number of gap-based thresholds to evaluate
        min_split: Minimum partition size

    Returns:
        Percentage of taxa in matching partition (0-100)

    Raises:
        Exception: From partition_taxa if partitioning fails (propagates to caller)
    """
    from spectraltree.spectral_tree_reconstruction import partition_taxa

    # Compute two partitions
    partition_full = partition_taxa(fiedler_full, similarity_for_full, num_gaps, min_split)
    partition_avg = partition_taxa(fiedler_avg, similarity_for_avg, num_gaps, min_split)

    # Compare (handle both orientations: A|B ≡ B|A)
    matches_direct = np.sum(partition_full == partition_avg)
    matches_flipped = np.sum(partition_full != partition_avg)
    max_matches = max(matches_direct, matches_flipped)

    return 100.0 * max_matches / len(fiedler_full)


def compute_fiedler_dot_product(
    fiedler_full: np.ndarray,
    fiedler_avg: np.ndarray
) -> float:
    """
    Compute dot product between normalized Fiedler vectors.

    Args:
        fiedler_full: Reference Fiedler vector
        fiedler_avg: Averaged Fiedler vector

    Returns:
        Absolute dot product (0-1, higher = better alignment)

    Raises:
        ValueError: If either vector has zero norm
    """
    u_norm = _normalize_vector(fiedler_full)
    v_norm = _normalize_vector(fiedler_avg)
    return float(np.abs(np.dot(u_norm, v_norm)))
```

**Design rationale**:
- ✅ Single `compute_partition_agreement` function handles both comparisons
- ✅ Reuses existing `partition_taxa` from spectraltree (no reimplementation)
- ✅ Handles partition orientation (A|B ≡ B|A)
- ✅ Shared normalization helper eliminates duplication
- ✅ Exceptions propagate to caller for proper error handling

---

#### Part B: Update Alignment Function (`experiment/bootstrap_sweep.py`)

**Modify `align_fiedler_by_dot_product`** (lines 16-54):

```python
def align_fiedler_by_dot_product(fiedler_vector: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
    """
    Align a Fiedler vector using magnitude-based dot product alignment.

    Algorithm:
    1. Normalize both vectors
    2. Compute dot product
    3. If dot product is negative, flip the sign

    Args:
        fiedler_vector: Fiedler vector to align
        reference_vector: Reference vector for alignment

    Returns:
        Aligned and normalized Fiedler vector
    """
    from utils.metrics import _normalize_vector

    try:
        v_normalized = _normalize_vector(fiedler_vector)
        u_normalized = _normalize_vector(reference_vector)
    except ValueError as e:
        log_warning('align', f"Normalization failed: {e}")
        return fiedler_vector

    # Compute dot product
    dot_product = np.dot(v_normalized, u_normalized)

    # Flip sign if needed
    if dot_product < 0:
        return -v_normalized
    else:
        return v_normalized
```

---

#### Part C: Update Imports (`experiment/bootstrap_sweep.py`)

**Change line 9**:
```python
from utils.metrics import (
    compute_sign_agreement,         # Keep for comparison
    compute_partition_agreement,    # NEW
    compute_fiedler_dot_product,    # NEW
    metric_composer
)
```

---

#### Part D: Initialize Storage (`experiment/bootstrap_sweep.py`)

**Change line 112**:
```python
sign_agreements: List[float] = []            # Legacy metric (kept for comparison)
partition_agreement_M: List[float] = []      # NEW: ideal scenario (both use M)
partition_agreement_S: List[float] = []      # NEW: realistic scenario (M vs S_avg)
dot_products: List[float] = []               # NEW: vector alignment
```

---

#### Part E: Bootstrap Loop - Streaming S Average (`experiment/bootstrap_sweep.py`)

**Key optimization**: Use Welford's online algorithm to compute running average of S matrices without storing them all in memory.

**Add before bootstrap loop** (line ~200):
```python
aligned_vectors = []
S_avg = None                # Running average of S matrices
n_bootstrap_collected = 0   # Counter for streaming average
```

**Update bootstrap loop** (lines ~213-265):
```python
for i in range(cfg.bootstrap_reps):
    bootstrap_seed = cfg.seed + i

    # Compute S once per bootstrap rep
    S = _subsample_matrix_entries(M, p, seed=bootstrap_seed)

    # Update running average of S (streaming - no storage!)
    # Uses Welford's online algorithm for numerical stability
    if S_avg is None:
        S_avg = S.copy()
        n_bootstrap_collected = 1
    else:
        n_bootstrap_collected += 1
        S_avg += (S - S_avg) / n_bootstrap_collected

    # Compute L_S for metrics
    try:
        L_S = compute_laplacian(S)
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
        except Exception as e:
            log_warning('bootstrap', f"Failed to compute metrics: {e}")
            all_metrics = None
    else:
        all_metrics = None

    # Store metrics (existing code)
    if all_metrics is None:
        all_metrics = {key: float('nan') for key in metric_keys}

    for key in metric_keys:
        metric_values[key].append(all_metrics.get(key, float('nan')))

    # CHANGED: Compute Fiedler from S directly (using new function from Phase 3)
    # No longer calls fiedler_method(observations, p, seed) which would recompute S
    f_est = cfg.fiedler_method(S)

    # Align using dot product and normalize
    f_aligned_normalized = align_fiedler_by_dot_product(f_est, fiedler_ref)
    aligned_vectors.append(f_aligned_normalized)

    # Update bootstrap progress bar
    if bootstrap_pbar:
        bootstrap_pbar.update(1)
```

**Why streaming average?**
- Memory: Avoids storing K bootstrap reps × n² matrix entries
- For n=4000, K=100: saves ~12.8 GB memory
- Welford's algorithm is numerically stable

---

#### Part F: Compute Partition Metrics After Bootstrap Loop (`experiment/bootstrap_sweep.py`)

**Replace lines ~266-288**:

```python
# After bootstrap loop: Average the aligned vectors
if len(aligned_vectors) > 0 and S_avg is not None:
    v_avg = np.mean(aligned_vectors, axis=0)

    # Normalize the averaged vector
    from utils.metrics import _normalize_vector
    try:
        v_avg = _normalize_vector(v_avg)
    except ValueError as e:
        log_warning('bootstrap', f"Averaged vector normalization failed: {e}")
        # Fallback: zero vector (will produce 0% agreement)
        v_avg = np.zeros_like(v_avg)

    # OLD metric (keep for comparison)
    sign_agreement = compute_sign_agreement(fiedler_ref, v_avg)

    # NEW METRIC 1: partition_agreement_M
    # Tests: How well does averaged Fiedler perform with clean STDR?
    # Computes 2 partitions:
    #   - partition_taxa(fiedler_full, M)  [reference]
    #   - partition_taxa(fiedler_avg, M)   [test vs M]
    # Then compares them
    try:
        partition_agr_M = compute_partition_agreement(
            fiedler_ref, v_avg, M, M,  # Both use M
            num_gaps=cfg.num_gaps,
            min_split=cfg.min_split
        )
    except Exception as e:
        log_warning('bootstrap', f"partition_agreement (M) failed: {e}")
        partition_agr_M = float('nan')

    # NEW METRIC 2: partition_agreement_S
    # Tests: Realistic scenario where test uses averaged subsampled data
    # Computes 2 partitions:
    #   - partition_taxa(fiedler_full, M)      [reference - same as above]
    #   - partition_taxa(fiedler_avg, S_avg)   [test vs S_avg]
    # Then compares them
    try:
        partition_agr_S = compute_partition_agreement(
            fiedler_ref, v_avg, M, S_avg,  # Reference uses M, test uses S_avg
            num_gaps=cfg.num_gaps,
            min_split=cfg.min_split
        )
    except Exception as e:
        log_warning('bootstrap', f"partition_agreement (S_avg) failed: {e}")
        partition_agr_S = float('nan')

    # NEW METRIC 3: Vector alignment
    try:
        dot_prod = compute_fiedler_dot_product(fiedler_ref, v_avg)
    except Exception as e:
        log_warning('bootstrap', f"dot_product failed: {e}")
        dot_prod = float('nan')
else:
    log_warning('bootstrap', f"No valid aligned vectors for p={p:.4g}")
    sign_agreement = 0.0
    partition_agr_M = 0.0
    partition_agr_S = 0.0
    dot_prod = 0.0

# Close bootstrap progress bar
if bootstrap_pbar:
    bootstrap_pbar.close()

# Store all metrics
sign_agreements.append(float(sign_agreement))
partition_agreement_M.append(float(partition_agr_M))
partition_agreement_S.append(float(partition_agr_S))
dot_products.append(float(dot_prod))
```

**Summary of 3 partitions computed**:
1. `partition_taxa(fiedler_full, M)` - computed in both metric 1 and metric 2
2. `partition_taxa(fiedler_avg, M)` - computed in metric 1
3. `partition_taxa(fiedler_avg, S_avg)` - computed in metric 2

**Note**: Partition 1 is computed twice (once per `compute_partition_agreement` call). This is acceptable since the function is called only once per p-value (not per bootstrap rep).

---

### Phase 2: Config Parameters

**File**: `utils/experiment_config.py`

**Add fields to Config dataclass** (after existing fields):
```python
@dataclass
class Config:
    # ... existing fields ...

    # Partition algorithm parameters (align with STDR defaults from spectral_tree_reconstruction.py:40)
    num_gaps: int = 1      # Number of gap-based thresholds to evaluate
    min_split: int = 1     # Minimum partition size
```

---

### Phase 3: New Fiedler Computer Function

**File**: `utils/random_entries.py`

**Add new function** (after existing `compute_fiedler_estimate`):
```python
def compute_fiedler_from_similarity(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Fiedler vector from pre-computed similarity matrix.

    This is a strict version that requires S to be provided (no fallback).
    Use this when you've already computed the similarity matrix and want
    to avoid recomputation.

    Args:
        similarity_matrix: Pre-computed similarity matrix S (n x n)

    Returns:
        Fiedler vector (n-dimensional)

    Raises:
        ValueError: If similarity_matrix is None

    Example:
        S = _subsample_matrix_entries(M, p=0.5, seed=42)
        fiedler = compute_fiedler_from_similarity(S)
    """
    if similarity_matrix is None:
        raise ValueError("similarity_matrix must be provided (cannot be None)")

    return compute_fielder_vector(similarity_matrix)
```

**Update Config to use new function** (`utils/experiment_config.py`):

**Change import** (line 12):
```python
from utils.random_entries import compute_fiedler_from_similarity
```

**Change default** (line 32):
```python
@dataclass
class Config:
    # ... existing fields ...

    # CHANGED: Use new strict function that requires S
    fiedler_method: Callable = compute_fiedler_from_similarity
```

**Rationale**:
- New function enforces passing S directly (fail-fast if not provided)
- Avoids accidental S recomputation in bootstrap loop
- Old `compute_fiedler_estimate` still exists for backward compatibility

---

### Phase 4: Update Return Values and Callers

#### Part A: Update `sweep_for_params` Signature and Return

**File**: `experiment/bootstrap_sweep.py`

**Change function signature** (line 68):
```python
def sweep_for_params(
    cfg: Config,
    n_taxa: int,
    seq_len: int,
    run_dir: str,
    incremental_save: bool = False,
    show_progress: bool = True
) -> Tuple[np.ndarray, List[float], List[float], List[float], List[float], Dict[str, List[Tuple[float, float, float]]]]:
    """
    Run sweep for a specific (n_taxa, seq_len) combination.

    Returns:
        Tuple of (fiedler_ref, sign_agreements, partition_agreement_M,
                  partition_agreement_S, dot_products, metrics_dict)
        - fiedler_ref: Reference Fiedler vector from full matrix (p=1.0)
        - sign_agreements: Legacy sign agreement metric (kept for comparison)
        - partition_agreement_M: Agreement using M for both partitions (ideal)
        - partition_agreement_S: Agreement using M vs S_avg (realistic)
        - dot_products: Vector alignment metric (0-1)
        - metrics_dict: Aggregated metrics (mean, median, std) for each metric type
    """
```

**Change return statement** (line ~422):
```python
return fiedler_ref, sign_agreements, partition_agreement_M, partition_agreement_S, dot_products, metrics_dict
```

---

#### Part B: Update Callers in `experiment_runner.py`

**3 locations that call `sweep_for_params`**:

**Change 1** (line 64 - single experiment):
```python
fiedler_ref, sign_agreements, partition_agreement_M, partition_agreement_S, dot_products, metrics_dict = sweep_for_params(
    cfg=self.cfg,
    n_taxa=self.cfg.num_taxa,
    seq_len=self.cfg.sequence_length,
    run_dir=self.run_dir,
    incremental_save=False,
    show_progress=True
)
```

**Change 2** (line 123 - taxa sweep):
```python
fiedler_ref, sig, partition_agr_M, partition_agr_S, dot_prod, metrics_dict = sweep_for_params(
    cfg=self.cfg,
    n_taxa=n_taxa,
    seq_len=self.cfg.sequence_length,
    run_dir=self.run_dir,
    incremental_save=False,
    show_progress=True
)
```

**Change 3** (line 200 - grid sweep):
```python
fiedler_ref, sig, partition_agr_M, partition_agr_S, dot_prod, metrics_dict = sweep_for_params(
    cfg=self.cfg,
    n_taxa=n_taxa,
    seq_len=seq_len,
    run_dir=self.run_dir,
    incremental_save=False,
    show_progress=True
)
```

---

#### Part C: Update `save_single_results` Function

**File**: `utils/summaries.py`

**Change function signature** (line 36):
```python
def save_single_results(
    *,
    run_dir: str,
    p_values: List[float],
    sign_agreements: List[float] | List[Tuple[float, float, float]] | List[Tuple[float, float]],
    partition_agreement_M: List[float] | None = None,
    partition_agreement_S: List[float] | None = None,
    dot_products: List[float] | None = None,
    metrics_dict: Dict[str, List[Tuple[float, float, float]]] | None = None
):
    """
    Save single-parameter results with new partition metrics.

    Args:
        run_dir: Directory to save results
        p_values: List of p-values
        sign_agreements: Legacy sign agreement metric (backward compatibility)
        partition_agreement_M: NEW - Agreement using M for both partitions
        partition_agreement_S: NEW - Agreement using M vs S_avg
        dot_products: NEW - Vector alignment metric
        metrics_dict: Matrix metrics (optional)
    """
```

**Update JSON construction** (lines 48-78):
```python
if sign_agreements and isinstance(sign_agreements[0], (int, float)):
    # New format with additional metrics
    columns = ["p", "sign_agreement"]
    rows = [{"p": float(p), "sign_agreement": float(agreement)}
            for p, agreement in zip(p_values, sign_agreements)]

    # Add new partition metrics if provided
    if partition_agreement_M is not None:
        columns.append("partition_agreement_M")
        for i, row in enumerate(rows):
            row["partition_agreement_M"] = float(partition_agreement_M[i])

    if partition_agreement_S is not None:
        columns.append("partition_agreement_S")
        for i, row in enumerate(rows):
            row["partition_agreement_S"] = float(partition_agreement_S[i])

    if dot_products is not None:
        columns.append("dot_product")
        for i, row in enumerate(rows):
            row["dot_product"] = float(dot_products[i])

elif sign_agreements and len(sign_agreements[0]) == 2:
    # Old format: (median, std) - no changes needed
    columns = ["p", "median", "std"]
    rows = [{"p": float(p), "median": float(m), "std": float(s)}
            for p, (m, s) in zip(p_values, sign_agreements)]
else:
    # Legacy format: (mean, median, std) - no changes needed
    columns = ["p", "mean", "median", "std"]
    rows = [{"p": float(p), "mean": float(mu), "median": float(m), "std": float(s)}
            for p, (mu, m, s) in zip(p_values, sign_agreements)]

# Add metrics if provided (existing code continues...)
```

---

#### Part D: Update Caller of `save_single_results`

**File**: `experiment/experiment_runner.py`

**Change** (line 74):
```python
save_single_results(
    run_dir=self.run_dir,
    p_values=self.cfg.p_values,
    sign_agreements=sign_agreements,
    partition_agreement_M=partition_agreement_M,  # NEW
    partition_agreement_S=partition_agreement_S,  # NEW
    dot_products=dot_products,                    # NEW
    metrics_dict=metrics_dict
)
```

---

#### Part E: Update Incremental Save Logic

**File**: `experiment/bootstrap_sweep.py`

**Update incremental save** (lines ~387-416):
```python
if incremental_save:
    # Save current progress
    current_p_values = cfg.p_values[:p_idx + 1]
    current_sign_agreements = sign_agreements
    current_partition_agreement_M = partition_agreement_M  # NEW
    current_partition_agreement_S = partition_agreement_S  # NEW
    current_dot_products = dot_products                    # NEW

    # Create current metrics dict with only completed p-values
    current_metrics_dict = {
        key: values[:p_idx + 1]
        for key, values in metrics_dict.items()
    }

    # Save as both single and taxa format for compatibility
    save_single_results(
        run_dir=run_dir,
        p_values=current_p_values,
        sign_agreements=current_sign_agreements,
        partition_agreement_M=current_partition_agreement_M,  # NEW
        partition_agreement_S=current_partition_agreement_S,  # NEW
        dot_products=current_dot_products,                    # NEW
        metrics_dict=current_metrics_dict
    )

    # Also save individual p-value results
    p_result = {
        "p": float(p),
        "sign_agreement": float(agreement),
        "partition_agreement_M": float(partition_agr_M),  # NEW
        "partition_agreement_S": float(partition_agr_S),  # NEW
        "dot_product": float(dot_prod)                    # NEW
    }
    save_json(
        p_result,
        os.path.join(run_dir, f"p_{p:.0e}_n_{n_taxa}_L={seq_len}.json")
    )

    log_info('bootstrap', f"Saved incremental results for p={p:.4g}")
```

---

#### Part F: Update Guardrails Logic

**File**: `experiment/bootstrap_sweep.py`

**Update guardrails fill logic** (lines ~342-376):
```python
# Fill remaining p-values with perfect agreement values
for remaining_p_idx in range(p_idx + 1, len(cfg.p_values)):
    remaining_p = cfg.p_values[remaining_p_idx]

    # Fill all agreement metrics with best values
    sign_agreements.append(100.0)
    partition_agreement_M.append(100.0)  # NEW
    partition_agreement_S.append(100.0)  # NEW
    dot_products.append(1.0)             # NEW - perfect alignment

    # Handle metrics based on flag
    if cfg.compute_metrics_on_guardrails:
        # Compute metrics for p=1.0 case (S=M, L_S=L_M)
        for key in metric_keys:
            if key == 'operator_norm_error':
                metrics_dict[key].append((0.0, 0.0, 0.0))
            elif key.endswith('_L_S'):
                l_m_key = key.replace('_L_S', '_L_M')
                val = M_constants.get(l_m_key, float('nan'))
                metrics_dict[key].append((float(val), float(val), 0.0))
            elif key.endswith('_S'):
                m_key = key.replace('_S', '_M')
                val = M_constants.get(m_key, float('nan'))
                metrics_dict[key].append((float(val), float(val), 0.0))
            else:
                val = M_constants.get(key, float('nan'))
                metrics_dict[key].append((float(val), float(val), 0.0))
    else:
        # Fill with constant M metrics for M/L_M, NaN for varying metrics
        for key in constant_metrics:
            val = M_constants.get(key, float('nan'))
            metrics_dict[key].append((float(val), float(val), 0.0))
        for key in varying_metrics:
            metrics_dict[key].append((float('nan'), float('nan'), float('nan')))

    # Update progress bar for skipped p-values
    if p_pbar:
        p_pbar.set_description(f"  p={remaining_p:.4g} [guardrails]")
        p_pbar.set_postfix({'sign_agreement': '100.0%'})
        p_pbar.update(1)
```

---

## Summary of Changes

| File | Lines Changed | Description |
|------|--------------|-------------|
| `utils/metrics.py` | +60 | Add `_normalize_vector`, `compute_partition_agreement`, `compute_fiedler_dot_product` |
| `utils/random_entries.py` | +25 | Add `compute_fiedler_from_similarity` (strict version) |
| `utils/experiment_config.py` | +5 | Add `num_gaps`, `min_split`, change fiedler_method import |
| `experiment/bootstrap_sweep.py` | ~80 | Update imports, storage, loop (streaming S), metrics computation, return value, incremental save, guardrails |
| `experiment/experiment_runner.py` | ~15 | Update 3 callers of `sweep_for_params`, update `save_single_results` call |
| `utils/summaries.py` | ~25 | Add parameters and JSON fields for new metrics |

**Total**: ~210 lines changed across 6 files

---

## Output Format

### JSON Fields

**New results.json structure**:
```json
{
  "columns": ["p", "sign_agreement", "partition_agreement_M", "partition_agreement_S", "dot_product", ...],
  "rows": [
    {
      "p": 1.0,
      "sign_agreement": 100.0,
      "partition_agreement_M": 100.0,
      "partition_agreement_S": 100.0,
      "dot_product": 1.0,
      ...
    },
    ...
  ]
}
```

**Metric interpretations**:
- `sign_agreement`: 0-100% (legacy)
- `partition_agreement_M`: 0-100% (ideal scenario: both use M)
- `partition_agreement_S`: 0-100% (realistic scenario: M vs S_avg)
- `dot_product`: 0-1 (vector alignment)

---

## Testing Strategy

### Validation Steps

1. **Sanity check**: For p=1.0, all metrics should be perfect:
   - `partition_agreement_M = 100%`
   - `partition_agreement_S = 100%`
   - `dot_product = 1.0`

2. **Comparison test**: Run small experiment (n=1024, L=500, K=10 reps)
   - Compare `partition_agreement_M` vs `sign_agreement`
   - Verify both are high for large p

3. **Divergence test**: Check if partition agreement differs from sign agreement for low p
   - Expected: More divergence at low p where gap-based thresholding matters

4. **Realistic vs ideal**: Compare `partition_agreement_M` vs `partition_agreement_S`
   - Expected: `partition_agreement_S` ≤ `partition_agreement_M` (realistic is harder)

### Expected Outcomes

- **High p (> 0.5)**:
  - `partition_agreement_M` ≈ `sign_agreement` ± 1%
  - `partition_agreement_S` ≈ `partition_agreement_M` ± 2%

- **Medium p (0.1-0.5)**:
  - Metrics may diverge by 2-5%
  - Gap-based thresholding effects become visible

- **Low p (< 0.1)**:
  - Significant divergence possible (5-15%)
  - `partition_agreement_S` may be notably lower than `partition_agreement_M`

---

## Key Design Decisions

### 1. Why compute 3 partitions?

We need to answer two questions:
1. **Vector quality**: Does averaged Fiedler give same partition as full Fiedler when both use ideal data (M)?
2. **Realistic performance**: How does the system perform when we only have subsampled data (S_avg)?

This requires 3 partitions and 2 comparisons.

### 2. Why streaming S average (Welford's algorithm)?

**Memory efficiency**: For n=4000, K=100 bootstrap reps:
- Storing all S: ~100 × 4000² × 8 bytes = 12.8 GB
- Streaming: Only stores one S_avg matrix = 128 MB

**Numerical stability**: Welford's algorithm is more stable than naive averaging.

**Bonus**: Can extend to compute variance of S using same algorithm if needed.

### 3. Why keep sign_agreement?

**Backward compatibility**: Allows comparison with previous results.

**Baseline**: Shows how much improvement partition-based metrics provide.

**Simplicity**: Cheap metric for quick sanity checks.

### 4. Why strict `compute_fiedler_from_similarity`?

**Fail-fast**: Catches bugs where S isn't passed correctly.

**Performance**: Prevents accidental S recomputation in tight loops.

**Clarity**: Function name and signature make intent explicit.

---

## Computational Cost Analysis

### Per p-value Cost

**Old approach** (sign agreement):
- O(n) for sign comparison
- Called once per p-value

**New approach** (partition agreement):
- O(n log n) for sorting (in `partition_taxa`)
- O(k × SVD(S_AB)) where k = `num_gaps`
  - For k=1 (default): 1-2 SVD computations
  - SVD cost: O(min(|A|, |B|)²)
- Called once per p-value (not per bootstrap!)

**Impact**:
- Negligible for small n (< 1000 taxa)
- Moderate for medium n (1000-4000 taxa): ~0.1-1 second per p-value
- For large n (> 4000 taxa): ~1-5 seconds per p-value

**Mitigation**: Results computed once per p-value (after bootstrap loop), not per bootstrap rep.

---

## Backward Compatibility

**Strategy**:
1. Keep `compute_sign_agreement` function (don't delete)
2. Keep `sign_agreements` in output (for comparison)
3. Old results remain valid (just use different metric)
4. New experiments automatically use new metrics

**No breaking changes**: Existing code continues to work with `compute_fiedler_estimate`.

---

## Next Steps After Implementation

1. Run small validation experiment (n=1024, L=500, K=10)
2. Compare new metrics with sign agreement
3. Verify p=1.0 gives 100% agreement
4. Update plotting scripts to visualize new metrics
5. Update README documentation
6. Consider adding variance tracking for S_avg using Welford's algorithm
