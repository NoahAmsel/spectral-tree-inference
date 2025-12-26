# The σ₂ Partition Quality Paradox

**Question:** Why does the partition quality metric σ₂ give better (lower) scores for meaningless partitions at high mutation rates?

---

## Background: STDR Partitioning

Spectral Tree Down Reconstruction (STDR) recursively partitions taxa by:
1. Computing similarity matrix **M** from DNA sequences
2. Extracting Fiedler vector **v** (2nd eigenvector of Laplacian)
3. Choosing partition threshold that minimizes **σ₂** (second singular value of cross-partition submatrix)

### Core Algorithm

```python
def partition_taxa(v, similarity, num_gaps=1, min_split=1):
    """Choose partition with minimum σ₂."""
    smin = np.inf
    partition_min = v > 0
    
    # Try multiple thresholds based on gaps in sorted v
    v_sort = np.sort(v)
    gaps = v_sort[min_split:m-min_split+1] - v_sort[min_split-1:m-min_split]
    sort_idx = np.argsort(gaps)
    
    for i in range(1, num_gaps+1):
        threshold = (v_sort[sort_idx[-i]+min_split-1] + v_sort[sort_idx[-i]+min_split])/2
        bool_bipartition = v < threshold
        
        # Extract cross-partition submatrix (group A × group B)
        s_sliced = similarity[bool_bipartition, :][:, ~bool_bipartition]
        
        # Compute second singular value
        s2 = svd2(s_sliced)
        
        # Keep partition with minimum σ₂
        if s2 < smin:
            partition_min = bool_bipartition
            smin = s2
    
    return partition_min
```

### Why Minimize σ₂?

**Theory:** For a correct phylogenetic partition, the cross-partition similarity matrix should be **rank-1**:

```
S_AB ≈ σ₁ · u₁ · v₁ᵀ
```

This is because all cross-partition evolutionary paths go through a single ancestral split. The second singular value **σ₂** measures deviation from rank-1:

- **σ₂ ≈ 0**: Perfect rank-1 structure → good partition
- **σ₂ >> 0**: High-rank structure → poor partition

---

## The Paradox: Experimental Evidence

**Experiment:** Balanced binary tree, 1024 taxa, varying mutation rates

| mutation_rate | σ₂ ⬇ | Partition | Signal |
|--------------|---------|-----------|--------|
| 0.1 | **0.0144** | 512 vs 512 ✓ | std=0.036 ✓ |
| 0.2 | 0.0002 | 512 vs 512 ✓ | std=0.032 ✓ |
| 0.3 | 0.0001 | 512 vs 512 ✓ | std=0.031 ✓ |
| 0.5 | **0.00008** | **1020 vs 4** ✗ | std=0.031 ⚠ |
| 0.7 | **0.00007** | **2 vs 1022** ✗ | std=0.031 ⚠ |
| 0.9 | **0.00007** | **2 vs 1022** ✗ | std=0.031 ⚠ |

### The Contradiction

**At high mutation rates:**
- σ₂ is **lowest** (0.00007 vs 0.0144) → algorithm says "best partition"
- Partition is **degenerate** (2 vs 1022) → obviously meaningless
- True structure is 512 vs 512 → completely missed

**Visual evidence:** At mutation_rate=0.9, the Fiedler vector is nearly flat (no clear structure), yet σ₂ = 0.00007 suggests "excellent" partition quality.

---

## Explanation: Two Regimes

### Regime 1: Low Mutation (Signal Present)

At **mutation_rate ≤ 0.3**:
- Clear phylogenetic structure in sequences
- Cross-partition similarities follow rank-1 pattern from **biological signal**
- σ₂ legitimately measures partition quality ✓

### Regime 2: High Mutation (Signal Saturated)

At **mutation_rate ≥ 0.5**:
- Mutations saturate → all sequences equally dissimilar
- Similarity matrix becomes **uniform**: all entries ≈ 0.001
- **Key insight:** Any partition of a uniform matrix is rank-1!

```
For uniform matrix M ≈ c·𝟙:
  S_AB = c·𝟙  (constant submatrix)
       = c·𝟙·𝟙ᵀ  (perfect rank-1)
  → σ₂ ≈ 0  (no deviation from rank-1)
```

The algorithm minimizes σ₂ by finding the partition that makes S_AB most uniform, which at high mutation rates means putting almost all taxa in one group (1020 vs 4 or 2 vs 1022).

---

## The Core Problem

**σ₂ measures low-rank structure, not partition quality.**

A uniform (information-less) matrix has perfect rank-1 structure. Therefore:

```
Low σ₂ = Low rank structure
       ≠ Good partition

Low σ₂ can mean:
  1. True rank-1 signal (good partition) ✓
  2. Uniform noise (no partition) ✗
```

The algorithm **cannot distinguish** between these cases using σ₂ alone.

---

## Resolution

Partition quality requires **multiple signals**:

```python
# Complete assessment
valid_partition = (
    sigma2 < 0.1                    # Low rank structure
    AND balance > 0.2               # Not degenerate (20-80% split)
    AND similarity_std > threshold  # Sufficient signal variance
)
```

**Example checks:**
- **mutation_rate=0.1**: σ₂=0.014, balance=512/1024=0.50, std=0.036 → ✓ Valid
- **mutation_rate=0.9**: σ₂=0.00007, balance=2/1024=0.002, std=0.031 → ✗ Invalid (degenerate)

---

## Summary

1. **σ₂ alone fails** because low-rank structure exists in both signal and noise
2. **High mutation rates** create uniform matrices with artificially low σ₂
3. **Partition balance** and **signal variance** are essential validation metrics
4. The STDR assumption (minimize σ₂ → best partition) **breaks down** without signal

---

**Data:** `results/comparisons/mutation_rate_sweep/`  
**Code:** `spectraltree/spectral_tree_reconstruction.py::partition_taxa()`
