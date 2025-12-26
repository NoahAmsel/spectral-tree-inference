# The σ₂ Partition Quality Paradox in Spectral Tree Reconstruction

**Date:** November 28, 2025  
**Context:** Critical analysis of partition quality metrics in STDR algorithm

---

## Background: Spectral Tree Down Reconstruction (STDR)

### The Task
Reconstruct phylogenetic trees from DNA sequence data using spectral methods. The algorithm recursively partitions taxa into two groups based on spectral properties of a similarity matrix.

### The Algorithm Flow

1. **Input**: DNA sequences from n taxa
2. **Compute similarity matrix M**: Pairwise similarities between all sequences (n×n matrix)
3. **Compute Laplacian**: L = D - M, where D is the degree matrix
4. **Extract Fiedler vector**: Second smallest eigenvector of L (spectral embedding)
5. **Partition taxa**: Split taxa into two groups using Fiedler vector
6. **Recurse**: Apply algorithm recursively to each group

### Core Partitioning Algorithm

The critical step is choosing the partition threshold. The algorithm selects the partition with **minimum σ₂** (second singular value):

```python
def partition_taxa(v, similarity, num_gaps=1, min_split=1):
    """
    Partition taxa using Fiedler vector v and similarity matrix.
    Chooses partition with minimum second singular value (σ₂).
    
    Args:
        v: Fiedler vector (n-dimensional)
        similarity: Similarity matrix (n×n)
        num_gaps: Number of gap-based thresholds to try
        min_split: Minimum partition size
    
    Returns:
        partition: Boolean array indicating group membership
    """
    m = len(v)
    partition_min = v > 0
    smin = np.inf
    
    # Try multiple threshold candidates based on gaps in sorted v
    v_sort = np.sort(v)
    gaps = v_sort[min_split:m-min_split+1] - v_sort[min_split-1:m-min_split]
    sort_idx = np.argsort(gaps)
    
    for i in range(1, num_gaps+1):
        threshold = (v_sort[sort_idx[-i]+min_split-1] + v_sort[sort_idx[-i]+min_split])/2
        bool_bipartition = v < threshold
        
        if np.minimum(np.sum(bool_bipartition), np.sum(~bool_bipartition)) >= min_split:
            # Extract cross-partition submatrix
            s_sliced = similarity[bool_bipartition, :]
            s_sliced = s_sliced[:, ~bool_bipartition]
            
            # Compute second singular value
            s2 = svd2(s_sliced)
            
            # Choose partition with minimum σ₂
            if s2 < smin:
                partition_min = bool_bipartition
                smin = s2
    
    return partition_min


def svd2(mat):
    """Return second singular value of matrix."""
    if (mat.shape[0] == 1) | (mat.shape[1] == 1):
        return 0
    elif (mat.shape[0] == 2) | (mat.shape[1] == 2):
        return np.linalg.svd(mat, False, False)[1]
    else:
        sigmas = TruncatedSVD(n_components=2).fit(mat).singular_values_
        return sigmas[1]
```

---

## Theory: Why Minimize σ₂?

### The Rank-1 Ideal

For a **correct phylogenetic partition**, the cross-partition similarity matrix should have **rank-1 structure**:

```
S_AB = similarities between group A and group B
S_AB ≈ σ₁ · u₁ · v₁ᵀ    (rank-1 approximation)
```

**Biological intuition:**
- Group A sequences share common ancestry within A
- Group B sequences share common ancestry within B  
- Cross-partition similarities determined by single ancestral split point

The second singular value **σ₂ measures deviation from rank-1**:
- **σ₂ ≈ 0**: Perfect rank-1 structure → ideal partition
- **σ₂ >> 0**: High-rank structure → poor partition

### Standard Interpretation

```
σ₂ < 0.05  → Excellent partition
σ₂ < 0.1   → Good partition
σ₂ > 0.5   → Poor partition
```

---

## The Paradox: Experimental Evidence

### Experiment Design

- **Tree model**: Balanced binary tree (perfect bifurcating structure)
- **Number of taxa**: 1,024 (2^10)
- **Expected partition**: 512 vs 512 at root split
- **Sequence length**: 5,000 bp
- **Evolution model**: Jukes-Cantor (JC69)
- **Swept parameter**: Mutation rate (0.1 to 0.9)

### Complete Experimental Results

| mutation_rate | σ₂ | group_a | group_b | sim_mean | sim_std | fiedler_std |
|--------------|---------|---------|---------|----------|---------|-------------|
| 0.1 | 0.0144 | 512 | 512 | 0.00383 | 0.0360 | 0.03125 |
| 0.2 | 0.0002 | 512 | 512 | 0.00131 | 0.0319 | 0.03125 |
| 0.3 | 0.0001 | 512 | 512 | 0.00109 | 0.0314 | 0.03125 |
| 0.5 | 0.00008 | **1020** | **4** | 0.00100 | 0.0312 | 0.03125 |
| 0.7 | 0.00007 | **2** | **1022** | 0.00098 | 0.0312 | 0.03125 |
| 0.9 | 0.00007 | **2** | **1022** | 0.00098 | 0.0312 | 0.03125 |

### Example: Full Metrics (mutation_rate = 0.1)

```json
{
  "metrics": {
    "s2": 0.014433,
    "group_a_size": 512,
    "group_b_size": 512,
    "similarity_mean": 0.00383,
    "similarity_std": 0.0360,
    "fiedler_mean": 1.4e-16,
    "fiedler_std": 0.03125
  }
}
```

### Example: Full Metrics (mutation_rate = 0.9)

```json
{
  "metrics": {
    "s2": 0.000074,
    "group_a_size": 2,
    "group_b_size": 1022,
    "similarity_mean": 0.00098,
    "similarity_std": 0.0312,
    "fiedler_mean": 2.8e-15,
    "fiedler_std": 0.03125
  }
}
```

---

## The Problem: Contradictory Signals

### Observation 1: σ₂ Decreases with Mutation Rate

```
mutation_rate = 0.1 → σ₂ = 0.0144  (14x higher)
mutation_rate = 0.9 → σ₂ = 0.00007 (100x lower!)
```

According to σ₂, **high mutation rates produce "better" partitions**.

### Observation 2: Partitions Become Degenerate

```
mutation_rate = 0.1 → partition: 512 vs 512  (balanced, correct)
mutation_rate = 0.9 → partition: 2 vs 1022   (degenerate, meaningless)
```

At high mutation rates, the algorithm places almost all taxa in one group.

### Observation 3: Fiedler Vector is Flat

The Fiedler vector standard deviation remains constant (**fiedler_std = 0.03125**) across all conditions, indicating no clear plateau structure emerges at high mutation rates despite lower σ₂.

### Observation 4: Signal Variance Collapses

```
mutation_rate = 0.1 → similarity_std = 0.0360  (high variance, strong signal)
mutation_rate = 0.9 → similarity_std = 0.0312  (low variance, weak signal)
```

The similarity matrix loses information content as mutations saturate.

---

## Root Cause Analysis

### Two Distinct Regimes

**σ₂ can be low for fundamentally different reasons:**

#### Regime 1: True Phylogenetic Signal (Low Mutation)

At **mutation_rate = 0.1-0.3**:
- Clear phylogenetic structure preserved in sequences
- Similarity matrix contains evolutionary information
- Cross-partition similarities follow rank-1 pattern from **true biological signal**
- Balanced partition (512 vs 512) reflects true tree structure
- **σ₂ ≈ 0.001-0.014**: Legitimately measures partition quality ✓

#### Regime 2: Signal Saturation (High Mutation)

At **mutation_rate = 0.7-0.9**:
- Mutations accumulate beyond saturation point
- All sequences become equally dissimilar
- Similarity matrix becomes nearly **uniform**: all entries ≈ 0.001
- **Any partition of a uniform matrix is approximately rank-1** (constant matrix)
- Algorithm finds trivial partition (2 vs 1022) by partitioning noise
- **σ₂ ≈ 0.00007**: Measures rank structure of **noise**, not signal ✗

### Mathematical Explanation

For a uniform matrix **M ≈ c·𝟙** (all entries approximately constant c):

```
Cross-partition submatrix: S_AB ≈ c·𝟙  (constant matrix)
SVD: S_AB = c·𝟙·𝟙ᵀ  (perfect rank-1)
σ₁ ≈ c·√(n_A·n_B)  (first singular value scales with partition size)
σ₂ ≈ 0  (second singular value vanishes)
```

**Key insight**: A uniform (information-less) matrix has **perfect rank-1 structure** regardless of how it's partitioned. Low σ₂ indicates low rank, **not** partition quality.

---

## Critical Interpretation

### ❌ Incorrect Interpretation

```
Lower σ₂ = Better partition quality
```

This fails because it conflates two distinct phenomena:
1. Low σ₂ from **structured signal** (good)
2. Low σ₂ from **uniform noise** (meaningless)

### ✓ Correct Interpretation

```python
# Pseudo-code for proper assessment
if partition_balance < 0.2:
    quality = "INVALID - Degenerate partition"
elif similarity_std < threshold:
    quality = "INVALID - Insufficient signal"
elif sigma2 < 0.05:
    quality = "Excellent partition"
elif sigma2 < 0.1:
    quality = "Good partition"
else:
    quality = "Poor partition"
```

**Partition quality = LOW σ₂ AND balanced split AND sufficient signal variance**

---

## Essential Additional Metrics

### 1. Partition Balance

```python
balance = min(group_a_size, group_b_size) / total_taxa
```

Extreme imbalance (e.g., 2/1024 = 0.002) indicates no meaningful partition was found.

### 2. Signal Variance

```python
signal_quality = similarity_std
```

Low variance indicates similarity matrix has lost phylogenetic information.

### 3. Fiedler Structure

Visual inspection or quantitative measures (gap size, plateau R²) of whether Fiedler vector has clear bipartition structure.

---

## Conclusions

1. **σ₂ alone is unreliable** for assessing partition quality in spectral tree reconstruction

2. At **high mutation rates**, σ₂ becomes **artificially low** due to:
   - Similarity matrix losing phylogenetic signal
   - Near-uniform matrices having perfect rank-1 structure
   - Algorithm partitioning noise rather than signal

3. The **partition becomes degenerate** (2 vs 1022) precisely when σ₂ is lowest (0.00007), contradicting the standard interpretation

4. Proper assessment requires **multiple validation signals**:
   - Partition balance (reject if < 20%)
   - Similarity matrix variance (reject if too low)
   - Fiedler vector structure (check for clear plateaus)
   - σ₂ value (only meaningful given other conditions)

5. The STDR algorithm's core assumption—that minimizing σ₂ yields the best partition—**breaks down** when phylogenetic signal is weak or saturated

---

**Experimental data:** `results/comparisons/mutation_rate_sweep/`  
**Algorithm code:** `spectraltree/spectral_tree_reconstruction.py::partition_taxa()`

