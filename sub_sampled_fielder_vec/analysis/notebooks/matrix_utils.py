"""Utility functions for target matrix analysis notebooks."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import sys

# Add parent directories to path for imports
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.core.metric_computer import MetricComputer


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cached_matrix(cache_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load all cached data from an experiment directory.

    Args:
        cache_dir: Path to cache directory (e.g., n512_L10000_mu0.100_balanced_binary_JC69)

    Returns:
        Dictionary with keys: 'similarity_matrix', 'fiedler_vector', 'observations', 'tree'
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise ValueError(f"Cache directory not found: {cache_dir}")

    data = {}

    # Load similarity matrix
    sim_path = cache_dir / "similarity_matrix.npz"
    if sim_path.exists():
        sim_data = np.load(sim_path)
        data['similarity_matrix'] = sim_data['similarity_matrix']

    # Load Fiedler reference
    fiedler_path = cache_dir / "fiedler_ref.npz"
    if fiedler_path.exists():
        fiedler_data = np.load(fiedler_path)
        # Extract the actual vector (might be stored under different keys)
        if 'fiedler_vector' in fiedler_data:
            data['fiedler_vector'] = fiedler_data['fiedler_vector']
        elif 'arr_0' in fiedler_data:
            data['fiedler_vector'] = fiedler_data['arr_0']
        else:
            # Take the first array if structure is unclear
            data['fiedler_vector'] = fiedler_data[list(fiedler_data.keys())[0]]

    # Load observations
    obs_path = cache_dir / "observations.npz"
    if obs_path.exists():
        obs_data = np.load(obs_path)
        if 'observations' in obs_data:
            data['observations'] = obs_data['observations']
        elif 'arr_0' in obs_data:
            data['observations'] = obs_data['arr_0']

    # Load metadata
    meta_path = cache_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            data['metadata'] = json.load(f)

    return data


def find_cache_by_params(n: int, L: int, mu: float, tree_model: str,
                         cache_root: Optional[Path] = None) -> Path:
    """
    Find cache directory by experiment parameters.

    Args:
        n: Number of taxa
        L: Sequence length
        mu: Mutation rate
        tree_model: Tree model name
        cache_root: Root cache directory (defaults to src/cache/)

    Returns:
        Path to matching cache directory
    """
    if cache_root is None:
        cache_root = PACKAGE_ROOT / "src" / "cache"

    cache_root = Path(cache_root)

    # Build expected directory name
    # Format: n{n}_L{L}_mu{mu:.3f}_{tree_model}_JC69
    expected_name = f"n{n}_L{L}_mu{mu:.3f}_{tree_model}_JC69"

    cache_dir = cache_root / expected_name
    if cache_dir.exists():
        return cache_dir

    # If exact match not found, try to find similar
    pattern = f"n{n}_L{L}_*{tree_model}*"
    matches = list(cache_root.glob(pattern))

    if matches:
        print(f"⚠️  Exact match not found. Did you mean: {matches[0].name}?")
        return matches[0]

    raise ValueError(f"No cache found for parameters: {expected_name}")


def list_available_caches(cache_root: Optional[Path] = None) -> List[Dict]:
    """
    List all available cached experiments with their parameters.

    Returns:
        List of dicts with cache info: name, path, n, L, mu, tree_model
    """
    if cache_root is None:
        cache_root = PACKAGE_ROOT / "src" / "cache"

    cache_root = Path(cache_root)
    caches = []

    for cache_dir in sorted(cache_root.iterdir()):
        if not cache_dir.is_dir() or cache_dir.name.startswith('.'):
            continue

        # Parse directory name: n{n}_L{L}_mu{mu}_{tree_model}_JC69
        parts = cache_dir.name.split('_')
        try:
            cache_info = {
                'name': cache_dir.name,
                'path': cache_dir,
                'n': int(parts[0][1:]),  # Remove 'n' prefix
                'L': int(parts[1][1:]),  # Remove 'L' prefix
                'mu': float(parts[2][2:]),  # Remove 'mu' prefix
                'tree_model': '_'.join(parts[3:-1])  # Everything between mu and JC69
            }
            caches.append(cache_info)
        except (IndexError, ValueError):
            # Skip directories that don't match expected format
            continue

    return caches


# ============================================================================
# MATRIX VISUALIZATIONS
# ============================================================================

def plot_matrix_heatmap(matrix: np.ndarray, title: str = "Similarity Matrix",
                        figsize: Tuple[int, int] = (10, 8),
                        cmap: str = 'viridis',
                        show_colorbar: bool = True,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a heatmap of the matrix.

    Args:
        matrix: 2D matrix to visualize
        title: Plot title
        figsize: Figure size (if ax is None)
        cmap: Colormap name
        show_colorbar: Whether to show colorbar
        vmin, vmax: Color scale limits (None = auto)
        ax: Matplotlib axes (if None, creates new figure)

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Taxa index', fontsize=11)
    ax.set_ylabel('Taxa index', fontsize=11)

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax


def plot_reordered_matrix(matrix: np.ndarray, fiedler_vector: np.ndarray,
                          title: str = "Matrix Reordered by Fiedler Vector",
                          figsize: Tuple[int, int] = (10, 8),
                          cmap: str = 'viridis',
                          show_partition: bool = True,
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot matrix reordered by Fiedler vector to reveal block structure.

    Args:
        matrix: Similarity matrix
        fiedler_vector: Fiedler vector for ordering
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        show_partition: Whether to show partition line
        ax: Matplotlib axes

    Returns:
        Matplotlib axes object
    """
    # Sort by Fiedler vector
    sort_idx = np.argsort(fiedler_vector)
    reordered = matrix[sort_idx][:, sort_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(reordered, cmap=cmap, aspect='auto', interpolation='nearest')

    # Add partition line if requested
    if show_partition:
        # Find split point (where Fiedler vector changes sign)
        partition = np.sum(fiedler_vector < 0)
        ax.axhline(partition - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(partition - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(0.02, 0.98, f'Partition at {partition}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Reordered taxa index', fontsize=11)
    ax.set_ylabel('Reordered taxa index', fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax


def plot_log_scale_matrix(matrix: np.ndarray,
                           title: str = "Matrix (Log Scale)",
                           figsize: Tuple[int, int] = (10, 8),
                           cmap: str = 'viridis',
                           epsilon: float = 1e-10,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot matrix with log-scale colors to reveal structure.

    Args:
        matrix: Similarity matrix
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        epsilon: Small value to add before log (avoid log(0))
        ax: Matplotlib axes

    Returns:
        Matplotlib axes object
    """
    # Apply log transform (add epsilon to avoid log(0))
    log_matrix = np.log10(np.abs(matrix) + epsilon)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(log_matrix, cmap=cmap, aspect='auto', interpolation='nearest')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Taxa index', fontsize=11)
    ax.set_ylabel('Taxa index', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log₁₀(|value|)', fontsize=10)

    return ax


def plot_difference_map(matrix1: np.ndarray, matrix2: np.ndarray,
                        label1: str = "Target", label2: str = "Subsampled",
                        figsize: Tuple[int, int] = (15, 5),
                        cmap: str = 'RdBu_r') -> plt.Figure:
    """
    Plot two matrices side-by-side with their difference.

    Args:
        matrix1: First matrix (e.g., target)
        matrix2: Second matrix (e.g., subsampled)
        label1: Label for first matrix
        label2: Label for second matrix
        figsize: Figure size
        cmap: Colormap for difference

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot first matrix
    im1 = axes[0].imshow(matrix1, cmap='viridis', aspect='auto', interpolation='nearest')
    axes[0].set_title(f'{label1} Matrix', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Taxa index')
    axes[0].set_ylabel('Taxa index')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot second matrix
    im2 = axes[1].imshow(matrix2, cmap='viridis', aspect='auto', interpolation='nearest')
    axes[1].set_title(f'{label2} Matrix', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Taxa index')
    axes[1].set_ylabel('Taxa index')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot difference
    diff = matrix1 - matrix2
    max_abs = np.max(np.abs(diff))
    im3 = axes[2].imshow(diff, cmap=cmap, aspect='auto',
                         vmin=-max_abs, vmax=max_abs, interpolation='nearest')
    axes[2].set_title(f'Difference ({label1} - {label2})', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Taxa index')
    axes[2].set_ylabel('Taxa index')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Add statistics
    diff_stats = f"Mean: {np.mean(diff):.4f}\nStd: {np.std(diff):.4f}\nMax|·|: {max_abs:.4f}"
    axes[2].text(0.02, 0.98, diff_stats, transform=axes[2].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    return fig


# ============================================================================
# SPECTRAL ANALYSIS
# ============================================================================

def plot_eigenvalue_spectrum(matrix: np.ndarray, k: int = 50,
                             title: str = "Eigenvalue Spectrum",
                             figsize: Tuple[int, int] = (12, 5),
                             highlight_gaps: bool = True) -> plt.Figure:
    """
    Plot eigenvalue spectrum with spectral gap analysis.

    Args:
        matrix: Symmetric matrix to analyze
        k: Number of eigenvalues to compute
        title: Plot title
        figsize: Figure size
        highlight_gaps: Whether to highlight spectral gaps

    Returns:
        Matplotlib figure object
    """
    # Compute eigenvalues (use eigh for symmetric matrices)
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

    # Take top k
    if len(eigenvalues) > k:
        eigenvalues = eigenvalues[:k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Eigenvalue magnitudes
    ax1.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-',
             linewidth=2, markersize=5, color='steelblue')
    ax1.set_xlabel('Index', fontsize=11)
    ax1.set_ylabel('Eigenvalue', fontsize=11)
    ax1.set_title('Eigenvalue Decay', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Highlight spectral gaps
    if highlight_gaps and len(eigenvalues) > 1:
        gaps = np.abs(np.diff(eigenvalues))
        max_gap_idx = np.argmax(gaps)
        ax1.axvline(max_gap_idx + 1.5, color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Max gap at {max_gap_idx + 1}')
        ax1.legend(fontsize=9)

    # Plot 2: Spectral gaps (differences)
    if len(eigenvalues) > 1:
        gaps = np.abs(np.diff(eigenvalues))
        ax2.bar(range(1, len(gaps) + 1), gaps, color='coral', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Index', fontsize=11)
        ax2.set_ylabel('Spectral Gap (|λᵢ - λᵢ₊₁|)', fontsize=11)
        ax2.set_title('Spectral Gaps', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_yscale('log')

        # Annotate max gap
        max_gap_idx = np.argmax(gaps)
        ax2.annotate(f'Max: {gaps[max_gap_idx]:.4f}',
                    xy=(max_gap_idx + 1, gaps[max_gap_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def analyze_coherence(matrix: np.ndarray, k: int = 2) -> Dict[str, float]:
    """
    Compute matrix coherence metrics.

    Args:
        matrix: Similarity matrix
        k: Number of top singular vectors to consider

    Returns:
        Dictionary with coherence metrics
    """
    metric_computer = MetricComputer(coherence_k=k)

    # Compute coherence
    eigvecs, singular_values = metric_computer._compute_largest_eigenvalues(matrix, k=k)
    coherence = metric_computer._compute_coherence(eigvecs)

    # Compute per-row max (another measure of coherence)
    row_max = np.max(np.abs(eigvecs), axis=1)

    return {
        'coherence': float(coherence),
        'mean_row_max': float(np.mean(row_max)),
        'std_row_max': float(np.std(row_max)),
        'max_row_max': float(np.max(row_max)),
        'top_k_singular_values': singular_values.tolist()
    }


def plot_coherence_distribution(matrix: np.ndarray, k: int = 2,
                                 fiedler_vector: Optional[np.ndarray] = None,
                                 figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """
    Visualize coherence distribution across rows.

    Args:
        matrix: Similarity matrix
        k: Number of top singular vectors
        fiedler_vector: Optional Fiedler vector for spatial coherence analysis
        figsize: Figure size (auto-calculated if None)

    Returns:
        Matplotlib figure object
    """
    metric_computer = MetricComputer(coherence_k=k)
    eigvecs, singular_values = metric_computer._compute_largest_eigenvalues(matrix, k=k)

    # Compute per-row max squared (coherence measure)
    row_max_squared = np.max(eigvecs**2, axis=1)

    # Compute coherence metrics
    overall_coherence = np.max(row_max_squared)
    mean_coherence = np.mean(row_max_squared)
    std_coherence = np.std(row_max_squared)

    # Determine number of subplots
    n_plots = 3 if fiedler_vector is not None else 2
    if figsize is None:
        figsize = (18, 5) if n_plots == 3 else (12, 5)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    else:
        axes = [axes[0], axes[1], axes[2]]

    # Plot 1: Distribution histogram
    axes[0].hist(row_max_squared, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(mean_coherence, color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {mean_coherence:.4f}')
    axes[0].set_xlabel('Row Max Squared (||uᵢ||²∞)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Coherence Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Per-row coherence
    axes[1].plot(row_max_squared, 'o-', markersize=3, linewidth=0.5, color='coral', alpha=0.6)
    axes[1].axhline(mean_coherence, color='red', linestyle='--',
                    linewidth=2, alpha=0.7, label='Mean')
    axes[1].set_xlabel('Row Index', fontsize=11)
    axes[1].set_ylabel('Row Max Squared', fontsize=11)
    axes[1].set_title('Per-Row Coherence', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Coherence ordered by Fiedler vector (if available)
    if fiedler_vector is not None:
        # Sort by Fiedler vector to reveal spatial patterns
        sort_idx = np.argsort(fiedler_vector)
        coherence_sorted = row_max_squared[sort_idx]
        fiedler_sorted = fiedler_vector[sort_idx]

        # Plot coherence in Fiedler order
        axes[2].plot(coherence_sorted, linewidth=2, color='purple', alpha=0.7)
        axes[2].scatter(range(len(coherence_sorted)), coherence_sorted,
                       s=15, alpha=0.4, color='purple')

        # Mark partition boundary (where Fiedler changes sign)
        partition_idx = np.sum(fiedler_sorted < 0)
        if 0 < partition_idx < len(fiedler_sorted):
            axes[2].axvline(partition_idx - 0.5, color='red', linestyle='--',
                           linewidth=2, alpha=0.7, label=f'Partition at {partition_idx}')

            # Add shaded regions for the two partitions
            axes[2].axvspan(0, partition_idx - 0.5, alpha=0.1, color='blue',
                           label='Partition 1')
            axes[2].axvspan(partition_idx - 0.5, len(coherence_sorted),
                           alpha=0.1, color='green', label='Partition 2')

        axes[2].set_xlabel('Index (Fiedler-ordered)', fontsize=11)
        axes[2].set_ylabel('Row Coherence (||uᵢ||²∞)', fontsize=11)
        axes[2].set_title('Coherence Spatial Pattern', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=9, loc='best')
        axes[2].grid(True, alpha=0.3)

    # Create subtitle with all coherence metrics
    subtitle_lines = [
        f"μ={overall_coherence:.6f}, mean={mean_coherence:.6f}, std={std_coherence:.6f}, max={np.max(row_max_squared):.6f}"
    ]
    if len(singular_values) > 0:
        sv_str = ', '.join([f'{sv:.4f}' for sv in singular_values[:k]])
        subtitle_lines.append(f"Top-{k} singular values: [{sv_str}]")

    fig.suptitle('Coherence Analysis\n' + ' | '.join(subtitle_lines),
                 fontsize=11, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_svd_components(matrix: np.ndarray, k: int = 4,
                        figsize: Optional[Tuple[int, int]] = None,
                        fiedler_vector: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Visualize top-k singular vectors.

    For symmetric matrices, only shows one set of vectors (since U = V).
    For non-symmetric matrices, shows both left (U) and right (Vᵀ) vectors.

    Args:
        matrix: Matrix to decompose
        k: Number of components to show
        figsize: Figure size (auto-calculated if None)
        fiedler_vector: If provided, plot vectors in Fiedler-ordered format

    Returns:
        Matplotlib figure object
    """
    # Check if matrix is symmetric (only for square matrices)
    is_symmetric = (matrix.shape[0] == matrix.shape[1] and
                   np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-8))

    # Compute SVD
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Determine ordering (only use fiedler for square symmetric matrices)
    if fiedler_vector is not None and is_symmetric:
        sort_idx = np.argsort(fiedler_vector)
    else:
        sort_idx = np.arange(matrix.shape[0])
        fiedler_vector = None  # Don't use fiedler for non-symmetric

    # Create subplots: 1 column for symmetric, 2 columns for non-symmetric
    n_cols = 1 if is_symmetric else 2
    if figsize is None:
        figsize = (7, 2.5 * k) if is_symmetric else (14, 2.5 * k)

    fig, axes = plt.subplots(k, n_cols, figsize=figsize, squeeze=False)

    for i in range(k):
        # Left singular vector (or eigenvector for symmetric case)
        left_vec = U[sort_idx, i]
        axes[i, 0].plot(left_vec, linewidth=2, color='steelblue')
        axes[i, 0].axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[i, 0].set_ylabel(f'σ={S[i]:.2f}', fontsize=10, fontweight='bold')
        axes[i, 0].grid(True, alpha=0.3)

        if i == 0:
            if is_symmetric:
                axes[i, 0].set_title('Eigenvectors (Singular Vectors)',
                                    fontsize=12, fontweight='bold')
            else:
                axes[i, 0].set_title('Left Singular Vectors (U)',
                                    fontsize=12, fontweight='bold')
        if i == k - 1:
            axes[i, 0].set_xlabel('Index (Fiedler-ordered)' if fiedler_vector is not None
                                  else 'Index', fontsize=10)

        # Right singular vector (only for non-symmetric matrices)
        if not is_symmetric:
            right_vec = Vt[i, :]  # No reordering for right vectors
            axes[i, 1].plot(right_vec, linewidth=2, color='coral')
            axes[i, 1].axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            axes[i, 1].set_ylabel(f'σ={S[i]:.2f}', fontsize=10, fontweight='bold')
            axes[i, 1].grid(True, alpha=0.3)

            if i == 0:
                axes[i, 1].set_title('Right Singular Vectors (Vᵀ)',
                                    fontsize=12, fontweight='bold')
            if i == k - 1:
                axes[i, 1].set_xlabel('Index', fontsize=10)

    # Add title with symmetry note
    title = f'Top-{k} SVD Components'
    if is_symmetric:
        title += ' (Symmetric Matrix: U = V)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    return fig


def compute_block_quality(matrix: np.ndarray, fiedler_vector: np.ndarray) -> Dict[str, float]:
    """
    Compute block structure quality metrics.

    Measures how well the matrix exhibits block structure based on
    Fiedler vector partitioning.

    Args:
        matrix: Similarity matrix
        fiedler_vector: Fiedler vector for partitioning

    Returns:
        Dictionary with block quality metrics
    """
    # Partition based on Fiedler vector sign
    block1_mask = fiedler_vector < 0
    block2_mask = ~block1_mask

    n1 = np.sum(block1_mask)
    n2 = np.sum(block2_mask)

    # Compute within-block and between-block similarities
    within_1 = matrix[np.ix_(block1_mask, block1_mask)]
    within_2 = matrix[np.ix_(block2_mask, block2_mask)]
    between = matrix[np.ix_(block1_mask, block2_mask)]

    # Compute statistics
    mean_within_1 = np.mean(within_1)
    mean_within_2 = np.mean(within_2)
    mean_within = (n1 * mean_within_1 + n2 * mean_within_2) / (n1 + n2)
    mean_between = np.mean(between)

    # Block quality ratio (higher = better block structure)
    quality_ratio = mean_within / (mean_between + 1e-10)

    return {
        'block1_size': int(n1),
        'block2_size': int(n2),
        'mean_within_block1': float(mean_within_1),
        'mean_within_block2': float(mean_within_2),
        'mean_within': float(mean_within),
        'mean_between': float(mean_between),
        'quality_ratio': float(quality_ratio),
        'contrast': float(mean_within - mean_between)
    }


def plot_block_quality(matrix: np.ndarray, fiedler_vector: np.ndarray,
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Visualize block structure quality.

    Args:
        matrix: Similarity matrix
        fiedler_vector: Fiedler vector
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    metrics = compute_block_quality(matrix, fiedler_vector)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Bar chart of means
    categories = ['Within\nBlock 1', 'Within\nBlock 2', 'Between\nBlocks']
    values = [metrics['mean_within_block1'],
              metrics['mean_within_block2'],
              metrics['mean_between']]
    colors = ['steelblue', 'cornflowerblue', 'coral']

    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Mean Similarity', fontsize=11)
    ax1.set_title('Block Structure Analysis', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Block size visualization
    labels = [f"Block 1\n(n={metrics['block1_size']})",
              f"Block 2\n(n={metrics['block2_size']})"]
    sizes = [metrics['block1_size'], metrics['block2_size']]
    colors_pie = ['steelblue', 'cornflowerblue']

    ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Partition Sizes', fontsize=12, fontweight='bold')

    # Add quality metrics as text
    quality_text = (f"Quality Ratio: {metrics['quality_ratio']:.3f}\n"
                   f"Contrast: {metrics['contrast']:.4f}")
    fig.text(0.5, 0.02, quality_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig
