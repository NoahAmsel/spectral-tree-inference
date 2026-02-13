"""Persistent disk-based caching for experiment data.

This module provides functions to save and load expensive-to-compute experiment
data (trees, sequences, similarity matrices, Fiedler vectors) to/from disk.

Design principles:
- Each function has a single, well-defined responsibility
- No hidden side effects except explicit save/load/clear operations
- Cache keys are deterministic and based on experiment parameters
- Uses NPZ format for efficient numpy array storage
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .logging import log_info, log_warning


def _get_cache_key(
    n_taxa: int,
    seq_len: int,
    mutation_rate: float,
    tree_model_name: str,
    seq_model_name: str
) -> str:
    """
    Generate deterministic cache key from experiment parameters.

    Note: Seed is NOT included - we always use fixed seed=42 for cached data.

    Args:
        n_taxa: Number of taxa
        seq_len: Sequence length
        mutation_rate: Mutation rate
        tree_model_name: Name of tree model (e.g., "balanced_binary")
        seq_model_name: Name of sequence model (e.g., "Jukes_Cantor")

    Returns:
        Cache key string (safe for filesystem)

    Example:
        >>> _get_cache_key(8192, 1000, 0.1, "balanced_binary", "Jukes_Cantor")
        'n8192_L1000_mu0.100_balanced_binary_Jukes_Cantor'
    """
    # Format mutation rate with 3 decimal places for consistency
    mu_str = f"{mutation_rate:.3f}"

    # Create filesystem-safe key
    cache_key = f"n{n_taxa}_L{seq_len}_mu{mu_str}_{tree_model_name}_{seq_model_name}"

    return cache_key


def _get_cache_dir(cache_key: str) -> Path:
    """
    Get cache directory path for a given cache key.

    Creates the directory if it doesn't exist.

    Args:
        cache_key: Cache key string

    Returns:
        Path object for cache directory

    Example:
        >>> _get_cache_dir("n8192_L1000_mu0.100_balanced_binary_JC")
        PosixPath('.../cache/n8192_L1000_mu0.100_balanced_binary_JC')
    """
    # Get base directory (sub_sampled_fielder_vec)
    base_dir = Path(__file__).parent.parent
    cache_root = base_dir / "cache"
    cache_dir = cache_root / cache_key

    # Create directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def save_experiment_data(
    cache_key: str,
    tree: Any,
    observations: np.ndarray,
    similarity_matrix: np.ndarray,
    fiedler_ref: np.ndarray,
    metadata: Dict[str, Any]
) -> None:
    """
    Save all experiment data to disk cache.

    Side effect: Writes files to cache directory.

    Args:
        cache_key: Cache key identifying this experiment
        tree: Tree object (will be converted to adjacency matrix or saved as Newick)
        observations: Sequence observations (n_taxa × seq_len)
        similarity_matrix: Full similarity matrix M
        fiedler_ref: Reference Fiedler vector
        metadata: Dictionary with experiment parameters for verification

    Files created:
        - tree.npz: Tree structure
        - observations.npz: Sequence data
        - similarity_matrix.npz: Full M matrix
        - fiedler_ref.npz: Reference Fiedler vector
        - metadata.json: Experiment parameters
    """
    cache_dir = _get_cache_dir(cache_key)

    try:
        # Save tree (convert to adjacency matrix if needed)
        tree_data = _serialize_tree(tree)
        np.savez_compressed(cache_dir / "tree.npz", **tree_data)

        # Save observations
        np.savez_compressed(cache_dir / "observations.npz", observations=observations)

        # Save similarity matrix
        np.savez_compressed(cache_dir / "similarity_matrix.npz", similarity_matrix=similarity_matrix)

        # Save reference Fiedler vector
        np.savez_compressed(cache_dir / "fiedler_ref.npz", fiedler_ref=fiedler_ref)

        # Save metadata
        metadata_path = cache_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        log_info('cache', f"Saved experiment data to cache: {cache_key}")

    except Exception as e:
        log_warning('cache', f"Failed to save cache {cache_key}: {e}")
        # Don't raise - caching is optional


def load_experiment_data(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Load experiment data from disk cache.

    Args:
        cache_key: Cache key identifying this experiment

    Returns:
        Dictionary with keys: {tree, observations, similarity_matrix, fiedler_ref, metadata}
        Returns None if cache doesn't exist or is invalid

    Example:
        >>> data = load_experiment_data("n8192_L1000_mu0.100_balanced_binary_JC")
        >>> if data:
        ...     tree = data['tree']
        ...     observations = data['observations']
    """
    cache_dir = _get_cache_dir(cache_key)

    # Check if all required files exist
    required_files = [
        "tree.npz",
        "observations.npz",
        "similarity_matrix.npz",
        "fiedler_ref.npz",
        "metadata.json"
    ]

    for filename in required_files:
        if not (cache_dir / filename).exists():
            log_info('cache', f"Cache miss: {cache_key} (missing {filename})")
            return None

    try:
        # Load tree
        tree_npz = np.load(cache_dir / "tree.npz", allow_pickle=True)
        tree = _deserialize_tree(tree_npz)

        # Load observations
        obs_npz = np.load(cache_dir / "observations.npz")
        observations = obs_npz['observations']

        # Load similarity matrix
        sim_npz = np.load(cache_dir / "similarity_matrix.npz")
        similarity_matrix = sim_npz['similarity_matrix']

        # Load Fiedler vector
        fiedler_npz = np.load(cache_dir / "fiedler_ref.npz")
        fiedler_ref = fiedler_npz['fiedler_ref']

        # Load metadata
        with open(cache_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        log_info('cache', f"Cache hit: {cache_key}")

        return {
            'tree': tree,
            'observations': observations,
            'similarity_matrix': similarity_matrix,
            'fiedler_ref': fiedler_ref,
            'metadata': metadata
        }

    except Exception as e:
        log_warning('cache', f"Failed to load cache {cache_key}: {e}")
        return None


def list_cached_experiments() -> List[Dict[str, Any]]:
    """
    List all cached experiments.

    Returns:
        List of dictionaries with cache info (cache_key, metadata)

    Example:
        >>> cached = list_cached_experiments()
        >>> for item in cached:
        ...     print(f"{item['cache_key']}: {item['metadata']['n_taxa']} taxa")
    """
    base_dir = Path(__file__).parent.parent
    cache_root = base_dir / "cache"

    if not cache_root.exists():
        return []

    cached_experiments = []

    for cache_dir in cache_root.iterdir():
        if not cache_dir.is_dir():
            continue

        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            cached_experiments.append({
                'cache_key': cache_dir.name,
                'metadata': metadata,
                'path': str(cache_dir)
            })
        except Exception as e:
            log_warning('cache', f"Failed to read metadata for {cache_dir.name}: {e}")

    return cached_experiments


def clear_cache(cache_key: Optional[str] = None) -> None:
    """
    Clear persistent cache.

    Side effect: Deletes cache files from disk.

    Args:
        cache_key: Specific cache to clear. If None, clears all caches.

    Example:
        >>> clear_cache("n8192_L1000_mu0.100_balanced_binary_JC")  # Clear specific
        >>> clear_cache()  # Clear all
    """
    base_dir = Path(__file__).parent.parent
    cache_root = base_dir / "cache"

    if not cache_root.exists():
        log_info('cache', "No cache directory found")
        return

    if cache_key is None:
        # Clear all caches
        import shutil
        try:
            shutil.rmtree(cache_root)
            log_info('cache', "Cleared all caches")
        except Exception as e:
            log_warning('cache', f"Failed to clear all caches: {e}")
    else:
        # Clear specific cache
        cache_dir = cache_root / cache_key
        if cache_dir.exists():
            import shutil
            try:
                shutil.rmtree(cache_dir)
                log_info('cache', f"Cleared cache: {cache_key}")
            except Exception as e:
                log_warning('cache', f"Failed to clear cache {cache_key}: {e}")
        else:
            log_info('cache', f"Cache not found: {cache_key}")


def _serialize_tree(tree: Any) -> Dict[str, np.ndarray]:
    """
    Convert tree object to serializable format.

    Args:
        tree: Tree object from spectraltree

    Returns:
        Dictionary with numpy arrays for NPZ storage
    """
    # Check if tree has adjacency_matrix attribute
    if hasattr(tree, 'adjacency_matrix'):
        return {'adjacency_matrix': tree.adjacency_matrix}

    # Check if tree has Newick string representation
    if hasattr(tree, 'newick'):
        newick_str = tree.newick() if callable(tree.newick) else str(tree.newick)
        # Store as numpy array of strings
        return {'newick': np.array([newick_str], dtype=object)}

    # Fallback: try to convert to string
    tree_str = str(tree)
    return {'tree_str': np.array([tree_str], dtype=object)}


def _deserialize_tree(tree_npz) -> Any:
    """
    Reconstruct tree object from NPZ data.

    Args:
        tree_npz: Loaded NPZ file

    Returns:
        Tree object (or serialized representation)
    """
    # Import dendropy for tree reconstruction
    import dendropy

    if 'adjacency_matrix' in tree_npz:
        # TODO: Reconstruct tree from adjacency matrix if needed
        return tree_npz['adjacency_matrix']
    elif 'newick' in tree_npz:
        # Reconstruct DendroPy Tree from Newick string
        newick_str = str(tree_npz['newick'][0])
        tree = dendropy.Tree.get(data=newick_str, schema="newick")
        return tree
    elif 'tree_str' in tree_npz:
        # Try to parse as Newick string
        tree_str = str(tree_npz['tree_str'][0])
        try:
            tree = dendropy.Tree.get(data=tree_str, schema="newick")
            return tree
        except:
            # If parsing fails, return the string
            return tree_str
    else:
        return None
