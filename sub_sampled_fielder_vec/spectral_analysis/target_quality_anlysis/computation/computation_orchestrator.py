"""Orchestrate all diagnostic computations."""
import numpy as np
from typing import Dict
from pathlib import Path
import sys

# Add parent directories to path for imports
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.core.utils import generate_sequences
from src.models.tree_models import get_tree_factory
from src.models.sequence_models import get_sequence_factory
from src.core.similarity_builder import SimilarityMatrixBuilder

from .coherence import compute_coherence
from .numerical_rank import compute_numerical_rank
from .partition import compute_partition_diagnostics
from .spectral_gaps import compute_spectral_gaps
from .eigenvalues import compute_eigenvalues_for_scree

# Import from output module for partition validation
from ..output.partition_validity import check_partition_valid_in_tree


def run_full_diagnostics(
    tree_config: Dict,
    seq_config: Dict,
    num_gaps: int = 1,
    min_split: int = 2,
    k_scree: int = 20
) -> Dict[str, any]:
    """
    Generate similarity matrix and compute ALL diagnostics.

    This is the main entry point that:
    1. Generates tree from tree_config
    2. Generates sequences from seq_config
    3. Builds full similarity matrix M
    4. Computes all diagnostic metrics

    Args:
        tree_config: Tree configuration dict with 'model' and 'params'
        seq_config: Sequence configuration dict with 'model', 'len', 'params'
        num_gaps: Number of gap-based thresholds for partition
        min_split: Minimum partition size
        k_scree: Number of eigenvalues for scree plots

    Returns:
        Dictionary with all computed metrics and metadata
    """
    # Extract parameters
    n = tree_config['params']['num_taxa']
    L = seq_config['len']
    mu = seq_config['params']['mutation_rate']

    # Generate tree
    tree_factory = get_tree_factory(
        tree_config['model'],
        {**tree_config['params']}
    )
    tree = tree_factory()

    # Generate sequence model
    seq_factory = get_sequence_factory(
        seq_config['model'],
        seq_config['params']
    )
    seq_model = seq_factory()

    # Generate sequences using the proper API
    observations = generate_sequences(
        num_taxa=n,
        sequence_length=L,
        mutation_rate=mu,
        tree_model=tree,
        seq_model=seq_model
    )

    # Build similarity matrix
    sim_builder = SimilarityMatrixBuilder()
    M = sim_builder.build_full(observations)

    # Compute all metrics
    coherence = compute_coherence(M, k=2)
    num_rank = compute_numerical_rank(M)
    partition_info = compute_partition_diagnostics(M, num_gaps=num_gaps, min_split=min_split)
    spectral_gaps = compute_spectral_gaps(M)
    eigenvalues = compute_eigenvalues_for_scree(M, k=k_scree)
    
    # Check if partition is valid in true tree
    is_valid = check_partition_valid_in_tree(tree, partition_info['partition'])

    # Assemble results
    results = {
        # Metadata
        'tree_model': tree_config['model'],
        'seq_model': seq_config['model'],
        'n': n,
        'L': L,
        'mu': mu,

        # Metrics
        'coherence': coherence,
        'numerical_rank': num_rank,
        'sigma2': partition_info['sigma2'],
        'partition_split': partition_info['partition_split'],
        'spectral_gap': spectral_gaps['gap'],
        'relative_spectral_gap': spectral_gaps['relative_gap'],
        'lambda2': spectral_gaps['lambda2'],
        'lambda3': spectral_gaps['lambda3'],
        'is_valid_partition': is_valid,

        # Data for plots & visualization
        'tree': tree,
        'M_eigenvalues': eigenvalues['M_eigenvalues'],
        'L_M_eigenvalues': eigenvalues['L_M_eigenvalues'],
        'fiedler_vector': partition_info['fiedler_vector'],
        'partition_mask': partition_info['partition']
    }

    return results

