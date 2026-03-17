"""Partition validity checking against ground truth tree."""
import numpy as np
from spectraltree.utils import TaxaMetadata, check_is_bipartition


def check_partition_valid_in_tree(tree, partition_mask: np.ndarray) -> bool:
    """
    Check if a partition corresponds to a real edge in the tree.
    
    Args:
        tree: dendropy Tree object (ground truth)
        partition_mask: Boolean array indicating partition membership
        
    Returns:
        True if partition matches an edge in tree, False otherwise
    """
    if partition_mask is None:
        return False
    
    tree.encode_bipartitions()
    # Use actual leaf nodes (not full namespace) to handle trees with extinction
    # where some taxa in the namespace don't appear as leaves
    leaf_taxa = [leaf.taxon for leaf in tree.leaf_nodes()]
    meta = TaxaMetadata(tree.taxon_namespace, leaf_taxa)
    return check_is_bipartition(tree, partition_mask, meta)

