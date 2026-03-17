"""
Tree model registry and factory functions.

This module provides a registry of available tree topology models and factory
functions to create tree instances with appropriate parameters.

Available tree models:
- balanced_binary: Perfectly balanced binary tree (requires num_taxa = power of 2)
- lopsided: Unbalanced tree (one leaf splits off at each step)
- kingman: Coalescent model with Kingman tree (realistic for population genetics)
- birth_death: Birth-death process tree (common in phylogenetics)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import spectraltree
from typing import Callable, Dict, Any


# Registry mapping model names to spectraltree functions
TREE_MODEL_REGISTRY: Dict[str, Callable] = {
    "balanced_binary": spectraltree.balanced_binary,
    "lopsided": spectraltree.lopsided_tree,
    "kingman": spectraltree.unrooted_pure_kingman_tree,
    "kingman_mean": spectraltree.unrooted_mean_kingman_tree,
    "birth_death": spectraltree.unrooted_birth_death_tree,
}


def get_tree_factory(model_name: str, params: Dict[str, Any]) -> Callable[[], object]:
    """
    Create a tree factory function from model name and parameters.

    The factory function takes no arguments and returns a dendropy Tree object.
    All parameters are bound at factory creation time.

    Args:
        model_name: Name of tree model (must be in TREE_MODEL_REGISTRY)
        params: Dictionary of parameters (must include 'num_taxa')

    Returns:
        Callable that creates a tree when called

    Raises:
        ValueError: If model_name is not recognized or params are invalid

    Example:
        >>> params = {"num_taxa": 128, "edge_length": 1.0}
        >>> factory = get_tree_factory("balanced_binary", params)
        >>> tree = factory()  # Creates a balanced binary tree with 128 taxa
    """
    if model_name not in TREE_MODEL_REGISTRY:
        available = ", ".join(TREE_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown tree model: '{model_name}'. "
            f"Available models: {available}"
        )

    base_fn = TREE_MODEL_REGISTRY[model_name]

    # Extract num_taxa (required for all models)
    if "num_taxa" not in params:
        raise ValueError("num_taxa is required in tree parameters")
    num_taxa = params["num_taxa"]

    # Create model-specific factory with bound parameters
    if model_name == "balanced_binary":
        edge_length = params.get("edge_length", 1.0)
        return lambda: base_fn(num_taxa=num_taxa, edge_length=edge_length)

    elif model_name == "lopsided":
        edge_length = params.get("edge_length", 1.0)
        return lambda: base_fn(num_taxa=num_taxa, edge_length=edge_length)

    elif model_name == "kingman":
        pop_size = params.get("pop_size", 1.0)
        # Note: spectraltree function is unrooted_pure_kingman_tree(num_taxa, taxon_namespace, pop_size, rng)
        return lambda: base_fn(num_taxa=num_taxa, pop_size=pop_size)

    elif model_name == "kingman_mean":
        pop_size = params.get("pop_size", 1.0)
        return lambda: base_fn(
            taxon_namespace=spectraltree.default_namespace(num_taxa),
            pop_size=pop_size
        )

    elif model_name == "birth_death":
        birth_rate = params.get("birth_rate", 0.5)
        death_rate = params.get("death_rate", 0.0)
        # Note: spectraltree function is unrooted_birth_death_tree(num_taxa, namespace, birth_rate, death_rate)
        return lambda: base_fn(
            num_taxa=num_taxa,
            birth_rate=birth_rate,
            death_rate=death_rate
        )

    else:
        # Should never reach here due to registry check above
        raise RuntimeError(f"Unhandled tree model: {model_name}")


def list_tree_models() -> Dict[str, str]:
    """
    Get list of available tree models with descriptions.

    Returns:
        Dictionary mapping model names to descriptions
    """
    return {
        "balanced_binary": "Perfectly balanced binary tree (num_taxa must be power of 2)",
        "lopsided": "Unbalanced tree where one leaf splits off at each step",
        "kingman": "Coalescent model (Kingman tree) - realistic for population genetics",
        "kingman_mean": "Mean-rate Kingman coalescent with deterministic branch lengths",
        "birth_death": "Birth-death process - common in phylogenetics",
    }


def validate_tree_model(model_name: str) -> bool:
    """
    Check if tree model name is valid.

    Args:
        model_name: Name to validate

    Returns:
        True if valid, False otherwise
    """
    return model_name in TREE_MODEL_REGISTRY
