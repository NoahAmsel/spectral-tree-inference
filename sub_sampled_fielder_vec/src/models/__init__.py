"""
Model registries and factory functions for tree and sequence models.

This module provides centralized access to all available tree topology models
and sequence evolution models.
"""

from .tree_models import (
    get_tree_factory,
    list_tree_models,
    validate_tree_model,
    TREE_MODEL_REGISTRY,
)

from .sequence_models import (
    get_sequence_factory,
    list_sequence_models,
    validate_sequence_model,
    get_model_parameters_info,
    SEQUENCE_MODEL_REGISTRY,
)

__all__ = [
    # Tree models
    "get_tree_factory",
    "list_tree_models",
    "validate_tree_model",
    "TREE_MODEL_REGISTRY",
    # Sequence models
    "get_sequence_factory",
    "list_sequence_models",
    "validate_sequence_model",
    "get_model_parameters_info",
    "SEQUENCE_MODEL_REGISTRY",
]
