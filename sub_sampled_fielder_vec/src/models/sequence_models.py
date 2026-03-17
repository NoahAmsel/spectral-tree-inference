"""
Sequence evolution model registry and factory functions.

This module provides a registry of available sequence evolution models and factory
functions to create model instances with appropriate parameters.

Available sequence models:
- JC69: Jukes-Cantor (simplest model, equal rates)
- HKY: Hasegawa-Kishino-Yano (allows different base frequencies and kappa)
- GTR: General Time Reversible (most flexible, all rates different)
- TN93: Tamura-Nei 93 (intermediate complexity)
- T92: Tamura 92 (GC content bias)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import spectraltree
import numpy as np
from typing import Callable, Dict, Any


# Registry mapping model names to spectraltree classes
SEQUENCE_MODEL_REGISTRY: Dict[str, type] = {
    "JC69": spectraltree.Jukes_Cantor,
    "HKY": spectraltree.HKY,
    "GTR": spectraltree.GTR,
    "TN93": spectraltree.TN93,
    "T92": spectraltree.T92,
}


def get_sequence_factory(model_name: str, params: Dict[str, Any]) -> Callable[[], object]:
    """
    Create a sequence model factory function from model name and parameters.

    The factory function takes no arguments and returns a sequence model object.
    All parameters are bound at factory creation time.

    Args:
        model_name: Name of sequence model (must be in SEQUENCE_MODEL_REGISTRY)
        params: Dictionary of parameters (must include 'mutation_rate')

    Returns:
        Callable that creates a sequence model when called

    Raises:
        ValueError: If model_name is not recognized or params are invalid

    Example:
        >>> params = {"mutation_rate": 0.1, "kappa": 2.0}
        >>> factory = get_sequence_factory("HKY", params)
        >>> model = factory()  # Creates an HKY model with kappa=2.0
    """
    if model_name not in SEQUENCE_MODEL_REGISTRY:
        available = ", ".join(SEQUENCE_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown sequence model: '{model_name}'. "
            f"Available models: {available}"
        )

    base_class = SEQUENCE_MODEL_REGISTRY[model_name]

    # mutation_rate is required but handled by generate_sequences(), not by model constructor
    # We validate it here but don't pass it to model constructors

    if "mutation_rate" not in params:
        raise ValueError("mutation_rate is required in sequence parameters")

    # Create model-specific factory with bound parameters
    if model_name == "JC69":
        num_classes = params.get("num_classes", 4)  # DNA: A, C, G, T
        return lambda: base_class(num_classes=num_classes)

    elif model_name == "HKY":
        kappa = params.get("kappa", 2.0)
        stationary_freqs = params.get("stationary_freqs", np.array([1, 1, 1, 1]))

        # Ensure stationary_freqs is numpy array
        if not isinstance(stationary_freqs, np.ndarray):
            stationary_freqs = np.array(stationary_freqs)

        return lambda: base_class(kappa=kappa, stationary_freqs=stationary_freqs)

    elif model_name == "GTR":
        # GTR requires both stationary_freqs and transition_rates
        if "transition_rates" not in params:
            raise ValueError("GTR model requires 'transition_rates' parameter")

        transition_rates = params["transition_rates"]
        stationary_freqs = params.get("stationary_freqs", np.ones(4))

        # Ensure arrays are numpy arrays
        if not isinstance(transition_rates, np.ndarray):
            transition_rates = np.array(transition_rates)
        if not isinstance(stationary_freqs, np.ndarray):
            stationary_freqs = np.array(stationary_freqs)

        return lambda: base_class(
            stationary_freqs=stationary_freqs,
            transition_rates=transition_rates
        )

    elif model_name == "TN93":
        kappa1 = params.get("kappa1", 2.0)
        kappa2 = params.get("kappa2", 2.0)
        stationary_freqs = params.get("stationary_freqs", np.ones(4))

        if not isinstance(stationary_freqs, np.ndarray):
            stationary_freqs = np.array(stationary_freqs)

        return lambda: base_class(
            stationary_freqs=stationary_freqs,
            kappa1=kappa1,
            kappa2=kappa2
        )

    elif model_name == "T92":
        theta = params.get("theta", 0.5)  # GC content
        kappa1 = params.get("kappa1", 2.0)
        kappa2 = params.get("kappa2", 2.0)

        return lambda: base_class(
            theta=theta,
            kappa1=kappa1,
            kappa2=kappa2
        )

    else:
        # Should never reach here due to registry check above
        raise RuntimeError(f"Unhandled sequence model: {model_name}")


def list_sequence_models() -> Dict[str, str]:
    """
    Get list of available sequence models with descriptions.

    Returns:
        Dictionary mapping model names to descriptions
    """
    return {
        "JC69": "Jukes-Cantor - simplest model with equal rates and base frequencies",
        "HKY": "Hasegawa-Kishino-Yano - allows different base frequencies and kappa (transition/transversion ratio)",
        "GTR": "General Time Reversible - most flexible, allows all rates to differ (requires transition_rates parameter)",
        "TN93": "Tamura-Nei 93 - intermediate complexity with two kappa parameters",
        "T92": "Tamura 92 - accounts for GC content bias via theta parameter",
    }


def validate_sequence_model(model_name: str) -> bool:
    """
    Check if sequence model name is valid.

    Args:
        model_name: Name to validate

    Returns:
        True if valid, False otherwise
    """
    return model_name in SEQUENCE_MODEL_REGISTRY


def get_model_parameters_info(model_name: str) -> Dict[str, str]:
    """
    Get information about required/optional parameters for a model.

    Args:
        model_name: Model name

    Returns:
        Dictionary describing parameters

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in SEQUENCE_MODEL_REGISTRY:
        raise ValueError(f"Unknown sequence model: {model_name}")

    param_info = {
        "JC69": {
            "mutation_rate": "Required - rate of evolution along branches",
            "num_classes": "Optional (default=4) - number of character states",
        },
        "HKY": {
            "mutation_rate": "Required - rate of evolution along branches",
            "kappa": "Optional (default=2.0) - transition/transversion rate ratio",
            "stationary_freqs": "Optional (default=[1,1,1,1]) - equilibrium base frequencies [A,C,G,T]",
        },
        "GTR": {
            "mutation_rate": "Required - rate of evolution along branches",
            "transition_rates": "Required - 6-element array of relative substitution rates",
            "stationary_freqs": "Optional (default=[1,1,1,1]) - equilibrium base frequencies",
        },
        "TN93": {
            "mutation_rate": "Required - rate of evolution along branches",
            "kappa1": "Optional (default=2.0) - transition rate for purines",
            "kappa2": "Optional (default=2.0) - transition rate for pyrimidines",
            "stationary_freqs": "Optional (default=[1,1,1,1]) - equilibrium base frequencies",
        },
        "T92": {
            "mutation_rate": "Required - rate of evolution along branches",
            "theta": "Optional (default=0.5) - GC content (must be in [0,1])",
            "kappa1": "Optional (default=2.0) - transition rate parameter",
            "kappa2": "Optional (default=2.0) - transition rate parameter",
        },
    }

    return param_info[model_name]
