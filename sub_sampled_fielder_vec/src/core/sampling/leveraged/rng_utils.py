"""Random number generator utilities."""
import numpy as np


def get_rng(seed=None):
    """
    Initialize random number generator.
    
    Args:
        seed: Random seed for reproducibility (None for default)
        
    Returns:
        Random number generator instance
    """
    if seed is not None:
        if hasattr(np.random, 'default_rng'):
            return np.random.default_rng(seed)
        else:
            return np.random.RandomState(seed)
    else:
        return np.random
