"""Sampling package registry for different subsampling methods."""
from abc import ABC, abstractmethod
from typing import Dict, Type
import numpy as np

from .base import BaseSampler


def get_sampler(method: str, **kwargs) -> BaseSampler:
    """
    Get a sampler instance based on method name.

    Args:
        method: Sampling method name ("uniform", "leveraged", or "lds")
               - uniform: Simple uniform sampling (baseline)
               - leveraged: IALM-based matrix completion (high accuracy, slow)
               - lds: LDS debiased estimator (high speed, good accuracy)
        **kwargs: Method-specific parameters passed to sampler constructor

    Returns:
        BaseSampler instance

    Raises:
        ValueError: If method is not recognized
    """
    if method == "uniform":
        from .uniform.sampler import UniformSampler
        return UniformSampler(**kwargs)
    elif method == "leveraged":
        from .leveraged.sampler import LeveragedSampler
        return LeveragedSampler(**kwargs)
    elif method == "lds":
        from .leveraged.lds_sampler import LDSSampler
        return LDSSampler(**kwargs)
    else:
        raise ValueError(f"Unknown sampling method: {method}. Must be 'uniform', 'leveraged', or 'lds'.")


__all__ = ["BaseSampler", "get_sampler"]

