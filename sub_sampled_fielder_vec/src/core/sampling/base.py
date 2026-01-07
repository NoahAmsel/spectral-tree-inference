"""Base abstract class for matrix samplers."""
from abc import ABC, abstractmethod
import numpy as np


class BaseSampler(ABC):
    """
    Abstract base class for matrix subsampling methods.
    
    All samplers must implement the sample() method which takes a full matrix
    and returns a subsampled or recovered version of it.
    """
    
    @abstractmethod
    def sample(self, matrix: np.ndarray, p: float, seed: int = None, **kwargs) -> np.ndarray:
        """
        Sample or recover a matrix from the full matrix.
        
        Args:
            matrix: Full symmetric similarity matrix (n x n)
            p: Sampling probability or budget parameter (0 < p <= 1)
            seed: Random seed for reproducibility
            **kwargs: Method-specific additional parameters
            
        Returns:
            Subsampled or recovered matrix (n x n) with diagonal = 1.0
        """
        pass

