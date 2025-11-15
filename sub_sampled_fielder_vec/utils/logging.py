"""Standardized logging and warning handlers."""
import warnings
from contextlib import contextmanager
from typing import Callable


# Component names for standardized logging
COMPONENTS = {
    'fiedler': 'COMPUTE_FIEDLER_VECTOR',
    'align': 'ALIGN_FIEDLER_VECTOR',
    'metrics': 'METRIC_COMPOSER',
    'similarity': 'SIMILARITY_COMPUTE',
    'laplacian': 'COMPUTE_LAPLACIAN',
    'experiment': 'EXPERIMENT_RUNNER',
    'bootstrap': 'BOOTSTRAP_SWEEP',
    'cache': 'CACHE_MANAGER'
}


def get_warning_handler(component: str) -> Callable:
    """
    Create a standardized warning handler for a component.
    
    Args:
        component: Component name (from COMPONENTS dict or custom)
        
    Returns:
        Warning handler function
    """
    comp_name = COMPONENTS.get(component, component.upper())
    
    def handler(msg, cat, filename, lineno, file=None, line=None):
        if cat == RuntimeWarning:
            print(f"{comp_name} | WARNING | {str(msg).strip()}")
    
    return handler


@contextmanager
def suppress_warnings(component: str):
    """
    Context manager for suppressing and logging warnings.
    
    Usage:
        with suppress_warnings('fiedler'):
            eigvals, eigvecs = scipy.linalg.eigh(L)
    
    Args:
        component: Component name for logging
    """
    original_showwarning = warnings.showwarning
    warnings.showwarning = get_warning_handler(component)
    try:
        yield
    finally:
        warnings.showwarning = original_showwarning


def log_info(component: str, message: str):
    """Log informational message."""
    comp_name = COMPONENTS.get(component, component.upper())
    print(f"{comp_name} | INFO | {message}")


def log_warning(component: str, message: str):
    """Log warning message."""
    comp_name = COMPONENTS.get(component, component.upper())
    print(f"{comp_name} | WARNING | {message}")


def log_error(component: str, message: str):
    """Log error message."""
    comp_name = COMPONENTS.get(component, component.upper())
    print(f"{comp_name} | ERROR | {message}")

