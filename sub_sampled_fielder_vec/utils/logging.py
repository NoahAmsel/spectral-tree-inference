"""Standardized logging and warning handlers."""
import warnings
from contextlib import contextmanager
from typing import Callable, Optional
from tqdm import tqdm

# Global display mode - set by experiment runner
_DISPLAY_MODE = "progress"  # "progress" or "debug"


class DummyProgressBar:
    """No-op progress bar for debug mode - implements tqdm interface but does nothing."""
    
    def __init__(self, *args, **kwargs):
        """Accept any arguments to match tqdm signature."""
        pass
    
    def update(self, n=1):
        """No-op update method."""
        pass
    
    def close(self):
        """No-op close method."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        pass


def set_display_mode(mode: str):
    """
    Set the global display mode.

    Args:
        mode: Either "progress" (clean progress bars) or "debug" (verbose logging)
    """
    global _DISPLAY_MODE
    if mode not in ["progress", "debug"]:
        raise ValueError(f"Invalid display mode: {mode}. Must be 'progress' or 'debug'")
    _DISPLAY_MODE = mode

    # In progress mode, suppress numpy/sklearn warnings
    if mode == "progress":
        suppress_numerical_warnings()


def get_display_mode() -> str:
    """Get the current display mode."""
    return _DISPLAY_MODE


def is_progress_mode() -> bool:
    """Check if we're in progress bar mode (clean output)."""
    return _DISPLAY_MODE == "progress"


def suppress_numerical_warnings():
    """Suppress common numerical warnings from numpy and sklearn."""
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.*')


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


def log_info(component: str, message: str, force: bool = False):
    """
    Log informational message.

    Args:
        component: Component name
        message: Message to log
        force: If True, log even in progress mode (for important final messages)
    """
    # In progress mode, only show forced messages
    if is_progress_mode() and not force:
        return

    comp_name = COMPONENTS.get(component, component.upper())
    print(f"{comp_name} | INFO | {message}")


def log_warning(component: str, message: str, force: bool = False):
    """
    Log warning message.

    Args:
        component: Component name
        message: Message to log
        force: If True, log even in progress mode
    """
    # In progress mode, only show forced warnings
    if is_progress_mode() and not force:
        return

    comp_name = COMPONENTS.get(component, component.upper())
    print(f"{comp_name} | WARNING | {message}")


def log_error(component: str, message: str):
    """Log error message (always shown)."""
    comp_name = COMPONENTS.get(component, component.upper())
    print(f"{comp_name} | ERROR | {message}")


def create_progress_bar(total: int, desc: str, unit: str = 'it',
                        leave: bool = True, position: Optional[int] = None) -> tqdm:
    """
    Create a standardized progress bar.

    Args:
        total: Total number of iterations
        desc: Description for the progress bar
        unit: Unit name for iterations (default: 'it')
        leave: Whether to leave the progress bar after completion
        position: Position for nested progress bars (0=outermost, 1=nested, etc.)

    Returns:
        tqdm progress bar object or DummyProgressBar in debug mode
    """
    # In debug mode, return a dummy progress bar (no visual output)
    if not is_progress_mode():
        return DummyProgressBar()
    
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        position=position,
        ncols=100,  # Fixed width for consistency
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )


def create_config_progress_bar(config_idx: int, total_configs: int,
                               n_taxa: int, seq_len: int,
                               num_p_values: int, position: int) -> tqdm:
    """
    Create a configuration-level progress bar for tracking p-value computation.

    Format: "Conf 1/4 L=500 n=1025: p-values 0%|...| 0/21"

    Args:
        config_idx: Current configuration index (1-based)
        total_configs: Total number of configurations
        n_taxa: Number of taxa (n)
        seq_len: Sequence length (L)
        num_p_values: Total number of p-values to compute
        position: Vertical position for this progress bar

    Returns:
        tqdm progress bar object or DummyProgressBar in debug mode
    """
    # In debug mode, return a dummy progress bar (no visual output)
    if not is_progress_mode():
        return DummyProgressBar()
    
    desc = f"Conf {config_idx}/{total_configs} L={seq_len:>5} n={n_taxa:>5}: p-values"
    return tqdm(
        total=num_p_values,
        desc=desc,
        unit='p',
        leave=True,
        position=position,
        ncols=100,
        bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
    )

