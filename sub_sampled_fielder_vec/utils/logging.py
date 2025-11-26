"""Standardized logging and warning handlers."""
import warnings
from contextlib import contextmanager
from typing import Callable, Optional
from tqdm import tqdm
import os
from datetime import datetime

# Global display mode - set by experiment runner
_DISPLAY_MODE = "progress"  # "progress" or "debug"

# Global log file handle
_LOG_FILE = None
_LOG_FILE_PATH = None


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


def setup_log_file(run_dir: str):
    """
    Set up file-based logging for the experiment.

    Args:
        run_dir: Directory where log file will be created
    """
    global _LOG_FILE, _LOG_FILE_PATH

    # Close existing log file if any
    if _LOG_FILE is not None:
        _LOG_FILE.close()

    # Create log file path
    _LOG_FILE_PATH = os.path.join(run_dir, "experiment.log")

    # Open log file in append mode
    _LOG_FILE = open(_LOG_FILE_PATH, 'w')

    # Write header
    _LOG_FILE.write(f"{'='*80}\n")
    _LOG_FILE.write(f"Experiment Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FILE.write(f"{'='*80}\n\n")
    _LOG_FILE.flush()


def close_log_file():
    """Close the log file if open."""
    global _LOG_FILE, _LOG_FILE_PATH

    if _LOG_FILE is not None:
        _LOG_FILE.write(f"\n{'='*80}\n")
        _LOG_FILE.write(f"Experiment Log Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        _LOG_FILE.write(f"{'='*80}\n")
        _LOG_FILE.close()
        _LOG_FILE = None


def get_log_file_path() -> Optional[str]:
    """Get the path to the current log file."""
    return _LOG_FILE_PATH


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

    # Always redirect warnings to logging system (suppress in progress mode, log in debug mode)
    if mode == "progress":
        suppress_numerical_warnings()
    else:
        # In debug mode, capture and log warnings
        setup_warning_logging()


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


def setup_warning_logging():
    """
    Configure warnings to be logged through the logging system.
    
    This captures all RuntimeWarnings and formats them using our logging infrastructure.
    Useful for debug mode where we want to see warnings but keep them organized.
    """
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        """Custom warning handler that routes warnings through our logging system."""
        if category == RuntimeWarning:
            # Extract module name from filename
            module_parts = filename.split('/')
            if 'numpy' in filename:
                component = 'numpy'
            elif 'scipy' in filename:
                component = 'scipy'
            elif 'bootstrap_sweep' in filename:
                component = 'bootstrap'
            elif 'similarity' in filename or 'cache' in filename:
                component = 'cache'
            else:
                component = 'runtime'
            
            # Format: COMPONENT | WARNING | message [file:line]
            comp_name = COMPONENTS.get(component, component.upper())
            location = f"{module_parts[-1]}:{lineno}"
            msg_str = str(message).strip()
            print(f"{comp_name} | WARNING | {msg_str} [{location}]")
        else:
            # For non-RuntimeWarnings, use default handler
            if file is None:
                import sys
                file = sys.stderr
            try:
                file.write(warnings.formatwarning(message, category, filename, lineno, line))
            except (AttributeError, OSError):
                pass
    
    # Set our custom warning handler
    warnings.showwarning = warning_handler


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
    
    Behavior:
        - In progress mode: Suppress all RuntimeWarnings completely
        - In debug mode: Capture RuntimeWarnings and log them with proper formatting
    """
    with warnings.catch_warnings(record=True) as caught_warnings:
        # Configure warning filters based on display mode
        if is_progress_mode():
            # In progress mode, suppress (ignore) all RuntimeWarnings
            warnings.simplefilter("ignore", RuntimeWarning)
        else:
            # In debug mode, capture all warnings for logging
            warnings.simplefilter("always", RuntimeWarning)
        
        yield
        
        # In debug mode, log captured warnings through our logging system
        if not is_progress_mode() and caught_warnings:
            comp_name = COMPONENTS.get(component, component.upper())
            for w in caught_warnings:
                if issubclass(w.category, RuntimeWarning):
                    # Extract filename (last part of path)
                    filename_parts = w.filename.split('/')
                    location = f"{filename_parts[-1]}:{w.lineno}"
                    msg_str = str(w.message).strip()
                    print(f"{comp_name} | WARNING | {msg_str} [{location}]")


def log_info(component: str, message: str, force: bool = False):
    """
    Log informational message.

    Args:
        component: Component name
        message: Message to log
        force: If True, log even in progress mode (for important final messages)
    """
    comp_name = COMPONENTS.get(component, component.upper())
    log_line = f"{comp_name} | INFO | {message}"

    # Always write to log file if available
    if _LOG_FILE is not None:
        timestamp = datetime.now().strftime('%H:%M:%S')
        _LOG_FILE.write(f"[{timestamp}] {log_line}\n")
        _LOG_FILE.flush()

    # In progress mode, only show forced messages to stdout
    if is_progress_mode() and not force:
        return

    print(log_line)


def log_warning(component: str, message: str, force: bool = False):
    """
    Log warning message.

    Args:
        component: Component name
        message: Message to log
        force: If True, log even in progress mode
    """
    comp_name = COMPONENTS.get(component, component.upper())
    log_line = f"{comp_name} | WARNING | {message}"

    # Always write to log file if available
    if _LOG_FILE is not None:
        timestamp = datetime.now().strftime('%H:%M:%S')
        _LOG_FILE.write(f"[{timestamp}] {log_line}\n")
        _LOG_FILE.flush()

    # In progress mode, only show forced warnings to stdout
    if is_progress_mode() and not force:
        return

    print(log_line)


def log_error(component: str, message: str):
    """Log error message (always shown)."""
    comp_name = COMPONENTS.get(component, component.upper())
    log_line = f"{comp_name} | ERROR | {message}"

    # Always write to log file if available
    if _LOG_FILE is not None:
        timestamp = datetime.now().strftime('%H:%M:%S')
        _LOG_FILE.write(f"[{timestamp}] {log_line}\n")
        _LOG_FILE.flush()

    print(log_line)


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

