"""Interactive UI components for STDR launcher.

Provides colored text, ASCII art logo, and input helpers for the interactive launcher.
"""
import sys
from typing import Optional, List, Dict, Any


# ANSI color codes for gradient effects
class Colors:
    """ANSI escape codes for colored terminal output."""
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    @staticmethod
    def gradient(text: str, colors: List[str]) -> str:
        """Apply color gradient to text line-by-line."""
        lines = text.split('\n')
        if len(lines) <= 1:
            return colors[0] + text + Colors.RESET

        colored_lines = []
        for i, line in enumerate(lines):
            color_idx = int((i / (len(lines) - 1)) * (len(colors) - 1))
            colored_lines.append(colors[color_idx] + line + Colors.RESET)

        return '\n'.join(colored_lines)


def print_logo():
    """Print Gemini-style STDR logo with gradient."""
    logo = """
    ███████╗████████╗██████╗ ██████╗
    ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗
    ███████╗   ██║   ██║  ██║██████╔╝
    ╚════██║   ██║   ██║  ██║██╔══██╗
    ███████║   ██║   ██████╔╝██║  ██║
    ╚══════╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝
    """

    subtitle = "    Subsampled Spectral Tree Recovery"

    # Apply gradient: blue → cyan → magenta
    gradient_colors = [Colors.BLUE, Colors.CYAN, Colors.MAGENTA]
    print(Colors.gradient(logo, gradient_colors))
    print(Colors.CYAN + subtitle + Colors.RESET)
    print()


def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{text}{Colors.RESET}")
    print("─" * len(text))


def print_option(key: str, description: str, highlight: bool = False):
    """Print menu option."""
    if highlight:
        print(f"  {Colors.GREEN}[{key}]{Colors.RESET} {Colors.BOLD}{description}{Colors.RESET}")
    else:
        print(f"  {Colors.CYAN}[{key}]{Colors.RESET} {description}")


def print_cached_matrix(index: int, metadata: Dict[str, Any]):
    """Print cached matrix info in compact format."""
    n = metadata.get('n_taxa', '?')
    L = metadata.get('seq_len', '?')
    mu = metadata.get('mutation_rate', '?')
    model = metadata.get('tree_model', '?')

    print(f"  {Colors.CYAN}[{index}]{Colors.RESET} n={n}, L={L}, μ={mu}, {model}")


def print_config_summary(config: Dict[str, Any], prefix: str = "  →"):
    """Print config summary in one line."""
    n = config.get('n_taxa', config.get('taxa_values', [None])[0])
    L = config.get('seq_len', config.get('sequence_length_values', [None])[0])
    mu = config.get('mutation_rate', '?')
    model = config.get('tree_model', '?')
    bootstrap_reps = config.get('bootstrap_reps', '?')

    print(f"{prefix} n={n}, L={L}, μ={mu}, {model}, {bootstrap_reps} bootstraps")


def get_input(prompt: str, default: Optional[str] = None, color: str = Colors.CYAN) -> str:
    """Get user input with optional default value."""
    if default is not None:
        full_prompt = f"{prompt} {Colors.BOLD}[{default}]{Colors.RESET}: "
    else:
        full_prompt = f"{prompt}: "

    try:
        user_input = input(color + full_prompt + Colors.RESET).strip()
        return user_input if user_input else default
    except (KeyboardInterrupt, EOFError):
        print(f"\n{Colors.RED}Interrupted by user{Colors.RESET}")
        sys.exit(0)


def get_choice(prompt: str, valid_choices: List[str], case_sensitive: bool = False) -> str:
    """Get user choice from valid options."""
    while True:
        choice = get_input(prompt)

        if choice is None:
            continue

        if not case_sensitive:
            choice = choice.lower()
            valid_choices = [c.lower() for c in valid_choices]

        if choice in valid_choices:
            return choice

        print(f"{Colors.RED}Not an option, try again. Valid choices: {', '.join(valid_choices)}{Colors.RESET}")


def confirm(prompt: str = "Continue?", default: bool = True) -> bool:
    """Ask for yes/no confirmation."""
    default_str = "Y/n" if default else "y/N"

    while True:
        response = get_input(f"{prompt} [{default_str}]", default="y" if default else "n")

        if response is None:
            return default

        response_lower = response.lower()
        if response_lower in ['y', 'yes']:
            return True
        elif response_lower in ['n', 'no']:
            return False
        else:
            print(f"{Colors.RED}Not an option, try again. Enter 'y' or 'n'{Colors.RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗ Error: {message}{Colors.RESET}")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def clear_screen():
    """Clear terminal screen (optional, for cleaner UX)."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_divider():
    """Print a visual divider."""
    print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")
