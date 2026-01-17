"""Result container for IALM solver with diagnostic information."""


class IALMResult:
    """Result container for IALM solver with diagnostic information."""
    
    def __init__(self, L, S, converged: bool, iterations: int, had_numerical_issues: bool):
        self.L = L
        self.S = S
        self.converged = converged
        self.iterations = iterations
        self.had_numerical_issues = had_numerical_issues
