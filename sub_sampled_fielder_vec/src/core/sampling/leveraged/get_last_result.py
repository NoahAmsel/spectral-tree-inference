"""Get diagnostic information from the last IALM solve."""


def get_last_result():
    """Get diagnostic information from the last IALM solve."""
    from .ialm_solve import ialm_solve
    return getattr(ialm_solve, '_last_result', None)
