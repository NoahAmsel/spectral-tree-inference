"""Eigenvalue spectrum analysis utilities."""
import pandas as pd
import numpy as np


def extract_eigenvalues(df: pd.DataFrame, matrix_name: str = "L_S") -> pd.DataFrame:
    """Extract lambda_2, lambda_3 columns for specified matrix.
    
    Args:
        df: DataFrame with columns like 'mean_lambda2_L_S', 'mean_lambda3_L_S'
        matrix_name: Matrix identifier ('L_S', 'L_M', 'S', 'M')
        
    Returns:
        DataFrame with columns: 'p', 'lambda2', 'lambda3', 'spectral_gap'
    """
    lambda2_col = f'mean_lambda2_{matrix_name}'
    lambda3_col = f'mean_lambda3_{matrix_name}'
    gap_col = f'mean_spectral_gap_{matrix_name}'
    
    result = pd.DataFrame({
        'p': df['p'].values,
        'lambda2': df[lambda2_col].values,
        'lambda3': df[lambda3_col].values,
        'spectral_gap': df[gap_col].values if gap_col in df.columns else np.nan,
    })
    
    return result


def compute_relative_gap(lambda2: float, lambda3: float) -> float:
    """Compute relative spectral gap: (lambda3 - lambda2) / lambda2.
    
    Args:
        lambda2: Second smallest eigenvalue (Fiedler eigenvalue)
        lambda3: Third smallest eigenvalue
        
    Returns:
        Relative gap. Returns inf if lambda2 is zero or negative.
    """
    if lambda2 <= 0 or np.isnan(lambda2) or np.isnan(lambda3):
        return np.nan
    
    gap = lambda3 - lambda2
    if gap < 0:
        return np.nan  # Invalid: lambda3 < lambda2
    
    return gap / lambda2
