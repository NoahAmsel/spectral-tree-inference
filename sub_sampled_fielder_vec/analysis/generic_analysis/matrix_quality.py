"""Matrix quality metrics extraction utilities."""
import pandas as pd
import numpy as np


def extract_numerical_rank(df: pd.DataFrame, matrix_name: str = "L_S") -> pd.DataFrame:
    """Extract numerical rank metrics for specified matrix.
    
    Args:
        df: DataFrame with columns like 'mean_empirical_rank_L_S'
        matrix_name: Matrix identifier ('L_S', 'L_M', 'S', 'M')
        
    Returns:
        DataFrame with columns: 'p', 'numerical_rank'
    """
    rank_col = f'mean_empirical_rank_{matrix_name}'
    
    if rank_col not in df.columns:
        return pd.DataFrame({
            'p': df['p'].values,
            'numerical_rank': np.full(len(df), np.nan),
        })
    
    return pd.DataFrame({
        'p': df['p'].values,
        'numerical_rank': df[rank_col].values,
    })


def extract_coherence(df: pd.DataFrame, matrix_name: str = "L_S") -> pd.DataFrame:
    """Extract coherence metrics for specified matrix.
    
    Args:
        df: DataFrame with columns like 'mean_coherence_L_S'
        matrix_name: Matrix identifier ('L_S', 'L_M', 'S', 'M')
        
    Returns:
        DataFrame with columns: 'p', 'coherence'
    """
    coherence_col = f'mean_coherence_{matrix_name}'
    
    if coherence_col not in df.columns:
        return pd.DataFrame({
            'p': df['p'].values,
            'coherence': np.full(len(df), np.nan),
        })
    
    return pd.DataFrame({
        'p': df['p'].values,
        'coherence': df[coherence_col].values,
    })
