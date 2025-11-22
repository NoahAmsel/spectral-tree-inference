"""Find transition points where partition agreement crosses threshold."""

import pandas as pd
from typing import Optional


def find_transition_point(
    df: pd.DataFrame,
    num_taxa: int,
    sequence_length: int,
    threshold: float = 90.0,
    metric: str = 'partition_agreement_M'
) -> Optional[pd.Series]:
    """
    Find transition point for a specific (n, L) combination.

    Args:
        df: DataFrame with results
        num_taxa: Number of taxa (n)
        sequence_length: Sequence length (L)
        threshold: Partition agreement threshold (default 90%)
        metric: Column name for metric (default 'partition_agreement_M')

    Returns:
        Row at transition point, or None if no transition found
    """
    subset = df[(df['num_taxa'] == num_taxa) & (df['sequence_length'] == sequence_length)]
    subset_sorted = subset.sort_values('p')

    # Fallback to 'mean' if partition_agreement_M not available
    if metric not in df.columns and metric == 'partition_agreement_M' and 'mean' in df.columns:
        metric = 'mean'

    # Find first point where metric >= threshold
    transition = subset_sorted[subset_sorted[metric] >= threshold]

    if len(transition) > 0:
        return transition.iloc[0]
    return None


def find_all_transitions(
    df: pd.DataFrame,
    threshold: float = 90.0,
    metric: str = 'partition_agreement_M'
) -> pd.DataFrame:
    """
    Find transition points for all (num_taxa, sequence_length) combinations.

    Args:
        df: DataFrame with results
        threshold: Partition agreement threshold (default 90%)
        metric: Column name for metric (default 'partition_agreement_M')

    Returns:
        DataFrame with all transition points
    """
    transitions = []

    for num_taxa in df['num_taxa'].unique():
        for seq_len in df['sequence_length'].unique():
            trans_point = find_transition_point(df, num_taxa, seq_len, threshold, metric)
            if trans_point is not None:
                transitions.append(trans_point)

    if transitions:
        return pd.DataFrame(transitions)
    return pd.DataFrame()
