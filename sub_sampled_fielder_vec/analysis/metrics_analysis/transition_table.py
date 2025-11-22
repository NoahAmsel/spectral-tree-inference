"""Create and save transition threshold table."""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import find_all_transitions


def create_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Create comprehensive table of transition points.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save table

    Returns:
        Transition table DataFrame
    """
    print("\n=== Creating Transition Table ===")

    transitions = find_all_transitions(df, threshold=90.0)

    if len(transitions) == 0:
        print("  No transitions found!")
        return pd.DataFrame()

    # Select key columns - use partition_agreement_M if available, else mean
    agreement_col = 'partition_agreement_M' if 'partition_agreement_M' in transitions.columns else 'mean'
    columns = [
        'num_taxa', 'sequence_length', 'p', agreement_col,
        'spectral_gap_ratio', 'rank_ratio_L_S',
        'mean_frobenius_error', 'mean_coherence_L_S',
        'mean_min_separation_L_S', 'effective_samples'
    ]

    trans_table = transitions[columns].copy()
    trans_table = trans_table.sort_values(['sequence_length', 'num_taxa'])

    # Save as CSV
    csv_path = output_dir / "transition_thresholds.csv"
    trans_table.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved: {csv_path}")

    return trans_table
