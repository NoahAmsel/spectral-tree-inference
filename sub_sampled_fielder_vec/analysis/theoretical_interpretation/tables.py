"""Generate numeric summary tables for theoretical interpretation."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import find_all_transitions


def create_transition_summary_table(transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Create detailed table of metrics at each transition point.

    Args:
        transitions: DataFrame with transition points

    Returns:
        DataFrame with key metrics at each transition point
    """
    if len(transitions) == 0:
        return pd.DataFrame()

    # Determine which error column is available
    error_col = None
    if 'mean_operator_norm_error' in transitions.columns:
        error_col = 'mean_operator_norm_error'
    elif 'mean_frobenius_error' in transitions.columns:
        error_col = 'mean_frobenius_error'  # Legacy name (actually spectral norm)

    # Select columns to include in summary
    columns_to_keep = [
        'num_taxa', 'sequence_length', 'p',
        'partition_agreement_M', 'partition_agreement_S',
        'mean_spectral_gap_L_M', 'mean_spectral_gap_L_S',
        'mean_coherence_L_S',
        'mean_min_separation_L_S'
    ]

    if error_col:
        columns_to_keep.append(error_col)

    # Filter to columns that exist
    available_cols = [col for col in columns_to_keep if col in transitions.columns]

    summary = transitions[available_cols].copy()

    # Add computed columns
    if error_col and error_col in summary.columns and 'mean_spectral_gap_L_M' in summary.columns:
        summary['DK_bound_spectral'] = summary[error_col] / summary['mean_spectral_gap_L_M']

    if 'mean_spectral_gap_L_S' in summary.columns and 'mean_spectral_gap_L_M' in summary.columns:
        summary['gap_ratio_LS_LM'] = summary['mean_spectral_gap_L_S'] / summary['mean_spectral_gap_L_M']

    # Rename columns for clarity
    rename_map = {
        'p': 'p_critical',
        'mean_operator_norm_error': 'spectral_norm_error',
        'mean_frobenius_error': 'spectral_norm_error',  # Legacy name
        'mean_spectral_gap_L_M': 'gap_L_M',
        'mean_spectral_gap_L_S': 'gap_L_S',
        'mean_coherence_L_S': 'coherence_L_S',
        'mean_min_separation_L_S': 'min_sep_L_S'
    }
    summary = summary.rename(columns=rename_map)

    # Round for readability
    numeric_cols = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_cols] = summary[numeric_cols].round(6)

    return summary


def create_statistical_summary_table(transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate statistics table across all transitions.

    Args:
        transitions: DataFrame with transition points

    Returns:
        DataFrame with aggregate statistics (mean, std, min, max)
    """
    if len(transitions) == 0:
        return pd.DataFrame()

    # Determine which error column is available
    error_col = None
    if 'mean_operator_norm_error' in transitions.columns:
        error_col = 'mean_operator_norm_error'
    elif 'mean_frobenius_error' in transitions.columns:
        error_col = 'mean_frobenius_error'

    # Metrics to aggregate
    metrics = [
        'p',
        'mean_spectral_gap_L_M',
        'mean_spectral_gap_L_S',
        'mean_coherence_L_S',
        'mean_min_separation_L_S'
    ]

    if error_col:
        metrics.append(error_col)

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in transitions.columns]

    if not available_metrics:
        return pd.DataFrame()

    # Compute statistics
    stats_dict = {}

    for metric in available_metrics:
        values = transitions[metric].dropna()
        if len(values) > 0:
            stats_dict[metric] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median()
            }

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_dict).T
    stats_df.index.name = 'metric'

    # Rename index for clarity
    rename_map = {
        'p': 'p_critical',
        'mean_operator_norm_error': 'spectral_norm_error',
        'mean_frobenius_error': 'spectral_norm_error',  # Legacy name
        'mean_spectral_gap_L_M': 'gap_L_M',
        'mean_spectral_gap_L_S': 'gap_L_S',
        'mean_coherence_L_S': 'coherence_L_S',
        'mean_min_separation_L_S': 'min_sep_L_S'
    }
    stats_df = stats_df.rename(index=rename_map)

    # Round for readability
    stats_df = stats_df.round(6)

    return stats_df


def compute_davis_kahan_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Davis-Kahan bounds across all p values (not just transitions).

    Args:
        df: Full results DataFrame

    Returns:
        DataFrame with DK bounds for each (n, L, p) combination
    """
    # Determine which error column is available
    error_col = None
    if 'mean_operator_norm_error' in df.columns:
        error_col = 'mean_operator_norm_error'
    elif 'mean_frobenius_error' in df.columns:
        error_col = 'mean_frobenius_error'

    if error_col is None or 'mean_spectral_gap_L_M' not in df.columns:
        return pd.DataFrame()

    result = df[['num_taxa', 'sequence_length', 'p']].copy()

    # Compute DK bound (using spectral norm)
    gap = df['mean_spectral_gap_L_M'].replace(0, np.nan)
    result['DK_bound'] = df[error_col] / gap

    # Add partition agreement if available
    if 'partition_agreement_M' in df.columns:
        result['partition_agreement'] = df['partition_agreement_M']
    elif 'mean' in df.columns:
        result['partition_agreement'] = df['mean']

    # Add spectral gaps for context
    if 'mean_spectral_gap_L_S' in df.columns:
        result['gap_L_S'] = df['mean_spectral_gap_L_S']
    if 'mean_spectral_gap_L_M' in df.columns:
        result['gap_L_M'] = df['mean_spectral_gap_L_M']

    return result


def save_table_as_markdown(df: pd.DataFrame, output_path: Path, title: str = None):
    """
    Save DataFrame as markdown table.

    Args:
        df: DataFrame to save
        output_path: Path to save markdown file
        title: Optional title for the table
    """
    with open(output_path, 'w') as f:
        if title:
            f.write(f"# {title}\n\n")

        # Manually create markdown table (avoid tabulate dependency)
        # Write header
        if df.index.name:
            f.write(f"| {df.index.name} | ")
        else:
            f.write("| | ")
        f.write(" | ".join(df.columns) + " |\n")

        # Write separator
        if df.index.name:
            f.write("|" + "-" * (len(df.index.name) + 2) + "|")
        else:
            f.write("| --- |")
        f.write(" | ".join([" --- " for _ in df.columns]) + " |\n")

        # Write rows
        for idx, row in df.iterrows():
            if isinstance(idx, tuple):
                idx_str = ", ".join(str(i) for i in idx)
            else:
                idx_str = str(idx)
            f.write(f"| {idx_str} | ")
            f.write(" | ".join([f"{v:.6f}" if isinstance(v, (int, float, np.number)) else str(v) for v in row]) + " |\n")

        f.write("\n")
    print(f"  Saved: {output_path}")


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Generate all numeric summary tables.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Dictionary with table statistics
    """
    print("\n=== Generating Numeric Summary Tables ===")

    # Find transitions
    transitions = find_all_transitions(df, threshold=90.0)

    stats = {
        'n_transitions': len(transitions)
    }

    if len(transitions) == 0:
        print("  No transitions found")
        return stats

    # 1. Transition summary table
    print("  Creating transition summary table...")
    transition_table = create_transition_summary_table(transitions)

    # Save as CSV and markdown
    transition_table.to_csv(output_dir / "transition_summary_table.csv", index=False)
    save_table_as_markdown(
        transition_table.set_index(['num_taxa', 'sequence_length']),
        output_dir / "transition_summary_table.md",
        title="Transition Point Summary"
    )

    # 2. Statistical summary table
    print("  Creating statistical summary table...")
    stats_table = create_statistical_summary_table(transitions)

    # Save as CSV and markdown
    stats_table.to_csv(output_dir / "statistical_summary.csv")
    save_table_as_markdown(
        stats_table,
        output_dir / "statistical_summary.md",
        title="Statistical Summary Across All Transitions"
    )

    # Store key statistics
    if not stats_table.empty and 'mean' in stats_table.columns:
        stats['mean_p_critical'] = float(stats_table.loc['p_critical', 'mean']) if 'p_critical' in stats_table.index else None
        stats['mean_DK_bound'] = float(stats_table.loc['spectral_norm_error', 'mean'] / stats_table.loc['gap_L_M', 'mean']) if all(x in stats_table.index for x in ['spectral_norm_error', 'gap_L_M']) else None

    # 3. Full DK bounds table (for plotting)
    print("  Computing Davis-Kahan bounds for all p values...")
    dk_bounds = compute_davis_kahan_bounds(df)
    if not dk_bounds.empty:
        dk_bounds.to_csv(output_dir / "davis_kahan_bounds_full.csv", index=False)
        print(f"  Saved: {output_dir / 'davis_kahan_bounds_full.csv'}")

    print(f"  Generated {len(transitions)} transition rows")

    return stats
