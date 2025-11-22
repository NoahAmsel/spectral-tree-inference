"""Create phase diagrams showing failure/success regions."""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))


def create_heatmap(df: pd.DataFrame, output_path: Path, metric: str = 'partition_agreement_M'):
    """
    Create phase diagram heatmap.

    Args:
        df: Full results DataFrame
        output_path: Where to save the plot
        metric: Metric to visualize (default: partition_agreement_M)
    """
    if metric not in df.columns:
        print(f"  {metric} not found in data")
        return

    # Get unique values for axes
    num_taxa_vals = sorted(df['num_taxa'].unique())
    seq_len_vals = sorted(df['sequence_length'].unique())
    p_vals = sorted(df['p'].unique())

    # Create one heatmap per num_taxa
    n_taxa = len(num_taxa_vals)
    fig, axes = plt.subplots(1, n_taxa, figsize=(6 * n_taxa, 5))
    if n_taxa == 1:
        axes = [axes]

    for idx, n in enumerate(num_taxa_vals):
        ax = axes[idx]

        # Filter data for this num_taxa
        df_n = df[df['num_taxa'] == n]

        # Create pivot table
        pivot = df_n.pivot_table(
            values=metric,
            index='sequence_length',
            columns='p',
            aggfunc='mean'
        )

        # Plot heatmap
        im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                       cmap='RdYlGn', vmin=0, vmax=100)

        # Set ticks
        ax.set_xticks(range(len(p_vals)))
        ax.set_xticklabels([f'{p:.2f}' for p in p_vals], rotation=45)
        ax.set_yticks(range(len(seq_len_vals)))
        ax.set_yticklabels(seq_len_vals)

        ax.set_xlabel('Sampling Probability (p)', fontsize=11)
        ax.set_ylabel('Sequence Length (L)', fontsize=11)
        ax.set_title(f'n={n} taxa', fontsize=12, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Partition Agreement (%)', fontsize=10)

    fig.suptitle('Phase Diagram: Success vs Failure Regions',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Create phase diagrams.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Empty dict (no statistics to return)
    """
    print("\n=== Creating Phase Diagrams ===")

    # Create heatmap
    create_heatmap(df, output_dir / "phase_diagram_heatmap.png")

    return {}
