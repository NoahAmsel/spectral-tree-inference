"""Generate coherence comparison plots."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple


PREFERRED_MODELS = ("balanced_binary", "kingman_mean")
MODEL_TITLES = {
    "balanced_binary": "Balanced Binary",
    "kingman_mean": "Kingman Mean"
}


def plot_coherence_comparison(results: List[Dict], output_dir: Path) -> None:
    """
    Plot coherence values across configurations with two subplots (balanced vs kingman).

    Creates a bar chart showing coherence for each configuration, with lower values
    (better incoherence) shown in green and higher values in red.
    Each subplot has independent y-axis scale.

    Args:
        results: List of diagnostic result dictionaries
        output_dir: Directory to save plots
    """
    if not results:
        raise ValueError("No results supplied for coherence plotting.")

    # Group by model
    model_entries: Dict[str, List[Tuple[str, Dict]]] = {}
    for result in results:
        label = f"n={result['n']}, L={result['L']}, μ={result['mu']:.2f}"
        model_entries.setdefault(result['tree_model'], []).append((label, result))

    # Check for extra models
    extra_models = sorted(set(model_entries.keys()) - set(PREFERRED_MODELS))
    if extra_models:
        print(
            f"⚠️ Skipping models not in preferred layout: {', '.join(extra_models)}"
        )

    fig, axes = plt.subplots(1, len(PREFERRED_MODELS), figsize=(14, 6), sharey=False)
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    for idx, model_name in enumerate(PREFERRED_MODELS):
        ax = axes[idx]
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Coherence', fontsize=12)

        title = MODEL_TITLES.get(model_name, model_name.replace("_", " ").title())
        ax.set_title(title, fontsize=13, fontweight='bold')

        entries = model_entries.get(model_name, [])
        entries.sort(key=lambda pair: (pair[1]['n'], pair[1]['L'], pair[1]['mu']))

        if not entries:
            ax.text(
                0.5,
                0.5,
                'No data for this model',
                ha='center',
                va='center',
                fontsize=12,
                alpha=0.6,
                transform=ax.transAxes
            )
            ax.grid(True, alpha=0.3)
            continue

        # Extract coherence values and labels
        coherences = [result['coherence'] for _, result in entries]
        labels = [label for label, _ in entries]
        x_pos = np.arange(len(coherences))

        # Bar plot
        bars = ax.bar(x_pos, coherences, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Color bars by value (gradient from green to red)
        max_coherence = max(coherences) if coherences else 1.0
        for i, (bar, coh) in enumerate(zip(bars, coherences)):
            # Lower coherence is better (greener), higher is worse (redder)
            color_val = coh / max_coherence
            bar.set_color(plt.cm.RdYlGn_r(color_val))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on top of bars
        for i, coh in enumerate(coherences):
            ax.text(i, coh, f'{coh:.4f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Matrix Coherence Comparison', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    output_path = output_dir / "C_coherence.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Coherence plot saved to: {output_path}")

