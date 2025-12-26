"""Generate eigenvalue scree plots for diagnostic visualization."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple


PREFERRED_MODELS = ("balanced_binary", "kingman_mean")
MODEL_TITLES = {
    "balanced_binary": "Balanced Binary",
    "kingman_mean": "Kingman Mean"
}


def _plot_scree_by_model_and_mu(
    model_entries: Dict[str, List[Tuple[str, Dict]]],
    eigenvalue_key: str,
    figure_title: str,
    ylabel: str,
    output_path: Path
) -> None:
    """
    Draw scree plots with columns for preferred models and rows per μ value.

    Each subplot shows the eigenvalue curve for a specific model × μ combination.
    """
    if not model_entries:
        raise ValueError("No model entries provided for plotting.")

    extra_models = sorted(set(model_entries.keys()) - set(PREFERRED_MODELS))
    if extra_models:
        print(
            f"⚠️ Skipping models not in preferred layout: {', '.join(extra_models)}"
        )

    mu_values = sorted(
        {
            round(result['mu'], 10)
            for entries in model_entries.values()
            for _, result in entries
        }
    )
    if not mu_values:
        raise ValueError("No μ values available for scree plotting.")

    n_rows = len(mu_values)
    n_cols = len(PREFERRED_MODELS)
    figsize = (6 * n_cols, 3.5 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharex=False,
        sharey=False
    )

    legend_handles = []
    legend_labels = []
    legend_label_set = set()

    ordered_labels: List[str] = []
    for model_name in PREFERRED_MODELS:
        entries = model_entries.get(model_name, [])
        entries.sort(key=lambda pair: (pair[1]['n'], pair[1]['L'], pair[1]['mu']))
        for label, result in entries:
            if label not in ordered_labels:
                ordered_labels.append(label)

    if ordered_labels:
        color_positions = np.linspace(0.1, 0.9, len(ordered_labels))
        cmap = plt.cm.get_cmap('viridis')
        label_to_color = {
            label: cmap(color_positions[idx])
            for idx, label in enumerate(ordered_labels)
        }
    else:
        label_to_color = {}

    for row_idx, mu in enumerate(mu_values):
        for col_idx, model_name in enumerate(PREFERRED_MODELS):
            ax = axes[row_idx][col_idx]
            ax.set_xlabel('Eigenvalue Index', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)

            pretty_model = MODEL_TITLES.get(model_name, model_name.replace("_", " ").title())
            ax.set_title(f"{pretty_model} | μ={mu:.2f}", fontsize=12, fontweight='bold')

            entries = [
                (label, result)
                for label, result in model_entries.get(model_name, [])
                if abs(result['mu'] - mu) < 1e-10
            ]
            entries.sort(key=lambda pair: (pair[1]['n'], pair[1]['L']))

            if not entries:
                ax.text(
                    0.5,
                    0.5,
                    'No data',
                    ha='center',
                    va='center',
                    fontsize=11,
                    alpha=0.6,
                    transform=ax.transAxes
                )
                ax.grid(True, alpha=0.3)
                continue

            for label, result in entries:
                eigenvalues = result[eigenvalue_key]
                indices = np.arange(1, len(eigenvalues) + 1)
                color = label_to_color.get(label)
                line, = ax.plot(
                    indices,
                    eigenvalues,
                    'o-',
                    label=label,
                    color=color,
                    linewidth=2,
                    markersize=5,
                    alpha=0.85
                )
                if label not in legend_label_set:
                    legend_label_set.add(label)
                    legend_handles.append(line)
                    legend_labels.append(label)
            ax.grid(True, alpha=0.3)

    fig.suptitle(figure_title, fontsize=16, fontweight='bold')

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            fontsize=9,
            ncol=min(len(legend_labels), 4),
            bbox_to_anchor=(0.5, -0.02)
        )
        fig.subplots_adjust(bottom=0.15)

    plt.tight_layout(rect=(0.02, 0.08, 0.98, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Scree plot saved to: {output_path}")


def plot_scree(
    eigenvalues: np.ndarray,
    title: str,
    ylabel: str,
    output_path: Path,
    figsize: tuple = (10, 6)
) -> None:
    """
    Create eigenvalue scree plot.

    Args:
        eigenvalues: Array of eigenvalues to plot
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    k = len(eigenvalues)
    indices = np.arange(1, k + 1)

    plt.figure(figsize=figsize)
    plt.plot(indices, eigenvalues, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Eigenvalue Index', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Scree plot saved to: {output_path}")


def generate_all_scree_plots(results: List[Dict], output_dir: Path) -> None:
    """
    Generate scree plots with two subplots per figure (balanced vs kingman).

    Two figures are produced:
        1. Similarity matrix M eigenvalues side-by-side
        2. Laplacian L_M eigenvalues side-by-side
    """
    if not results:
        raise ValueError("No results supplied for scree plotting.")

    model_entries: Dict[str, List[Tuple[str, Dict]]] = {}
    for result in results:
        label = f"n={result['n']}, L={result['L']}, μ={result['mu']:.2f}"
        model_entries.setdefault(result['tree_model'], []).append((label, result))

    _plot_scree_by_model_and_mu(
        model_entries=model_entries,
        eigenvalue_key='M_eigenvalues',
        figure_title='Similarity Matrix M - μ-Stratified Comparison',
        ylabel='Eigenvalue Magnitude',
        output_path=output_dir / "A_similarity_eigenvalues.png"
    )

    _plot_scree_by_model_and_mu(
        model_entries=model_entries,
        eigenvalue_key='L_M_eigenvalues',
        figure_title='Laplacian L_M - μ-Stratified Comparison',
        ylabel='Eigenvalue Magnitude',
        output_path=output_dir / "B_laplacian_eigenvalues.png"
    )


def plot_comparison_scree(
    results: List[Dict],
    output_dir: Path,
    group_by: str = 'model'
) -> None:
    """
    Create comparison scree plots with balanced vs kingman displayed side-by-side.

    Args:
        results: List of diagnostic result dictionaries
        output_dir: Directory to save plots
        group_by: How to group comparisons ('model', 'n', 'L', or 'mu')
    """
    if not results:
        raise ValueError("No results provided for comparison scree plotting.")

    if group_by == 'model':
        model_entries = {}
        for result in results:
            label = f"n={result['n']}, L={result['L']}, μ={result['mu']:.2f}"
            model_entries.setdefault(result['tree_model'], []).append((label, result))

        _plot_scree_by_model_and_mu(
            model_entries=model_entries,
            eigenvalue_key='M_eigenvalues',
            figure_title='Similarity Matrix - Cross-Model Comparison',
            ylabel='Eigenvalue Magnitude',
            output_path=output_dir / "D_comparison_similarity.png"
        )
        _plot_scree_by_model_and_mu(
            model_entries=model_entries,
            eigenvalue_key='L_M_eigenvalues',
            figure_title='Laplacian - Cross-Model Comparison',
            ylabel='Eigenvalue Magnitude',
            output_path=output_dir / "E_comparison_laplacian.png"
        )
        return

    groups = {}
    for result in results:
        if group_by == 'n':
            key = result['n']
            label = f"n={result['n']} ({result['tree_model']})"
        elif group_by == 'L':
            key = result['L']
            label = f"L={result['L']} ({result['tree_model']})"
        elif group_by == 'mu':
            key = result['mu']
            label = f"μ={result['mu']:.2f} ({result['tree_model']})"
        else:
            raise ValueError(f"Invalid group_by parameter: {group_by}")

        groups.setdefault(key, []).append((label, result))

    for group_key, group_results in groups.items():
        model_entries: Dict[str, List[Tuple[str, Dict]]] = {}
        for label, result in group_results:
            model_entries.setdefault(result['tree_model'], []).append((label, result))

        title_suffix = f"(Grouped by {group_by}: {group_key})"

        _plot_scree_by_model_and_mu(
            model_entries=model_entries,
            eigenvalue_key='M_eigenvalues',
            figure_title=f'Similarity Matrix - {title_suffix}',
            ylabel='Eigenvalue Magnitude',
            output_path=output_dir / f"X_similarity_{group_by}_{group_key}.png"
        )

        _plot_scree_by_model_and_mu(
            model_entries=model_entries,
            eigenvalue_key='L_M_eigenvalues',
            figure_title=f'Laplacian - {title_suffix}',
            ylabel='Eigenvalue Magnitude',
            output_path=output_dir / f"Y_laplacian_{group_by}_{group_key}.png"
        )

