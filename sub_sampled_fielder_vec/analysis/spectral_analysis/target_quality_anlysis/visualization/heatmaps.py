"""Generate metric heatmaps for diagnostic visualization."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Any, Iterable, Optional


PREFERRED_MODELS = ("balanced_binary", "kingman_mean")
MODEL_TITLES = {
    "balanced_binary": "Balanced Binary",
    "kingman_mean": "Kingman Mean"
}


def _group_by_model_and_mu(results: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, float], List[Dict]]:
    """Group diagnostic results by (tree_model, μ)."""
    grouped: Dict[Tuple[str, float], List[Dict]] = {}
    for result in results:
        key = (result['tree_model'], float(result['mu']))
        grouped.setdefault(key, []).append(result)
    return grouped


def _prepare_heatmap_grid(
    entries: List[Dict[str, Any]],
    metric_key: str
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Build an n × L grid for the requested metric.

    Missing (n, L) combinations are filled with NaN so the layout is stable.
    """
    n_values = sorted({entry['n'] for entry in entries})
    L_values = sorted({entry['L'] for entry in entries})

    if not n_values or not L_values:
        raise ValueError("Entries must contain at least one n and L value.")

    n_index = {n_val: idx for idx, n_val in enumerate(n_values)}
    L_index = {L_val: idx for idx, L_val in enumerate(L_values)}

    grid = np.full((len(n_values), len(L_values)), np.nan)
    for entry in entries:
        row_idx = n_index[entry['n']]
        col_idx = L_index[entry['L']]
        grid[row_idx, col_idx] = entry.get(metric_key)

    return grid, n_values, L_values


def plot_metric_heatmaps(
    results: List[Dict],
    output_dir: Path,
    metric_key: str,
    metric_label: str,
    output_prefix: str,
    cmap: str = 'viridis',
    value_format: str = ".2f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> Path:
    """
    Draw a grid of heatmaps for a metric across n × L configurations.

    One figure is created per metric, with rows grouped by model and
    columns grouped by mutation rate μ. Each subplot contains an n × L
    heatmap for that (model, μ) combination.
    """
    if not results:
        raise ValueError("No results provided for heatmap plotting.")

    grouped = _group_by_model_and_mu(results)
    if not grouped:
        raise ValueError("No (model, μ) groups could be formed for heatmaps.")

    model_order: List[str] = [
        model for model in PREFERRED_MODELS if any(key[0] == model for key in grouped.keys())
    ]
    # Include any other models at the end to avoid silently dropping data.
    extra_models = sorted(
        {key[0] for key in grouped.keys()} - set(model_order)
    )
    model_order.extend(extra_models)

    mu_values = sorted({key[1] for key in grouped.keys()})

    model_norms: Dict[str, plt.Normalize] = {}
    for model_name in model_order:
        model_values = [
            entry.get(metric_key)
            for (m_name, _), entries in grouped.items()
            if m_name == model_name
            for entry in entries
            if entry.get(metric_key) is not None
        ]
        if not model_values:
            continue
        local_min = vmin if vmin is not None else min(model_values)
        local_max = vmax if vmax is not None else max(model_values)
        if np.isclose(local_min, local_max):
            # Avoid zero-range color scales
            local_min -= 0.5
            local_max += 0.5
        model_norms[model_name] = plt.Normalize(vmin=local_min, vmax=local_max)

    n_rows = max(1, len(model_order))
    n_cols = max(1, len(mu_values))

    figsize = (n_cols * 4.0, n_rows * 3.5)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharex=False,
        sharey=False
    )

    for row_idx, model_name in enumerate(model_order):
        for col_idx, mu in enumerate(mu_values):
            ax = axes[row_idx][col_idx]
            entries = grouped.get((model_name, mu))
            if not entries:
                ax.set_axis_off()
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
                continue

            norm = model_norms.get(model_name)
            if norm is None:
                ax.set_axis_off()
                continue

            grid, n_values, L_values = _prepare_heatmap_grid(entries, metric_key)
            im = ax.imshow(
                grid,
                cmap=cmap,
                aspect='auto',
                norm=norm,
                origin='upper'
            )

            ax.set_xticks(np.arange(len(L_values)))
            ax.set_xticklabels([str(L) for L in L_values], rotation=45, ha='right', fontsize=9)
            ax.set_yticks(np.arange(len(n_values)))
            ax.set_yticklabels([str(n_val) for n_val in n_values], fontsize=9)

            if row_idx == n_rows - 1:
                ax.set_xlabel('L', fontsize=11)
            if col_idx == 0:
                ax.set_ylabel('n', fontsize=11)

            pretty_model = MODEL_TITLES.get(model_name, model_name.replace("_", " ").title())
            title = f"{pretty_model}\nμ={mu:.2f}"
            ax.set_title(title, fontsize=12, fontweight='bold')

            for i in range(len(n_values)):
                for j in range(len(L_values)):
                    value = grid[i, j]
                    if np.isnan(value):
                        continue
                    ax.text(
                        j,
                        i,
                        format(value, value_format),
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='black'
                    )

    for row_idx, model_name in enumerate(model_order):
        norm = model_norms.get(model_name)
        if norm is None:
            continue
        pretty_model = MODEL_TITLES.get(model_name, model_name.replace("_", " ").title())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axes[row_idx, -1],
            orientation='vertical',
            fraction=0.05,
            pad=0.02
        )

    fig.suptitle(f"{metric_label} Heatmaps", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0.04, 0.04, 0.96, 0.94))

    file_path = output_dir / f"{output_prefix}.png"
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ {metric_label} heatmaps saved to: {file_path}")
    return file_path

