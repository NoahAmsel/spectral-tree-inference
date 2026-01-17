"""Plot eigenvalue trajectories showing BBP transition."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path


def _extract_neighborhood(
    config_data: List[Dict[str, Any]], p_center: float, n_neighbors: int
) -> List[Dict[str, Any]]:
    """Return up to n_neighbors samples on each side of the exact p_center record."""
    if not config_data or p_center is None or n_neighbors <= 0:
        return []

    sorted_rows = sorted(config_data, key=lambda row: row["p"])
    p_vals = np.array([row["p"] for row in sorted_rows], dtype=float)
    if p_vals.size == 0:
        return []

    matches = np.where(np.isclose(p_vals, p_center, rtol=1e-9, atol=0))[0]
    if matches.size == 0:
        return []

    center_idx = int(matches[0])

    # Collect neighbors strictly on each side (ascending order retained)
    left_rows: List[Dict[str, Any]] = []
    for offset in range(n_neighbors, 0, -1):
        idx = center_idx - offset
        if idx >= 0:
            left_rows.append(sorted_rows[idx])

    center_row = sorted_rows[center_idx]

    right_rows: List[Dict[str, Any]] = []
    for offset in range(1, n_neighbors + 1):
        idx = center_idx + offset
        if idx < len(sorted_rows):
            right_rows.append(sorted_rows[idx])

    return left_rows + [center_row] + right_rows


def plot_eigenvalue_pop(
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]],
    output_path: Path,
    p_crit_map: Dict[Tuple[int, int], float],
    eigenvalue_crossing_map: Dict[Tuple[int, int], float],
) -> None:
    """Plot λ₂(p) and λ₃(p) trajectories in 5×4 grid (N rows × L columns).

    Graph B: The Eigenvalue "Pop" (Enhanced)
    - Grid: 5 rows (N values) × 4 columns (L values)
    - X-axis: Sampling rate p (log scale)
    - Y-axis: Eigenvalues
    - Noise ocean: Shaded gray region from 0 to λ₃(p)
    - Signal: Bold red line for λ₂(p)
    - Transitions: Two vertical markers
      * Blue dashed: λ₂ crosses λ₃ (BBP transition)
      * Green dashed: partition_agreement ≥ 95% (p_crit)

    Args:
        grouped: Dictionary mapping (N, L) -> list of rows
        output_path: Path to save PNG file
        p_crit_map: Dictionary mapping (N, L) -> p_crit
        eigenvalue_crossing_map: Dictionary mapping (N, L) -> p where λ₂ > λ₃
    """
    # Extract unique N and L values
    N_vals = sorted(set(k[0] for k in grouped.keys()))
    L_vals = sorted(set(k[1] for k in grouped.keys()))

    n_rows = len(N_vals)
    n_cols = len(L_vals)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)

    for i, N in enumerate(N_vals):
        for j, L in enumerate(L_vals):
            ax = axes[i, j] if n_rows > 1 else axes[j]

            if (N, L) not in grouped:
                ax.axis('off')
                continue

            config_data = grouped[(N, L)]
            p_vals = np.array([row["p"] for row in config_data])
            # Convert to float64, None becomes NaN
            lambda_2_S = np.array([row["mean_lambda2_L_S"] for row in config_data], dtype=np.float64)
            lambda_3_S = np.array([row["mean_lambda3_L_S"] for row in config_data], dtype=np.float64)

            # Filter out NaN values for plotting
            valid_mask = ~(np.isnan(lambda_2_S) | np.isnan(lambda_3_S))
            p_valid = p_vals[valid_mask]
            l2_valid = lambda_2_S[valid_mask]
            l3_valid = lambda_3_S[valid_mask]

            # Reference values (use any non-NaN entry across runs)
            lambda_2_M_vals = np.array(
                [row["mean_lambda2_L_M"] for row in config_data], dtype=float
            )
            lambda_3_M_vals = np.array(
                [row["mean_lambda3_L_M"] for row in config_data], dtype=float
            )
            lambda_2_M = (
                float(np.nanmean(lambda_2_M_vals))
                if np.any(~np.isnan(lambda_2_M_vals))
                else np.nan
            )
            lambda_3_M = (
                float(np.nanmean(lambda_3_M_vals))
                if np.any(~np.isnan(lambda_3_M_vals))
                else np.nan
            )

            # ENHANCEMENT 1: Shade "noise ocean" (0 to λ₃)
            if len(p_valid) > 0:
                ax.fill_between(
                    p_valid, 0, l3_valid, color="lightgray", alpha=0.5, label="Noise bulk"
                )

            # ENHANCEMENT 2: Bold red λ₂ line (signal emerging)
            ax.plot(
                p_valid,
                l2_valid,
                "-",
                color="red",
                linewidth=2.5,
                label="λ₂(L_S)",
                zorder=3,
            )

            # λ₃ trajectory (thin line)
            ax.plot(
                p_valid, l3_valid, "-", color="darkgray", linewidth=1, label="λ₃(L_S)"
            )

            # Reference lines (full matrix)
            if not np.isnan(lambda_2_M):
                ax.axhline(
                    lambda_2_M,
                    linestyle=":",
                    color="#b71c1c",
                    alpha=0.9,
                    linewidth=3,
                    zorder=4,
                    label="λ₂(L_M)",
                )
            if not np.isnan(lambda_3_M):
                ax.axhline(
                    lambda_3_M,
                    linestyle=":",
                    color="#1565c0",
                    alpha=0.9,
                    linewidth=3,
                    zorder=4,
                    label="λ₃(L_M)",
                )

            # ENHANCEMENT 3: Transition markers
            # Blue: Eigenvalue crossing (BBP)
            p_cross = eigenvalue_crossing_map.get((N, L))
            if p_cross is not None:
                ax.axvline(
                    p_cross,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label="λ₂>λ₃",
                )

            # Green: Partition agreement threshold
            p_crit = p_crit_map.get((N, L))
            if p_crit is not None:
                ax.axvline(
                    p_crit,
                    color="green",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label="p_crit",
                )

            ax.set_xscale("log")
            ax.grid(alpha=0.2)

            # Labels
            if i == 0:
                ax.set_title(f"L={L}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"N={N}\nEigenvalue", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("p", fontsize=9)
    # Build shared legend from all plotted artists
    legend_items: Dict[str, Any] = {}
    for ax in np.array(axes).ravel():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in legend_items:
                legend_items[label] = handle

    if legend_items:
        fig.legend(
            legend_items.values(),
            legend_items.keys(),
            loc="upper center",
            ncol=max(1, len(legend_items)),
            fontsize=11,
            frameon=True,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.93),
        )

    plt.suptitle("Graph B: BBP Eigenvalue Transition", fontsize=16, y=0.995)
    plt.tight_layout(rect=(0, 0.02, 1, 0.9))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_eigenvalue_pop_zoom(
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]],
    output_path: Path,
    p_crit_map: Dict[Tuple[int, int], float],
    eigenvalue_crossing_map: Dict[Tuple[int, int], float],
    n_neighbors: int = 3,
) -> None:
    """Zoomed view around p_crit using ±n_neighbors sampling rates."""

    N_vals = sorted(set(k[0] for k in grouped.keys()))
    L_vals = sorted(set(k[1] for k in grouped.keys()))

    n_rows = len(N_vals)
    n_cols = len(L_vals)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=False)

    for i, N in enumerate(N_vals):
        for j, L in enumerate(L_vals):
            ax = axes[i, j] if n_rows > 1 else axes[j]

            if (N, L) not in grouped:
                ax.axis("off")
                continue

            config_data = grouped[(N, L)]
            p_crit = p_crit_map.get((N, L))
            if p_crit is None:
                ax.axis("off")
                continue

            neighborhood = _extract_neighborhood(config_data, p_crit, n_neighbors)
            if not neighborhood:
                ax.axis("off")
                continue

            p_vals = np.array([row["p"] for row in neighborhood])
            # Convert to float64, None becomes NaN
            lambda_2_S = np.array([row["mean_lambda2_L_S"] for row in neighborhood], dtype=np.float64)
            lambda_3_S = np.array([row["mean_lambda3_L_S"] for row in neighborhood], dtype=np.float64)

            valid_mask = ~(np.isnan(lambda_2_S) | np.isnan(lambda_3_S))
            p_valid = p_vals[valid_mask]
            l2_valid = lambda_2_S[valid_mask]
            l3_valid = lambda_3_S[valid_mask]

            # Check if we have enough valid data to plot
            if len(p_valid) < 2:
                ax.text(
                    0.5, 0.5,
                    f"Insufficient valid data\nnear p_crit={p_crit:.4f}",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=10, color="gray"
                )
                ax.axis("off")
                continue

            # Note: p_crit might not have eigenvalue data if guardrails stopped computation
            # This is expected behavior when compute_metrics_on_trigger=false

            lambda_2_M_vals = np.array(
                [row["mean_lambda2_L_M"] for row in config_data], dtype=float
            )
            lambda_3_M_vals = np.array(
                [row["mean_lambda3_L_M"] for row in config_data], dtype=float
            )
            lambda_2_M = (
                float(np.nanmean(lambda_2_M_vals))
                if np.any(~np.isnan(lambda_2_M_vals))
                else np.nan
            )
            lambda_3_M = (
                float(np.nanmean(lambda_3_M_vals))
                if np.any(~np.isnan(lambda_3_M_vals))
                else np.nan
            )

            if len(p_valid) > 0:
                ax.fill_between(
                    p_valid,
                    0,
                    l3_valid,
                    color="lightgray",
                    alpha=0.5,
                    label="Noise bulk",
                )

            ax.plot(
                p_valid,
                l2_valid,
                "-",
                color="red",
                linewidth=2.5,
                label="λ₂(L_S)",
                zorder=3,
            )

            ax.plot(
                p_valid,
                l3_valid,
                "-",
                color="darkgray",
                linewidth=1,
                label="λ₃(L_S)",
            )

            if not np.isnan(lambda_2_M):
                ax.axhline(
                    lambda_2_M,
                    linestyle=":",
                    color="#b71c1c",
                    alpha=0.9,
                    linewidth=3,
                    zorder=4,
                    label="λ₂(L_M)",
                )
            if not np.isnan(lambda_3_M):
                ax.axhline(
                    lambda_3_M,
                    linestyle=":",
                    color="#1565c0",
                    alpha=0.9,
                    linewidth=3,
                    zorder=4,
                    label="λ₃(L_M)",
                )

            p_cross = eigenvalue_crossing_map.get((N, L))
            if p_cross is not None:
                ax.axvline(
                    p_cross,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label="λ₂>λ₃",
                )

            # Always show p_crit marker, even if outside valid data range
            if p_crit is not None:
                # Check if p_crit is within the valid data range
                p_crit_in_range = (len(p_valid) > 0 and
                                   p_valid.min() <= p_crit <= p_valid.max())

                ax.axvline(
                    p_crit,
                    color="green",
                    linestyle="--" if p_crit_in_range else ":",
                    linewidth=1.5,
                    alpha=0.7 if p_crit_in_range else 0.4,
                    label="p_crit" + ("" if p_crit_in_range else " (no data)"),
                )

            ax.set_xscale("log")
            positive_window = p_vals[p_vals > 0]
            if positive_window.size > 0 and p_crit > 0:
                log_window = np.log10(positive_window)
                center_log = np.log10(p_crit)
                min_log = log_window.min()
                max_log = log_window.max()
                delta = max(center_log - min_log, max_log - center_log)
                if not np.isfinite(delta) or delta <= 0:
                    delta = 0.05
                ax.set_xlim(10 ** (center_log - delta), 10 ** (center_log + delta))
            elif len(p_valid) >= 2:
                left, right = p_valid.min(), p_valid.max()
                if left == right:
                    left *= 0.9
                    right *= 1.1
                ax.set_xlim(left, right)
            ax.grid(alpha=0.2)

            if i == 0:
                ax.set_title(f"L={L}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"N={N}\nEigenvalue", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("p (zoomed)", fontsize=9)

    legend_items: Dict[str, Any] = {}
    for ax in np.array(axes).ravel():
        if not hasattr(ax, "get_legend_handles_labels"):
            continue
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in legend_items:
                legend_items[label] = handle

    if legend_items:
        fig.legend(
            legend_items.values(),
            legend_items.keys(),
            loc="upper center",
            ncol=max(1, len(legend_items)),
            fontsize=11,
            frameon=True,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.93),
        )

    plt.suptitle("Graph B1: Eigenvalues near p_crit", fontsize=16, y=0.995)
    plt.tight_layout(rect=(0, 0.02, 1, 0.9))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
