"""Plot Empirical Rank vs sampling rate showing rank recovery behavior."""

import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


def plot_empirical_rank(
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]],
    output_path: Path,
    p_crit_map: Dict[Tuple[int, int], float],
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot Empirical Rank vs p in 5×4 grid (N rows × L columns).

    Graph F_1: Empirical Rank Recovery
    - Grid: 5 rows (N values) × 4 columns (L values)
    - X-axis: Sampling rate p (log scale)
    - Y-axis: mean_empirical_rank_L (primary: sampled L_S, reference: full L_M)
    - Comparison: Solid line (L_S) vs dashed line (L_M) to show convergence
    - Marker: Green vertical line at p_crit

    Expected: Rank increases as sampling improves, converging toward full matrix rank.

    Args:
        grouped: Dictionary mapping (N, L) -> list of rows
        output_path: Path to save PNG file
        p_crit_map: Dictionary mapping (N, L) -> p_crit
        xlim: Optional x-axis limits to sync with other plots
    """
    # Extract unique N and L values
    N_vals = sorted(set(k[0] for k in grouped.keys()))
    L_vals = sorted(set(k[1] for k in grouped.keys()))

    n_rows = len(N_vals)
    n_cols = len(L_vals)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)

    # Collect all rank values to determine if log scale is appropriate
    all_rank_L_S = []
    all_rank_L_M = []
    for config_data in grouped.values():
        all_rank_L_S.extend([row["mean_empirical_rank_L_S"] for row in config_data if row["mean_empirical_rank_L_S"] is not None])
        all_rank_L_M.extend([row["mean_empirical_rank_L_M"] for row in config_data if row["mean_empirical_rank_L_M"] is not None])

    # Determine y-scale: use log if values span > 2 orders of magnitude
    if all_rank_L_S and all_rank_L_M:
        min_val = min(min(all_rank_L_S), min(all_rank_L_M))
        max_val = max(max(all_rank_L_S), max(all_rank_L_M))
        use_log_scale = (max_val / min_val) > 100 if min_val > 0 else False
    else:
        use_log_scale = False

    for i, N in enumerate(N_vals):
        for j, L in enumerate(L_vals):
            ax = axes[i, j] if n_rows > 1 else axes[j]

            if (N, L) not in grouped:
                ax.axis('off')
                continue

            config_data = grouped[(N, L)]

            # Filter out None values for L_S
            data_L_S = [(row["p"], row["mean_empirical_rank_L_S"])
                        for row in config_data if row["mean_empirical_rank_L_S"] is not None]

            # Get L_M values (should be constant, but average over runs)
            rank_L_M_values = [row["mean_empirical_rank_L_M"]
                               for row in config_data
                               if row["mean_empirical_rank_L_M"] is not None]

            # Plot sampled rank (primary) if data exists
            if data_L_S:
                p_vals_S, rank_L_S = zip(*data_L_S)
                ax.plot(
                    p_vals_S,
                    rank_L_S,
                    "o-",
                    color="steelblue",
                    markersize=3,
                    linewidth=1.5,
                    label="L_S (sampled)",
                )

            # Plot full rank (reference) as horizontal line if data exists
            if rank_L_M_values:
                avg_rank_L_M = sum(rank_L_M_values) / len(rank_L_M_values)
                ax.axhline(
                    avg_rank_L_M,
                    linestyle="--",
                    color="gray",
                    linewidth=1.5,
                    alpha=0.7,
                    label="L_M (full)",
                )

            # Mark p_crit with vertical line
            p_crit = p_crit_map.get((N, L))
            if p_crit is not None and data_L_S:
                p_vals_S_array = list(p_vals_S)
                p_crit_in_range = min(p_vals_S_array) <= p_crit <= max(p_vals_S_array)
                ax.axvline(
                    p_crit,
                    color="green",
                    linestyle="--" if p_crit_in_range else ":",
                    linewidth=1.5,
                    alpha=0.7 if p_crit_in_range else 0.4,
                )

            ax.set_xscale("log")
            if use_log_scale:
                ax.set_yscale("log")
            ax.grid(alpha=0.2)

            # Sync x-axis if limits provided
            if xlim is not None:
                ax.set_xlim(xlim)

            # Labels
            if i == 0:
                ax.set_title(f"L={L}", fontsize=11)
            if j == 0:
                ylabel = f"N={N}\nRank"
                if use_log_scale:
                    ylabel += " (log)"
                ax.set_ylabel(ylabel, fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("p", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best", framealpha=0.9)

    plt.suptitle("Graph F_1: Empirical Rank Recovery", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
