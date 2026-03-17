"""Plot Davis-Kahan ratio vs sampling rate."""

import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from pathlib import Path


def plot_stability_curve(
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]], output_path: Path
) -> None:
    """Plot Davis-Kahan ratio vs p in 5×4 grid (N rows × L columns).

    Graph C: Stability Verification
    - Grid: 5 rows (N values) × 4 columns (L values)
    - X-axis: Sampling rate p
    - Y-axis: mean_dk_ratio_S (log scale)
    - Threshold: Horizontal red line at 0.5

    Expected: Ratio crashes below 0.5 when partition_agreement_M → 100%.

    Args:
        grouped: Dictionary mapping (N, L) -> list of rows
        output_path: Path to save PNG file
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
            p_vals = [row["p"] for row in config_data]
            dk_ratios = [row["mean_dk_ratio_S"] for row in config_data]

            # Plot DK ratio
            ax.plot(p_vals, dk_ratios, "o-", color="purple", markersize=3, linewidth=1.5)

            # Stability threshold
            ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.2)

            # Labels
            if i == 0:
                ax.set_title(f"L={L}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"N={N}\nDK Ratio", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("p", fontsize=9)

    plt.suptitle("Graph C: Davis-Kahan Stability", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
