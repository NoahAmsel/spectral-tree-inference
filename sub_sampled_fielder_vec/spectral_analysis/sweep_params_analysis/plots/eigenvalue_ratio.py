"""Plot eigenvalue ratio λ₂/λ₃ vs sampling rate."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path


def plot_eigenvalue_ratio(
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]],
    output_path: Path,
    p_crit_map: Dict[Tuple[int, int], float],
) -> None:
    """Plot λ₂/λ₃ ratio vs p in 5×4 grid (N rows × L columns).

    Graph B3: Eigenvalue Ratio
    - Grid: 5 rows (N values) × 4 columns (L values)
    - X-axis: Sampling rate p (log scale)
    - Y-axis: λ₂/λ₃ ratio
    - Horizontal line: y=1 (where λ₂ = λ₃)
    - Marker: Green vertical line at p_crit

    Expected: Ratio crosses 1 when λ₂ overtakes λ₃ (signal emerges).

    Args:
        grouped: Dictionary mapping (N, L) -> list of rows
        output_path: Path to save PNG file
        p_crit_map: Dictionary mapping (N, L) -> p_crit
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

            # Reference values (use nanmean across all p values since L_M doesn't depend on p)
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

            # Compute ratios (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_S = lambda_2_S / lambda_3_S
                ratio_M = lambda_2_M / lambda_3_M if not (np.isnan(lambda_2_M) or np.isnan(lambda_3_M) or lambda_3_M == 0) else np.nan

            # Filter out invalid values (NaN, inf) for subsampled ratio
            valid_mask_S = np.isfinite(ratio_S)
            p_valid_S = p_vals[valid_mask_S]
            ratio_valid_S = ratio_S[valid_mask_S]

            if len(p_valid_S) < 2:
                ax.text(
                    0.5, 0.5,
                    "Insufficient data",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=10, color="gray"
                )
                ax.axis("off")
                continue

            # Plot true ratio (full matrix) as horizontal reference line
            if np.isfinite(ratio_M):
                ax.axhline(
                    ratio_M,
                    linestyle=":",
                    color="darkred",
                    alpha=0.8,
                    linewidth=2.5,
                    label="λ₂/λ₃(L_M)",
                )

            # Plot subsampled ratio
            ax.plot(
                p_valid_S,
                ratio_valid_S,
                "o-",
                color="darkorange",
                linewidth=2,
                markersize=4,
                label="λ₂/λ₃(L_S)",
            )

            # Reference line at y=1 (where λ₂ = λ₃)
            ax.axhline(
                1.0,
                color="black",
                linestyle=":",
                linewidth=1,
                alpha=0.3,
            )

            # Mark p_crit
            p_crit = p_crit_map.get((N, L))
            if p_crit is not None:
                p_crit_in_range = p_valid_S.min() <= p_crit <= p_valid_S.max()
                ax.axvline(
                    p_crit,
                    color="green",
                    linestyle="--" if p_crit_in_range else ":",
                    linewidth=1.5,
                    alpha=0.7 if p_crit_in_range else 0.4,
                )

            ax.set_xscale("log")
            ax.grid(alpha=0.2)

            # Labels
            if i == 0:
                ax.set_title(f"L={L}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"N={N}\nλ₂/λ₃", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("p", fontsize=9)

            # Add legend to first subplot only
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=7)

    plt.suptitle("Graph B3: Eigenvalue Ratio λ₂/λ₃", fontsize=16, y=0.995)
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
