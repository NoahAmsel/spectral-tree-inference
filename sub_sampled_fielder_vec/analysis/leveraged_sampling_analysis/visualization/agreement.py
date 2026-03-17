"""Agreement visualization utilities for leveraged sampling analysis.

This module provides plotting functions for visualizing partition agreement
vs sampling probability.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_agreement_vs_p(
    data: Dict[str, pd.DataFrame],
    methods: List[str] = ["uniform", "leveraged"],
    n_taxa_values: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Plot partition agreement vs p for each method and n.

    Args:
        data: Dict mapping method -> DataFrame with columns: p, partition_agreement_M, num_taxa
        methods: List of methods to plot
        n_taxa_values: Optional list of n_taxa to plot (if None, plots all available)
        output_path: Optional path to save figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure
    """
    # Get all n_taxa values if not specified
    if n_taxa_values is None:
        all_n_values = set()
        for method in methods:
            if method in data:
                all_n_values.update(data[method]["num_taxa"].unique())
        n_taxa_values = sorted(all_n_values)

    n_plots = len(n_taxa_values)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = axes.flatten()

    colors = {"uniform": "#2E86AB", "leveraged": "#A23B72"}
    markers = {"uniform": "o", "leveraged": "s"}

    for idx, n in enumerate(n_taxa_values):
        ax = axes[idx]

        for method in methods:
            if method not in data:
                continue

            method_df = data[method]
            n_df = method_df[method_df["num_taxa"] == n].sort_values("p")

            if len(n_df) == 0:
                continue

            ax.plot(
                n_df["p"],
                n_df["partition_agreement_M"],
                label=method.capitalize(),
                marker=markers.get(method, "o"),
                linewidth=2,
                markersize=6,
                color=colors.get(method, None),
            )

        ax.set_xlabel("Sampling probability p", fontsize=12)
        ax.set_ylabel("Partition agreement (%)", fontsize=12)
        ax.set_xscale("log")
        ax.set_title(f"n = {n} taxa", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        ax.axhline(95, color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_merged_agreement_vs_p(
    df: pd.DataFrame,
    method_name: Optional[str] = None,
    subtitle: Optional[str] = None,
    config: Optional[Dict] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    global_n_taxa: Optional[List[int]] = None,
    phase_threshold: float = 80.0,
) -> plt.Figure:
    """Plot partition agreement vs p for all n values in a single merged plot.
    
    Creates a single-panel plot showing all n_taxa values on one figure,
    similar to the reference partition_agreement.png format.
    
    Args:
        df: DataFrame with columns: p, partition_agreement_M, num_taxa
        method_name: Optional method name for title (e.g., "Uniform", "Leveraged")
        subtitle: Optional subtitle (overridden if config is provided)
        config: Optional config dict with tree model parameters. If provided,
            generates subtitle from: L (sequence_length), mutation_rate, tree.model,
            and tree.params (e.g., pop_size for Kingman coalescent).
            Accepts both flat dict and nested config structures.
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        global_n_taxa: Optional global sorted list of all n_taxa values across all
            runs being compared. When provided, colors are assigned by position in
            this global list so the same n value always gets the same color across
            plots that may have different subsets of n values.
        phase_threshold: Agreement level (%) at which to mark the phase transition.
            The last point below this threshold is drawn as a filled dot. Default 80.

    Returns:
        matplotlib Figure
    """
    # Build subtitle from config if provided
    if config is not None:
        subtitle_parts = []
        
        # Extract sequence length (L)
        seq_len = None
        if "sequence_length" in config:
            seq_len = config["sequence_length"]
        elif "sequence" in config and isinstance(config["sequence"], dict):
            seq_len = config["sequence"].get("length")
        elif "sequence_length_values" in config:
            # Sweep config format
            seq_len = config["sequence_length_values"][0] if config["sequence_length_values"] else None
        
        if seq_len is not None:
            subtitle_parts.append(f"L = {seq_len:,}")
        
        # Extract mutation rate
        mutation_rate = None
        if "mutation_rate" in config:
            mutation_rate = config["mutation_rate"]
        elif "sequence" in config and isinstance(config["sequence"], dict):
            params = config["sequence"].get("params", {})
            mutation_rate = params.get("mutation_rate")
        
        if mutation_rate is not None:
            subtitle_parts.append(f"μ = {mutation_rate}")
        
        # Extract tree model
        tree_model = None
        if "tree_model" in config:
            tree_model = config["tree_model"]
        elif "tree" in config and isinstance(config["tree"], dict):
            tree_model = config["tree"].get("model")
        
        if tree_model is not None:
            # Format tree model name nicely
            tree_display = tree_model.replace("_", " ").title()
            subtitle_parts.append(f"Tree: {tree_display}")
        
        # Extract tree parameters (e.g., pop_size for Kingman)
        tree_params = {}
        if "tree_params" in config:
            tree_params = config["tree_params"]
        elif "tree" in config and isinstance(config["tree"], dict):
            tree_params = config["tree"].get("params", {})
        
        # Add notable tree parameters
        if tree_params:
            if "pop_size" in tree_params:
                subtitle_parts.append(f"Ne = {tree_params['pop_size']}")
            if "branch_length" in tree_params:
                subtitle_parts.append(f"branch = {tree_params['branch_length']}")
        
        # Combine parts into subtitle
        if subtitle_parts:
            subtitle = ", ".join(subtitle_parts)
    
    # Get unique n_taxa values
    n_taxa_values = sorted(df["num_taxa"].unique())

    # Build a consistent color map: anchor by global_n_taxa when provided so the
    # same n value always maps to the same color across plots with different subsets.
    reference_list = sorted(global_n_taxa) if global_n_taxa is not None else n_taxa_values
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(reference_list), 1)))
    color_map = {n: palette[i] for i, n in enumerate(reference_list)}
    # Fallback for any n not in reference_list (shouldn't happen in normal use)
    for i, n in enumerate(n_taxa_values):
        if n not in color_map:
            color_map[n] = plt.cm.tab10((i + len(reference_list)) / 10)

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot each n_taxa value
    for n in n_taxa_values:
        n_df = df[df["num_taxa"] == n].sort_values("p")

        if len(n_df) == 0:
            continue

        # Extract data
        p_vals = n_df["p"].values
        agreement_vals = n_df["partition_agreement_M"].values
        color = color_map[n]

        # Compute phase transition point for label and filled dot
        below_mask = agreement_vals < phase_threshold
        if np.any(below_mask) and not np.all(below_mask):
            last_below_idx = int(np.where(below_mask)[0][-1])
            p_star = p_vals[last_below_idx]
            label = f"n={n},  p*={p_star:.3g}"
        else:
            last_below_idx = None
            label = f"n={n}"

        # Plot line with hollow markers
        ax.plot(
            p_vals,
            agreement_vals,
            "-",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.5,
            markeredgecolor=color,
            label=label,
            linewidth=2,
            markersize=6,
            color=color,
            zorder=2,
        )

        # Phase transition dot: filled solid dot on top of the hollow marker
        if last_below_idx is not None:
            ax.plot(
                p_vals[last_below_idx],
                agreement_vals[last_below_idx],
                "o",
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=9,
                zorder=4,
            )
    
    # Set axes properties
    ax.set_xscale("log")
    ax.set_xlabel("p", fontsize=14, fontweight="bold")
    ax.set_ylabel("Partition agreement (%)", fontsize=14, fontweight="bold")
    ax.set_ylim([40, 105])
    ax.grid(True, alpha=0.3)
    
    # Add 95% reference line
    ax.axhline(95, color="red", linestyle="--", alpha=0.5, linewidth=1, zorder=1)
    
    # Add theoretical sample complexity reference lines
    # Paper: O(nr log(n)) samples needed for Phase 1
    # For symmetric matrices: p_min ≈ 4 * r * log(n) / n (conservative estimate)
    # For r=2 (Fiedler vector): p_min ≈ 8 * log(n) / n
    if method_name and method_name.lower() == "leveraged":
        r = 1  # Target rank for Fiedler vector
        # Get data range for p values
        all_p_vals = df["p"].values
        p_min_data = np.min(all_p_vals)
        p_max_data = np.max(all_p_vals)
        
        for idx, n in enumerate(n_taxa_values):
            # Theoretical minimum p based on O(nr log(n)) sample complexity
            # Using conservative estimate: p_min = 4 * r * log(n) / n
            p_min_theoretical = 4 * r * np.log(n) / n if n > 1 else 0.01

            # Only show if it's within the data range
            if p_min_theoretical >= p_min_data and p_min_theoretical <= p_max_data:
                ax.axvline(
                    p_min_theoretical,
                    color="gray",
                    linestyle=":",
                    alpha=0.6,
                    linewidth=1.5,
                    zorder=1,
                )
                # Add annotation at different y positions to avoid overlap
                y_pos = 20 + (idx % 3) * 4  # Stagger annotations at ~20% height
                # (idx kept for stagger positioning only)
                ax.text(
                    p_min_theoretical,
                    y_pos,
                    f"$\\mathcal{{O}}(nr\\log n)$\n(n={n})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="gray",
                    style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray", linewidth=0.5),
                    zorder=3,
                )
    
    # Set title and subtitle
    if method_name:
        if subtitle:
            # Use suptitle for main title, ax.set_title for subtitle below it
            fig.suptitle(method_name, fontsize=16, fontweight="bold", y=0.98)
            ax.set_title(subtitle, fontsize=12, style="italic", color="gray", pad=10)
        else:
            ax.set_title(method_name, fontsize=16, fontweight="bold", pad=15)
    elif subtitle:
        ax.set_title(subtitle, fontsize=14, pad=10)
    
    # Add legend in lower right corner
    ax.legend(
        loc="lower right",
        ncol=1,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor=(0.85, 0.85, 0.85),
        borderpad=0.6,
        handlelength=1.5,
        handletextpad=0.5,
        fontsize=11,
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
    
    return fig
