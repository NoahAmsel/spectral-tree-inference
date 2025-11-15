# plotting.py
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter


def _build_series_from_table(d):
    """
    Accepts:
      { "columns": ["num_taxa","p","median","std"],
        "rows": [ {"num_taxa":..., "p":..., "median":..., "std":...}, ... ] }
    Returns series dict: { "n=1024": {"x":[...], "mean":[...], "std":[...]}, ... }
    """
    cols = d.get("columns"); rows = d.get("rows")
    if not (isinstance(cols, list) and isinstance(rows, list)):
        return None

    def col(name, alts=()):
        for nm in (name, *alts):
            if nm in cols: return nm
        return None

    c_n   = col("num_taxa", ("n", "num_taxon"))
    c_x   = col("p", ("alpha", "x"))
    c_mu  = col("sign_agreement", ("median", "mean", "y", "value"))  # Prefer new format
    c_std = col("std", ("stderr", "sigma"))
    if not (c_n and c_x and c_mu):
        return None

    series = {}
    for r in rows:
        key = f"n={int(r[c_n])}"
        x   = float(r[c_x])
        mu  = float(r[c_mu])
        sd  = float(r[c_std]) if (c_std and r.get(c_std) is not None) else 0.0
        s = series.setdefault(key, {"x": [], "mean": [], "std": []})
        s["x"].append(x); s["mean"].append(mu); s["std"].append(sd)

    # sort each by x
    import numpy as np
    for k, s in series.items():
        arr = np.array(list(zip(s["x"], s["mean"], s["std"])), dtype=float)
        arr = arr[arr[:, 0].argsort()]
        s["x"], s["mean"], s["std"] = arr[:, 0].tolist(), arr[:, 1].tolist(), arr[:, 2].tolist()
    return series

def plot_from_json_simple(
    json_path,
    output_path="Figure_1.png",
    xlabel=r"$p$",
    ylabel="Sign agreement (%)",
    phase_min=1e-2,
    phase_max=1,
    dpi=600,
):
    """
    Works with either:
      {"series": {"n=...": {"x":[...], "mean":[...], "std":[...]}, ...}}
    or table:
      {"columns": ["num_taxa","p","median","std"], "rows": [...]}
    """
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib import rcParams
    from cycler import cycler
    import math

    # --- MATLAB-ish palette + trimmed sizes ---
    matlab_colors = [
        (0.0000, 0.4470, 0.7410),
        (0.8500, 0.3250, 0.0980),
        (0.9290, 0.6940, 0.1250),
        (0.4940, 0.1840, 0.5560),
        (0.4660, 0.6740, 0.1880),
        (0.3010, 0.7450, 0.9330),
        (0.6350, 0.0780, 0.1840),
    ]
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.prop_cycle": cycler("color", matlab_colors),
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": dpi, "savefig.dpi": dpi,
    })

    # --- load ---
    with open(json_path, "r") as f:
        data = json.load(f)

    def _build_series(d):
        if "columns" in d and "rows" in d:
            series = {}
            for r in d["rows"]:
                key = f"n={int(r['num_taxa'])}"
                s = series.setdefault(key, {"x": [], "mean": [], "std": []})
                s["x"].append(float(r["p"]))
                # Prefer new format (sign_agreement), fall back to median or mean
                s["mean"].append(float(r.get("sign_agreement", r.get("median", r.get("mean", 0)))))
                s["std"].append(float(r.get("std", 0)))
            for s in series.values():
                arr = np.array(list(zip(s["x"], s["mean"], s["std"])), float)
                arr = arr[arr[:, 0].argsort()]
                s["x"], s["mean"], s["std"] = arr[:, 0], arr[:, 1], arr[:, 2]
            return series
        return data.get("series")

    series = _build_series(data)
    if not series:
        raise ValueError("Could not extract series from JSON")

    # --- figure (more space) ---
    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax in (ax_all, ax_zoom):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_zoom.yaxis.set_visible(False)

    def _plot(ax, xlim):
        # put bands behind lines
        z_band, z_line = 1, 2
        for name, s in series.items():
            x = np.asarray(s["x"]); y = np.asarray(s["mean"]); sd = np.asarray(s["std"])
            m = x > 0
            x, y, sd = x[m], y[m], sd[m]
            ax.fill_between(x, y - sd, y + sd, alpha=0.08, linewidth=0, zorder=z_band)
            ax.plot(x, y, "-", marker="o", markerfacecolor="white",
                    markeredgewidth=0.9, label=name, zorder=z_line)

        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(20, 102)  # less vertical crowding; tweak if needed

        # labeled powers only
        ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
        labels = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
        t_in = [t for t in ticks if xlim[0] <= t <= xlim[1]]
        l_in = [labels[ticks.index(t)] for t in t_in]
        ax.xaxis.set_major_locator(FixedLocator(t_in))
        ax.xaxis.set_major_formatter(FixedFormatter(l_in))
        ax.minorticks_off()

        # sparser y ticks
        ax.set_yticks([20, 40, 60, 80, 100])

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)

    _plot(ax_all, (1e-4, 1e0))
    _plot(ax_zoom, (phase_min, phase_max))

    # --- legend outside, 2 rows to avoid overlap ---
    handles, labels = ax_all.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_zoom.get_legend_handles_labels()
    if labels:
        n = len(labels)
        ncol = n
        fig.legend(
            handles, labels,
            loc="lower center", ncol=ncol,
            frameon=True, fancybox=False, facecolor="white",
            edgecolor=(0.85, 0.85, 0.85), borderpad=0.2,
            handlelength=1.2, columnspacing=1.0, handletextpad=0.5
        )
        fig.subplots_adjust(bottom=0.25, left=0.10, right=0.98, top=0.97, wspace=0.15)
    else:
        fig.subplots_adjust(bottom=0.10, left=0.10, right=0.98, top=0.97, wspace=0.15)

    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


def plot_fiedler_vectors(
    run_dir: str,
    output_path: str = "fiedler_vectors.png",
    dpi: int = 600,
    figsize: tuple = (10, 6)
):
    """
    Plot the true Fiedler vectors for each taxa count.
    
    Args:
        run_dir: Directory containing the experiment results
        output_path: Path to save the plot
        dpi: DPI for the plot
        figsize: Figure size (width, height)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from cycler import cycler
    
    # Set up plotting style
    matlab_colors = [
        (0.0000, 0.4470, 0.7410),
        (0.8500, 0.3250, 0.0980),
        (0.9290, 0.6940, 0.1250),
        (0.4940, 0.1840, 0.5560),
        (0.4660, 0.6740, 0.1880),
        (0.3010, 0.7450, 0.9330),
        (0.6350, 0.0780, 0.1840),
    ]
    
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.prop_cycle": cycler("color", matlab_colors),
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": dpi, "savefig.dpi": dpi,
    })
    
    # Find all Fiedler vector files
    fiedler_files = []
    for filename in os.listdir(run_dir):
        if filename.startswith("fiedler_ref") and filename.endswith(".npy"):
            fiedler_files.append(filename)
    
    if not fiedler_files:
        print(f"No Fiedler vector files found in {run_dir}")
        return None
    
    # Sort files by taxa count
    def extract_taxa_count(filename):
        if filename == "fiedler_ref.npy":
            # Single taxa run - try to get taxa count from config
            config_path = os.path.join(run_dir, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config.get('num_taxa', 0)
            return 0
        else:
            # Multi-taxa run - extract from filename
            try:
                return int(filename.split('_n=')[1].split('.npy')[0])
            except:
                return 0
    
    fiedler_files.sort(key=extract_taxa_count)
    
    # Create subplots in a single column
    n_files = len(fiedler_files)
    if n_files == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
    else:
        n_cols = 1  # Single column
        n_rows = n_files
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
        if n_files == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_files > 1 else [axes]
    
    for i, filename in enumerate(fiedler_files):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        # Load Fiedler vector
        fiedler_path = os.path.join(run_dir, filename)
        fiedler_vector = np.load(fiedler_path)
        
        # Get taxa count
        taxa_count = extract_taxa_count(filename)
        
        # Plot the Fiedler vector
        x = np.arange(len(fiedler_vector))
        ax.plot(x, fiedler_vector, 'o-', markersize=2, linewidth=1)
        
        # Styling
        ax.set_xlabel('Taxon Index', fontsize=10)
        ax.set_ylabel('Fiedler Vector Value', fontsize=10)
        ax.set_title(f'n = {taxa_count}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Set reasonable y-limits
        y_min, y_max = fiedler_vector.min(), fiedler_vector.max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Hide unused subplots
    for i in range(len(fiedler_files), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    
    print(f"Fiedler vectors plot saved to: {output_path}")
    return output_path


def plot_faceted_by_sequence_length(
    json_path: str,
    output_path: str = "plot_faceted.png",
    xlabel: str = r"$p$",
    ylabel: str = "Sign agreement (%)",
    phase_min: float = 1e-2,
    phase_max: float = 1,
    dpi: int = 600,
):
    """
    Create faceted plots (small multiples) for grid search results.
    One panel per sequence_length, each showing Sign Agreement vs p with different lines for different taxa.
    
    Args:
        json_path: Path to JSON file with grid results (columns: num_taxa, sequence_length, p, mean, median, std)
        output_path: Path to save the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        phase_min: Minimum x-value for zoom panel
        phase_max: Maximum x-value for zoom panel
        dpi: DPI for the plot
    """
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib import rcParams
    from cycler import cycler
    
    # --- MATLAB-ish palette + trimmed sizes ---
    matlab_colors = [
        (0.0000, 0.4470, 0.7410),
        (0.8500, 0.3250, 0.0980),
        (0.9290, 0.6940, 0.1250),
        (0.4940, 0.1840, 0.5560),
        (0.4660, 0.6740, 0.1880),
        (0.3010, 0.7450, 0.9330),
        (0.6350, 0.0780, 0.1840),
    ]
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.prop_cycle": cycler("color", matlab_colors),
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": dpi, "savefig.dpi": dpi,
    })
    
    # --- load data ---
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if "columns" not in data or "rows" not in data:
        raise ValueError("JSON must have 'columns' and 'rows' keys")
    
    rows = data["rows"]
    cols = data["columns"]
    
    # Group by sequence_length
    seq_lengths = sorted(set(r["sequence_length"] for r in rows))
    taxa_values = sorted(set(r["num_taxa"] for r in rows))
    
    # Build series for each (seq_len, taxa) combination
    series_by_seq = {}  # {seq_len: {f"n={taxa}": {"x": [...], "mean": [...], "std": [...]}}}
    
    for seq_len in seq_lengths:
        series_by_seq[seq_len] = {}
        for n_taxa in taxa_values:
            key = f"n={n_taxa}"
            series_by_seq[seq_len][key] = {"x": [], "mean": [], "std": []}
            
            # Extract rows for this (seq_len, taxa) combination
            relevant_rows = [r for r in rows if r["sequence_length"] == seq_len and r["num_taxa"] == n_taxa]
            
            for r in relevant_rows:
                series_by_seq[seq_len][key]["x"].append(float(r["p"]))
                # Prefer new format (sign_agreement), fall back to median or mean
                series_by_seq[seq_len][key]["mean"].append(float(r.get("sign_agreement", r.get("median", r.get("mean", 0)))))
                series_by_seq[seq_len][key]["std"].append(float(r.get("std", 0)))
            
            # Sort by x
            arr = np.array(list(zip(series_by_seq[seq_len][key]["x"], 
                                    series_by_seq[seq_len][key]["mean"],
                                    series_by_seq[seq_len][key]["std"])), dtype=float)
            arr = arr[arr[:, 0].argsort()]
            series_by_seq[seq_len][key]["x"] = arr[:, 0].tolist()
            series_by_seq[seq_len][key]["mean"] = arr[:, 1].tolist()
            series_by_seq[seq_len][key]["std"] = arr[:, 2].tolist()
    
    # Create subplots: one row per sequence_length, two columns (full + zoom)
    n_facets = len(seq_lengths)
    fig, axes = plt.subplots(n_facets, 2, figsize=(7.2, 3.2 * n_facets))
    
    # Handle case where there's only one facet
    if n_facets == 1:
        axes = axes.reshape(1, -1)
    
    def _plot_panel(ax, series_dict, xlim, seq_len, show_yaxis=True):
        """Plot a single panel with all taxa lines."""
        z_band, z_line = 1, 2
        
        for name, s in series_dict.items():
            x = np.asarray(s["x"])
            y = np.asarray(s["mean"])
            sd = np.asarray(s["std"])
            m = x > 0
            x, y, sd = x[m], y[m], sd[m]
            
            ax.fill_between(x, y - sd, y + sd, alpha=0.08, linewidth=0, zorder=z_band)
            ax.plot(x, y, "-", marker="o", markerfacecolor="white",
                    markeredgewidth=0.9, label=name, zorder=z_line)
        
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(20, 102)
        
        # Labeled powers only
        ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
        labels = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
        t_in = [t for t in ticks if xlim[0] <= t <= xlim[1]]
        l_in = [labels[ticks.index(t)] for t in t_in]
        ax.xaxis.set_major_locator(FixedLocator(t_in))
        ax.xaxis.set_major_formatter(FixedFormatter(l_in))
        ax.minorticks_off()
        
        # Sparser y ticks
        ax.set_yticks([20, 40, 60, 80, 100])
        
        ax.set_xlabel(xlabel, fontsize=8)
        if show_yaxis:
            ax.set_ylabel(ylabel, fontsize=8)
        else:
            ax.yaxis.set_visible(False)
        
        # Add sequence length label as title
        ax.set_title(f"$L = {seq_len}$", fontsize=9, pad=5)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Plot each facet
    for i, seq_len in enumerate(seq_lengths):
        series = series_by_seq[seq_len]
        
        # Full range panel
        _plot_panel(axes[i, 0], series, (1e-4, 1e0), seq_len, show_yaxis=True)
        
        # Zoom panel
        _plot_panel(axes[i, 1], series, (phase_min, phase_max), seq_len, show_yaxis=False)
    
    # Add legend for the first panel only (or we could add for all)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        n = len(labels)
        ncol = n
        fig.legend(
            handles, labels,
            loc="lower center", ncol=ncol,
            frameon=True, fancybox=False, facecolor="white",
            edgecolor=(0.85, 0.85, 0.85), borderpad=0.2,
            handlelength=1.2, columnspacing=1.0, handletextpad=0.5
        )
        # Adjust bottom margin - use smaller increment for more facets
        # For 1-2 facets: 0.08-0.10, for 3-4 facets: 0.06-0.08
        if n_facets <= 2:
            bottom_margin = 0.10
        else:
            # Decrease margin as facets increase (legend is relatively smaller)
            bottom_margin = max(0.05, 0.12 - 0.015 * (n_facets - 1))
        fig.subplots_adjust(bottom=bottom_margin, left=0.10, right=0.98, top=0.97, 
                           hspace=0.3, wspace=0.15)
    else:
        fig.subplots_adjust(bottom=0.10, left=0.10, right=0.98, top=0.97,
                           hspace=0.3, wspace=0.15)
    
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Faceted plot saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage - you can modify these paths as needed
    
    # Plot sign agreement results
    json_file = "/Users/itaygonnen/Python_Repos/spectral-tree-inference/sub_sampled_fielder_vec/results/taxa_sweep_mu_0.3/results_taxa.json"
    output_path = "/Users/itaygonnen/Python_Repos/spectral-tree-inference/sub_sampled_fielder_vec/results/taxa_sweep_mu_0.3/custom_plot.png"
    plot_from_json_simple(
        json_path=json_file,
        output_path=output_path,
        phase_min=1e-2,
        phase_max=1,
    )
    
    # # Plot Fiedler vectors
    # run_dir = "/Users/itaygonnen/Python_Repos/spectral-tree-inference/sub_sampled_fielder_vec/results/20251016-214812-taxa_sweep"
    # fiedler_output = "/Users/itaygonnen/Python_Repos/spectral-tree-inference/sub_sampled_fielder_vec/results/20251016-214812-taxa_sweep/fiedler_vectors.png"
    # plot_fiedler_vectors(
    #     run_dir=run_dir,
    #     output_path=fiedler_output
    # )