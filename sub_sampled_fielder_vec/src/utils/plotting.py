# plotting.py
import math
import sys
import json
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

try:  # pragma: no cover - fallback for direct execution
    from src.utils.threshold_utils import compute_phase_transition_thresholds
except ImportError:  # pragma: no cover
    from threshold_utils import compute_phase_transition_thresholds


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
    c_mu  = col("partition_agreement_M", ("sign_agreement", "median", "mean", "y", "value"))  # Partition agreement between averaged f and M
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
    ylabel="Partition agreement (%)",
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
                # Handle both single experiments (no num_taxa field) and taxa/grid experiments
                if "num_taxa" in r:
                    key = f"n={int(r['num_taxa'])}"
                else:
                    key = "single"  # Single experiment, use generic key
                s = series.setdefault(key, {"x": [], "mean": [], "std": []})
                s["x"].append(float(r["p"]))
                # Prefer partition_agreement_M (averaged f vs M), fall back to sign_agreement, then median or mean
                s["mean"].append(float(r.get("partition_agreement_M", r.get("sign_agreement", r.get("median", r.get("mean", 0))))))
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

    # --- figure (single plot, no zoom) ---
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
    ax.set_xlim(1e-4, 1e0)
    ax.set_ylim(20, 102)  # less vertical crowding; tweak if needed

    # labeled powers only
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    labels = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.minorticks_off()

    # sparser y ticks
    ax.set_yticks([20, 40, 60, 80, 100])

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)

    # --- legend outside ---
    handles, labels = ax.get_legend_handles_labels()
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
        fig.subplots_adjust(bottom=0.25, left=0.12, right=0.98, top=0.97)
    else:
        fig.subplots_adjust(bottom=0.10, left=0.12, right=0.98, top=0.97)

    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


def plot_fiedler_vectors(
    run_dir: str,
    output_path: str = "fiedler_vectors.png",
    dpi: int = 600,
    figsize: tuple = (10, 6),
    include_metadata: bool = True,
    fiedler_ref: np.ndarray = None
):
    """
    Plot the true Fiedler vectors for each taxa count.

    Args:
        run_dir: Directory containing the experiment results
        output_path: Path to save the plot
        dpi: DPI for the plot
        facets_per_row: Number of subplots per row (controls grid layout)
        run_title: Optional figure title (e.g., run directory name)
        figsize: Figure size (width, height)
        include_metadata: If True, read config.json and include L and μ in title
        fiedler_ref: Optional Fiedler vector to plot (if provided, will not load from disk)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from cycler import cycler
    import json
    
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
    
    # Read metadata from config.json if available
    seq_len, mu = None, None
    if include_metadata:
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                seq_len = config.get('sequence_length')
                mu = config.get('mutation_rate')
            except Exception as e:
                print(f"Warning: Could not read config.json: {e}")

    # If fiedler_ref is provided directly, use it
    if fiedler_ref is not None:
        # Single Fiedler vector provided in memory
        fiedler_vectors = [fiedler_ref]
        taxa_counts = [len(fiedler_ref)]  # Infer taxa count from vector length

        # Try to get actual taxa count from config
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                taxa_counts = [config.get('num_taxa', len(fiedler_ref))]
            except:
                pass
    else:
        # Fall back to loading from disk (for backward compatibility / standalone plotting)
        fiedler_files = []
        for filename in os.listdir(run_dir):
            if filename.startswith("fiedler_ref") and filename.endswith(".npy"):
                fiedler_files.append(filename)

        if not fiedler_files:
            print(f"No Fiedler vector files found in {run_dir} and no fiedler_ref provided")
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

        # Load vectors from disk
        fiedler_vectors = []
        taxa_counts = []
        for filename in fiedler_files:
            fiedler_path = os.path.join(run_dir, filename)
            fiedler_vectors.append(np.load(fiedler_path))
            taxa_counts.append(extract_taxa_count(filename))

    # Create subplots in a single column
    n_files = len(fiedler_vectors)
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

    for i, (fiedler_vector, taxa_count) in enumerate(zip(fiedler_vectors, taxa_counts)):
        ax = axes[i] if i < len(axes) else axes[-1]
        
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
    for i in range(len(fiedler_vectors), len(axes)):
        axes[i].set_visible(False)

    # Add overall figure title with metadata if available
    if include_metadata and seq_len is not None and mu is not None:
        fig.suptitle(f"Fiedler Vectors (L={seq_len}, μ={mu})",
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])  # Make room for suptitle
    else:
        plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    
    print(f"Fiedler vectors plot saved to: {output_path}")
    return output_path


def plot_fiedler_vectors_grid(
    run_dir: str,
    output_path: str = "fiedler_vectors_grid.png",
    dpi: int = 600,
    figsize_per_subplot: tuple = (3, 2.5),
    include_metadata: bool = True
):
    """
    Plot the Fiedler vectors in a grid organized by N (taxa count) and L (sequence length).
    
    Args:
        run_dir: Directory containing the experiment results
        output_path: Path to save the plot
        dpi: DPI for the plot
        figsize_per_subplot: Size of each subplot (width, height)
        include_metadata: If True, read config.json and include μ in title
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from cycler import cycler
    import json
    from collections import defaultdict
    
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
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": dpi, "savefig.dpi": dpi,
    })
    
    # Read metadata from config.json if available
    mu = None
    if include_metadata:
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                mu = config.get('mutation_rate')
            except Exception as e:
                print(f"Warning: Could not read config.json: {e}")

    # Find all Fiedler vector files and organize by N and L
    fiedler_data = defaultdict(dict)  # {n_value: {L_value: filepath}}
    
    for filename in os.listdir(run_dir):
        if filename.startswith("fiedler_ref") and filename.endswith(".npy") and "_n=" in filename and "_L=" in filename:
            try:
                # Extract n and L from filename like "fiedler_ref_n=1024_L=1000.npy"
                n_part = filename.split('_n=')[1].split('_L=')[0]
                l_part = filename.split('_L=')[1].split('.npy')[0]
                n_value = int(n_part)
                l_value = int(l_part)
                fiedler_data[n_value][l_value] = os.path.join(run_dir, filename)
            except Exception as e:
                print(f"Warning: Could not parse filename {filename}: {e}")
                continue

    if not fiedler_data:
        print(f"No Fiedler vector files with N and L found in {run_dir}")
        return None
    
    # Get sorted unique values
    n_values = sorted(fiedler_data.keys())
    all_l_values = set()
    for n_dict in fiedler_data.values():
        all_l_values.update(n_dict.keys())
    l_values = sorted(all_l_values)
    
    print(f"Found {len(n_values)} N values: {n_values}")
    print(f"Found {len(l_values)} L values: {l_values}")
    
    # Create grid: rows = N values, columns = L values
    n_rows = len(n_values)
    n_cols = len(l_values)
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
        squeeze=False
    )
    
    # Plot each combination
    for i, n_val in enumerate(n_values):
        for j, l_val in enumerate(l_values):
            ax = axes[i, j]
            
            # Check if this combination exists
            if l_val in fiedler_data[n_val]:
                fiedler_path = fiedler_data[n_val][l_val]
                fiedler_vector = np.load(fiedler_path)
                
                # Plot the Fiedler vector
                x = np.arange(len(fiedler_vector))
                ax.plot(x, fiedler_vector, '-', linewidth=0.8, alpha=0.8)
                
                # Styling
                ax.set_title(f'N={n_val}, L={l_val}', fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.2, linewidth=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                
                # Set reasonable y-limits
                y_min, y_max = fiedler_vector.min(), fiedler_vector.max()
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                
                # Only show x-label on bottom row
                if i == n_rows - 1:
                    ax.set_xlabel('Taxon Index', fontsize=8)
                
                # Only show y-label on leftmost column
                if j == 0:
                    ax.set_ylabel('Fiedler Value', fontsize=8)
            else:
                # No data for this combination
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
    
    # Add overall figure title with metadata if available
    if include_metadata and mu is not None:
        fig.suptitle(f"Fiedler Vectors (μ={mu})",
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Make room for suptitle
    else:
        plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    
    print(f"Fiedler vectors grid plot saved to: {output_path}")
    return output_path


def plot_faceted_by_sequence_length(
    json_path: str,
    output_path: str = "plot_faceted.png",
    xlabel: str = r"$p$",
    ylabel: str = "Partition agreement (%)",
    phase_min: float = 1e-2,
    phase_max: float = 1,
    dpi: int = 600,
    *,
    facets_per_row: int = 1,
    run_title: Optional[str] = None,
):
    """
    Create faceted plots (small multiples) for grid search results.
    One panel per sequence_length, each showing Partition Agreement vs p with different lines for different taxa.
    
    Args:
        json_path: Path to JSON file with grid results (columns: num_taxa, sequence_length, p, mean, median, std)
        output_path: Path to save the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        phase_min: Minimum x-value for zoom panel
        phase_max: Maximum x-value for zoom panel
        dpi: DPI for the plot
        facets_per_row: Number of subplots per row (1 keeps the legacy single column)
        run_title: Optional figure title (e.g., run directory name)
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
    
    def _format_p_value(value: float) -> str:
        if value >= 0.1:
            return f"{value:.2f}"
        if value >= 0.01:
            return f"{value:.3f}"
        return f"{value:.2g}"

    def _taxa_sort_key(label: str):
        try:
            return (0, int(label.split("=")[1]))
        except (ValueError, IndexError):
            return (1, label)

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
                # Prefer partition_agreement_M (averaged f vs M), fall back to sign_agreement, then median or mean
                val = r.get("partition_agreement_M", r.get("sign_agreement", r.get("median", r.get("mean", 0))))
                series_by_seq[seq_len][key]["mean"].append(float(val))
                series_by_seq[seq_len][key]["std"].append(float(r.get("std", 0)))
            
            # Sort by x (skip if no data for this combination)
            if not series_by_seq[seq_len][key]["x"]:
                continue
                
            arr = np.array(list(zip(series_by_seq[seq_len][key]["x"], 
                                    series_by_seq[seq_len][key]["mean"],
                                    series_by_seq[seq_len][key]["std"])), dtype=float)
            arr = arr[arr[:, 0].argsort()]
            
            # Fill NaN values with 50% for p < 1e-3 (left side of plot)
            x_vals = arr[:, 0]
            y_vals = arr[:, 1]
            std_vals = arr[:, 2]
            
            nan_mask = np.isnan(y_vals)
            left_mask = x_vals < 1e-3
            fill_mask = nan_mask & left_mask
            y_vals[fill_mask] = 50.0
            std_vals[fill_mask] = 0.0
            
            series_by_seq[seq_len][key]["x"] = x_vals.tolist()
            series_by_seq[seq_len][key]["mean"] = y_vals.tolist()
            series_by_seq[seq_len][key]["std"] = std_vals.tolist()
    
    # Create subplots with configurable layout
    n_facets = len(seq_lengths)
    ncols = max(1, facets_per_row)
    nrows = math.ceil(n_facets / ncols) or 1

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.2 * nrows),
    )

    axes_array = np.atleast_1d(axes).flatten()
    
    def _plot_panel(ax, series_dict, seq_len):
        """Plot a single panel with all taxa lines."""
        z_band, z_line = 1, 2
        line_colors = {}
        filtered_series = {}

        for name, s in series_dict.items():
            x = np.asarray(s["x"])
            y = np.asarray(s["mean"])
            sd = np.asarray(s["std"])
            m = x >= 1e-3
            x, y, sd = x[m], y[m], sd[m]

            filtered_series[name] = {"x": x.tolist(), "mean": y.tolist()}

            line, = ax.plot(
                x,
                y,
                "-",
                marker="o",
                markerfacecolor="white",
                markeredgewidth=0.9,
                label=name,
                zorder=z_line,
            )
            color = line.get_color()
            line_colors[name] = color

            if x.size:
                ax.fill_between(
                    x,
                    y - sd,
                    y + sd,
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                    zorder=z_band,
                )

        thresholds = compute_phase_transition_thresholds(filtered_series)
        annotation_entries = []

        for name, threshold in thresholds.items():
            if not threshold:
                continue
            p_val, y_val = threshold
            color = line_colors.get(name)
            if color is None:
                continue
            ax.scatter(
                [p_val],
                [y_val],
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=z_line + 1,
                s=22,
            )
            annotation_entries.append((name, p_val, color))

        annotation_entries.sort(key=lambda entry: _taxa_sort_key(entry[0]))

        if annotation_entries:
            base_x = 0.98
            base_y = 0.05
            line_height = 0.055

            for idx, (label, p_val, color) in enumerate(annotation_entries):
                ax.text(
                    base_x,
                    base_y + idx * line_height,
                    f"{label} p50={_format_p_value(p_val)}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=6,
                    color=color,
                )
        
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1e0)
        ax.set_ylim(20, 102)
        
        # Labeled powers only
        ticks = [1e-3, 1e-2, 1e-1, 1e0]
        labels = [r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
        ax.minorticks_off()
        
        # Sparser y ticks
        ax.set_yticks([20, 40, 60, 80, 100])
        
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        
        # Add sequence length label as title
        ax.set_title(f"$L = {seq_len}$", fontsize=9, pad=5)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Plot each facet
    for i, seq_len in enumerate(seq_lengths):
        series = series_by_seq[seq_len]
        _plot_panel(axes_array[i], series, seq_len)

    # Hide any unused panels
    for ax in axes_array[n_facets:]:
        ax.set_visible(False)

    # Add legend
    handles, labels = [], []
    for ax in axes_array[:n_facets]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break

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
        # Adjust bottom margin
        if n_facets <= 2:
            bottom_margin = 0.10
        else:
            bottom_margin = max(0.05, 0.12 - 0.015 * (n_facets - 1))
        top_margin = 0.96 if run_title else 0.97
        fig.subplots_adjust(
            bottom=bottom_margin,
            left=0.12,
            right=0.98,
            top=top_margin,
            hspace=0.3,
            wspace=0.18,
        )
    else:
        top_margin = 0.96 if run_title else 0.97
        fig.subplots_adjust(
            bottom=0.10,
            left=0.12,
            right=0.98,
            top=top_margin,
            hspace=0.3,
            wspace=0.18,
        )

    if run_title:
        fig.suptitle(run_title, fontsize=11, y=0.995)

    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Faceted plot saved to: {output_path}")
    return output_path


def _extract_model_name(dirname: str) -> str:
    """Extract model name from directory like '20251227-152754-kingman_mean_taxa_sweep_L10k_Ne1'"""
    import re
    
    # Model name mappings
    model_mappings = {
        "kingman": "Kingman Coalescent",
        "birthdeath": "Birth-Death",
        "caterpillar": "Caterpillar",
    }
    
    # Extract the model name from directory name
    # Pattern: look for common model names in the directory name
    dirname_lower = dirname.lower()
    for key, value in model_mappings.items():
        if key in dirname_lower:
            return value
    
    # If no match, try to extract the first meaningful word after timestamp
    # Pattern: YYYYMMDD-HHMMSS-<model_name>... or YYYYMMDD-HHMMSS-<model_name>_...
    match = re.search(r'\d{8}-\d{6}-([a-z]+)', dirname_lower)
    if match:
        model_word = match.group(1)
        # Capitalize first letter
        return model_word.capitalize()
    
    # Fallback: return a default
    return "Tree Model"


def plot_taxa_sweep(
    json_path: str,
    output_path: str = "partition_agreement.png",
    model_name: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: str = r"$p$",
    ylabel: str = "Partition agreement (%)",
    dpi: int = 600,
):
    """
    Create a single-panel plot for taxa sweep results (single L value).
    
    Args:
        json_path: Path to JSON file with grid results (columns: num_taxa, sequence_length, p, partition_agreement_M, etc.)
        output_path: Path to save the plot
        model_name: Model name for the title (e.g., "Kingman Coalescent")
        subtitle: Subtitle with details (e.g., "L = 10000, 10 bootstrap reps")
        xlabel: Label for x-axis
        ylabel: Label for y-axis
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
    
    def _format_p_value(value: float) -> str:
        if value >= 0.1:
            return f"{value:.2f}"
        if value >= 0.01:
            return f"{value:.3f}"
        return f"{value:.2g}"

    def _taxa_sort_key(label: str):
        try:
            return (0, int(label.split("=")[1]))
        except (ValueError, IndexError):
            return (1, label)

    # --- load data ---
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if "columns" not in data or "rows" not in data:
        raise ValueError("JSON must have 'columns' and 'rows' keys")
    
    rows = data["rows"]
    
    # Get unique sequence lengths and taxa values
    seq_lengths = sorted(set(r["sequence_length"] for r in rows))
    taxa_values = sorted(set(r["num_taxa"] for r in rows))
    
    # For single-L sweeps, we expect only one sequence length
    if len(seq_lengths) > 1:
        raise ValueError(f"Expected single sequence length, found {len(seq_lengths)}: {seq_lengths}")
    
    seq_len = seq_lengths[0]
    
    # Build series for each taxa value
    series = {}
    for n_taxa in taxa_values:
        key = f"n={n_taxa}"
        series[key] = {"x": [], "mean": [], "std": []}
        
        # Extract rows for this taxa value
        relevant_rows = [r for r in rows if r["num_taxa"] == n_taxa]
        
        for r in relevant_rows:
            series[key]["x"].append(float(r["p"]))
            # Prefer partition_agreement_M (averaged f vs M), fall back to sign_agreement
            val = r.get("partition_agreement_M", r.get("sign_agreement", r.get("median", r.get("mean", 0))))
            series[key]["mean"].append(float(val))
            series[key]["std"].append(float(r.get("std", 0)))
        
        # Sort by x
        if not series[key]["x"]:
            continue
            
        arr = np.array(list(zip(series[key]["x"], 
                                series[key]["mean"],
                                series[key]["std"])), dtype=float)
        arr = arr[arr[:, 0].argsort()]
        
        # Fill NaN values with 50% for p < 1e-3 (left side of plot)
        x_vals = arr[:, 0]
        y_vals = arr[:, 1]
        std_vals = arr[:, 2]
        
        nan_mask = np.isnan(y_vals)
        left_mask = x_vals < 1e-3
        fill_mask = nan_mask & left_mask
        y_vals[fill_mask] = 50.0
        std_vals[fill_mask] = 0.0
        
        series[key]["x"] = x_vals.tolist()
        series[key]["mean"] = y_vals.tolist()
        series[key]["std"] = std_vals.tolist()
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2))
    
    z_band, z_line = 1, 2
    line_colors = {}
    filtered_series = {}

    # Plot each taxa line
    for name, s in series.items():
        x = np.asarray(s["x"])
        y = np.asarray(s["mean"])
        sd = np.asarray(s["std"])
        m = x >= 1e-3
        x, y, sd = x[m], y[m], sd[m]

        filtered_series[name] = {"x": x.tolist(), "mean": y.tolist()}

        line, = ax.plot(
            x,
            y,
            "-",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=name,
            zorder=z_line,
        )
        color = line.get_color()
        line_colors[name] = color

        if x.size:
            ax.fill_between(
                x,
                y - sd,
                y + sd,
                color=color,
                alpha=0.08,
                linewidth=0,
                zorder=z_band,
            )

    # Compute and plot phase transition thresholds
    thresholds = compute_phase_transition_thresholds(filtered_series)
    annotation_entries = []

    for name, threshold in thresholds.items():
        if not threshold:
            continue
        p_val, y_val = threshold
        color = line_colors.get(name)
        if color is None:
            continue
        ax.scatter(
            [p_val],
            [y_val],
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=z_line + 1,
            s=22,
        )
        annotation_entries.append((name, p_val, color))

    annotation_entries.sort(key=lambda entry: _taxa_sort_key(entry[0]))

    if annotation_entries:
        base_x = 0.98
        base_y = 0.05
        line_height = 0.055

        for idx, (label, p_val, color) in enumerate(annotation_entries):
            ax.text(
                base_x,
                base_y + idx * line_height,
                f"{label} p50={_format_p_value(p_val)}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                color=color,
            )
    
    # Set axes properties
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e0)
    ax.set_ylim(20, 102)
    
    # Labeled powers only
    ticks = [1e-3, 1e-2, 1e-1, 1e0]
    labels = [r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.minorticks_off()
    
    # Sparser y ticks
    ax.set_yticks([20, 40, 60, 80, 100])
    
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add legend in lower right corner, single column
    ax.legend(
        loc="lower right",
        ncol=1,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor=(0.85, 0.85, 0.85),
        borderpad=0.4,
        handlelength=1.2,
        handletextpad=0.5,
    )
    
    # Set title and subtitle with proper spacing
    if model_name:
        ax.set_title(model_name, fontsize=11, pad=15)
        if subtitle:
            # Add subtitle using text with increased spacing
            # Use LaTeX rendering for mathematical symbols
            ax.text(0.5, 1.05, subtitle, transform=ax.transAxes,
                   ha="center", va="bottom", fontsize=9,
                   style="italic", color="gray")
    elif subtitle:
        ax.set_title(subtitle, fontsize=9, pad=10)
    
    # Adjust layout - increase top margin to accommodate subtitle
    top_margin = 0.88 if (model_name and subtitle) else 0.92
    fig.subplots_adjust(
        bottom=0.12,
        left=0.12,
        right=0.95,
        top=top_margin,
    )

    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Taxa sweep plot saved to: {output_path}")
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