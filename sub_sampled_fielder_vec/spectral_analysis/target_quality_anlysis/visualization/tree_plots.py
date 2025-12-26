"""
Generate circular tree visualizations with Fiedler vector heatmap rings.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
import toytree
import toyplot.pdf
import toyplot.svg
import dendropy
import tempfile
import os
from PIL import Image


def plot_tree_with_partition(
    tree: dendropy.Tree,
    partition_mask: np.ndarray,
    fiedler_vector: np.ndarray,
    output_path: Path,
    title: str = "Tree Partition",
    stats_dict: dict = None
) -> None:
    """
    Creates a circular tree visualization with Fiedler values as outer heatmap ring.

    Args:
        tree: The dendropy Tree object.
        partition_mask: Boolean array indicating partition membership.
        fiedler_vector: The fiedler vector for coloring.
        output_path: Path to save the plot.
        title: Plot title.
        stats_dict: Dictionary containing metrics for the subtitle.
    """
    import re

    # Convert dendropy tree to toytree using Newick format
    newick_str = tree.as_string(
        schema="newick",
        suppress_internal_node_labels=True,
        suppress_internal_taxon_labels=True
    )
    # Strip out NHX metadata annotations like [&U] that toytree can't parse
    newick_str = re.sub(r'\[&[^\]]*\]', '', newick_str)
    ttree = toytree.tree(newick_str)

    # Get tip labels and create mappings
    tip_labels_list = ttree.get_tip_labels()
    n_tips = len(tip_labels_list)

    # Use actual leaf nodes (not full namespace) to handle trees with extinction
    leaf_taxa = [leaf.taxon for leaf in tree.leaf_node_iter()]
    taxon_to_partition = {taxon.label: mask for taxon, mask in zip(leaf_taxa, partition_mask)}
    taxon_to_fiedler = {taxon.label: val for taxon, val in zip(leaf_taxa, fiedler_vector)}

    # Create ordered arrays for circular layout
    tip_colors = []
    fiedler_values_ordered = []

    for tip_label in tip_labels_list:
        if tip_label in taxon_to_partition:
            is_partition_1 = taxon_to_partition[tip_label]
            color = '#d62728' if is_partition_1 else '#1f77b4'  # Red / Blue
            tip_colors.append(color)
            fiedler_values_ordered.append(taxon_to_fiedler[tip_label])
        else:
            tip_colors.append('black')
            fiedler_values_ordered.append(0.0)

    fiedler_values_ordered = np.array(fiedler_values_ordered)

    # Determine canvas size (square for circular layout)
    canvas_size = max(1200, min(2400, n_tips * 4))  # Scale with tree size

    # ========== Create Circular Tree Visualization ==========
    # Adjust font size based on number of tips
    font_size = max(6, min(10, 200 // max(1, n_tips // 10)))

    canvas, axes, mark = ttree.draw(
        tree_style='c',  # CIRCULAR layout
        tip_labels_colors=tip_colors,
        tip_labels_align=False,
        tip_labels_style={"font-size": font_size, "font-weight": "normal"},
        node_sizes=0,
        edge_style={"stroke": "#262626", "stroke-width": 1.5},
        edge_align_style={"stroke": "none"},
        width=canvas_size,
        height=canvas_size,
    )

    # Save circular tree as standalone files
    toyplot.pdf.render(canvas, str(output_path.with_suffix('.tree.pdf')))
    toyplot.svg.render(canvas, str(output_path.with_suffix('.tree.svg')))

    # ========== Create Separate Fiedler Vector Visualization ==========
    # Simple sorted bar plot showing Fiedler values

    # Sort by Fiedler value for clarity
    sort_indices = np.argsort(fiedler_values_ordered)
    sorted_fiedler = fiedler_values_ordered[sort_indices]
    sorted_labels = [tip_labels_list[i] for i in sort_indices]
    sorted_colors = [tip_colors[i] for i in sort_indices]

    # Determine figure height based on number of taxa
    fig_height = max(8, min(20, n_tips * 0.15))

    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Create horizontal bar plot with clean styling
    y_positions = np.arange(n_tips)
    bars = ax.barh(y_positions, sorted_fiedler, color=sorted_colors, alpha=0.7)

    # Add vertical line at x=0 (partition boundary)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Customize plot
    ax.set_yticks(y_positions)

    # Show labels for small trees, indices for large trees
    if n_tips <= 50:
        ax.set_yticklabels(sorted_labels, fontsize=8)
    else:
        # For large trees, show every Nth label
        label_step = max(1, n_tips // 40)
        ytick_labels = [sorted_labels[i] if i % label_step == 0 else '' for i in range(n_tips)]
        ax.set_yticklabels(ytick_labels, fontsize=7)

    ax.set_xlabel('Fiedler Vector Value', fontsize=11)
    ax.set_ylabel('Taxa (sorted by Fiedler value)', fontsize=11)
    ax.grid(axis='x', alpha=0.2, linestyle=':')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.7, label=f'Partition 1 (n={partition_mask.sum()})'),
        Patch(facecolor='#1f77b4', alpha=0.7, label=f'Partition 0 (n={(~partition_mask).sum()})')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    # Add cleaner title
    if stats_dict:
        p_split = stats_dict.get('partition_split', ('N/A', 'N/A'))
        title_text = f"{title} - Fiedler Vector\nσ₂={stats_dict.get('sigma2', 0):.4f}, Gap={stats_dict.get('spectral_gap', 0):.4f}, Coherence={stats_dict.get('coherence', 0):.4f}"
    else:
        title_text = f"{title} - Fiedler Vector"

    ax.set_title(title_text, fontsize=12, pad=15)

    plt.tight_layout()

    # Save Fiedler plot
    fiedler_plot_path = output_path.with_suffix('.fiedler.pdf')
    fiedler_png_path = output_path.with_suffix('.fiedler.png')
    plt.savefig(fiedler_plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(fiedler_png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Circular tree saved to: {output_path.with_suffix('.tree.pdf')} and {output_path.with_suffix('.tree.svg')}")
    print(f"✓ Fiedler vector plot saved to: {fiedler_plot_path} and {fiedler_png_path}")

    # Always save the raw data
    tree.write(path=output_path.with_suffix(".nwk"), schema="newick")
    np.savetxt(output_path.with_suffix(".fiedler.txt"), fiedler_vector)

    partition_list_path = output_path.with_suffix(".partition.txt")
    with open(partition_list_path, 'w') as f:
        f.write("# Taxon Label, Partition Group (0 or 1), Fiedler Value\n")
        # Use actual leaf nodes (not full namespace) to handle trees with extinction
        leaf_nodes = list(tree.leaf_node_iter())
        for i, leaf in enumerate(leaf_nodes):
            f.write(f"{leaf.taxon.label},{int(partition_mask[i])},{fiedler_vector[i]:.6f}\n")

    print(f"✓ Data files (.nwk, .fiedler.txt, .partition.txt) saved.")


def plot_combined_tree_and_fiedler(
    tree: dendropy.Tree,
    partition_mask: np.ndarray,
    fiedler_vector: np.ndarray,
    output_path: Path,
    title: str = "Tree Partition",
    stats_dict: dict = None
) -> None:
    """
    Creates a combined visualization with vertical tree (left) and Fiedler bars (right).

    Args:
        tree: The dendropy Tree object.
        partition_mask: Boolean array indicating partition membership.
        fiedler_vector: The fiedler vector for coloring.
        output_path: Path to save the plot.
        title: Plot title.
        stats_dict: Dictionary containing metrics for the subtitle.
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    # Get leaf nodes (handling trees with extinction)
    leaf_taxa = [leaf.taxon for leaf in tree.leaf_node_iter()]
    n_tips = len(leaf_taxa)

    # Create mappings
    taxon_to_partition = {taxon.label: mask for taxon, mask in zip(leaf_taxa, partition_mask)}
    taxon_to_fiedler = {taxon.label: val for taxon, val in zip(leaf_taxa, fiedler_vector)}

    # Compute pairwise distances for dendrogram
    # Use patristic distances (branch length distances) from the tree
    pdm = tree.phylogenetic_distance_matrix()

    # Extract distance matrix in the order of leaf_taxa
    dist_matrix = np.zeros((n_tips, n_tips))
    for i, taxon_i in enumerate(leaf_taxa):
        for j, taxon_j in enumerate(leaf_taxa):
            if i != j:
                dist_matrix[i, j] = pdm.distance(taxon_i, taxon_j)

    # Convert to condensed distance matrix for scipy
    condensed_dist = squareform(dist_matrix)

    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(condensed_dist, method='average')

    # Create figure with two subplots
    fig = plt.figure(figsize=(16, max(8, min(20, n_tips * 0.15))))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.3)

    ax_tree = fig.add_subplot(gs[0])
    ax_fiedler = fig.add_subplot(gs[1])

    # ========== LEFT SUBPLOT: Vertical Dendrogram ==========
    # Create dendrogram with custom colors
    dendro = hierarchy.dendrogram(
        linkage_matrix,
        ax=ax_tree,
        orientation='top',
        labels=[taxon.label for taxon in leaf_taxa],
        leaf_font_size=8 if n_tips <= 50 else 6,
        color_threshold=0,
        above_threshold_color='#262626'
    )

    # Get leaf ordering from dendrogram
    leaf_order = dendro['leaves']
    ordered_labels = [leaf_taxa[i].label for i in leaf_order]

    # Color the leaf labels by partition
    leaf_colors = []
    for label in ordered_labels:
        if label in taxon_to_partition:
            is_partition_1 = taxon_to_partition[label]
            color = '#d62728' if is_partition_1 else '#1f77b4'
            leaf_colors.append(color)
        else:
            leaf_colors.append('black')

    # Update x-axis tick colors
    for tick_label, color in zip(ax_tree.get_xticklabels(), leaf_colors):
        tick_label.set_color(color)

    ax_tree.set_ylabel('Patristic Distance', fontsize=11)
    ax_tree.set_xlabel('Taxa', fontsize=11)
    ax_tree.set_title('Phylogenetic Tree', fontsize=12, pad=10)
    ax_tree.spines['top'].set_visible(False)
    ax_tree.spines['right'].set_visible(False)

    # Rotate labels if there are many tips
    if n_tips > 30:
        ax_tree.set_xticklabels(ax_tree.get_xticklabels(), rotation=90, ha='right')
    else:
        ax_tree.set_xticklabels(ax_tree.get_xticklabels(), rotation=45, ha='right')

    # ========== RIGHT SUBPLOT: Fiedler Vector Bars ==========
    # Order Fiedler values to match the tree leaf order
    ordered_fiedler = [taxon_to_fiedler[label] for label in ordered_labels]
    ordered_partition_colors = leaf_colors

    y_positions = np.arange(n_tips)
    ax_fiedler.barh(y_positions, ordered_fiedler, color=ordered_partition_colors, alpha=0.7)

    # Add vertical line at x=0 (partition boundary)
    ax_fiedler.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    ax_fiedler.set_yticks(y_positions)

    # Show labels for small trees
    if n_tips <= 50:
        ax_fiedler.set_yticklabels(ordered_labels, fontsize=8)
    else:
        # For large trees, show every Nth label
        label_step = max(1, n_tips // 40)
        ytick_labels = [ordered_labels[i] if i % label_step == 0 else '' for i in range(n_tips)]
        ax_fiedler.set_yticklabels(ytick_labels, fontsize=7)

    ax_fiedler.set_xlabel('Fiedler Vector Value', fontsize=11)
    ax_fiedler.set_ylabel('Taxa (ordered by tree)', fontsize=11)
    ax_fiedler.grid(axis='x', alpha=0.2, linestyle=':')
    ax_fiedler.set_title('Fiedler Vector', fontsize=12, pad=10)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.7, label=f'Partition 1 (n={partition_mask.sum()})'),
        Patch(facecolor='#1f77b4', alpha=0.7, label=f'Partition 0 (n={(~partition_mask).sum()})')
    ]
    ax_fiedler.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    # ========== Overall Title ==========
    if stats_dict:
        p_split = stats_dict.get('partition_split', ('N/A', 'N/A'))
        suptitle = f"{title}\nσ₂={stats_dict.get('sigma2', 0):.4f}, Gap={stats_dict.get('spectral_gap', 0):.4f}, Coherence={stats_dict.get('coherence', 0):.4f}"
    else:
        suptitle = title

    fig.suptitle(suptitle, fontsize=13, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save combined plot
    combined_plot_path = output_path.with_suffix('.combined.pdf')
    combined_png_path = output_path.with_suffix('.combined.png')
    plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(combined_png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Combined tree+Fiedler plot saved to: {combined_plot_path} and {combined_png_path}")