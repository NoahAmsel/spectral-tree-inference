"""
Mutation Rate Fiedler Validation Visualization

This script generates a figure showing how Fiedler vectors behave across different 
mutation rates, with plateau validation metrics displayed for each rate.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import spectraltree

# Add parent directory to path for imports
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from utils.utils import generate_sequences, compute_fielder_vector
from fiedler_plateau_validator import assess_fiedler_vector

# Configuration
N_TAXA = 1024
SEQ_LEN = 1024
MUT_RATES = [0.1, 0.3, 0.5, 0.7, 0.9]
SEED = 42

# Set random seed for reproducibility
np.random.seed(SEED)


def simulate_fiedler_for_rate(mutation_rate, n_taxa, seq_len):
    """
    Generate sequences and compute Fiedler vector with plateau assessment.
    
    Args:
        mutation_rate: Mutation rate for sequence generation
        n_taxa: Number of taxa (leaves in tree)
        seq_len: Sequence length
        
    Returns:
        Tuple of (fiedler_vector, assessment_dict)
    """
    # 1. Generate tree
    tree = spectraltree.balanced_binary(n_taxa)
    
    # 2. Generate sequences
    seq_model = spectraltree.Jukes_Cantor()
    observations = generate_sequences(
        num_taxa=n_taxa,
        sequence_length=seq_len,
        mutation_rate=mutation_rate,
        tree_model=tree,
        seq_model=seq_model
    )
    
    # 3. Compute similarity matrix
    similarity_matrix = spectraltree.JC_similarity_matrix(observations)
    
    # 4. Compute Fiedler vector
    fiedler_vec = compute_fielder_vector(similarity_matrix)
    
    # 5. Assess Fiedler vector with plateau validator
    assessment = assess_fiedler_vector(fiedler_vec, W=similarity_matrix)
    
    return fiedler_vec, assessment


def plot_panel(ax, mu, fiedler_vec, assessment):
    """
    Plot a single panel showing Fiedler vector with plateau assessment.
    
    Args:
        ax: Matplotlib axis object
        mu: Mutation rate value
        fiedler_vec: Fiedler vector to plot
        assessment: Assessment dictionary from assess_fiedler_vector
    """
    n = len(fiedler_vec)
    
    # Sort Fiedler vector to identify plateaus
    order = np.argsort(fiedler_vec)
    fiedler_sorted = fiedler_vec[order]
    
    # Get assessment metrics
    threshold = assessment['threshold']
    means = assessment['means']
    sizes = assessment['sizes']
    step_R2 = assessment['step_R2']
    cohens_d = assessment['cohens_d']
    is_valid = assessment['is_valid']
    
    # Get conductance if available
    conductance = assessment.get('conductance', None)
    
    # Find split index in sorted array
    # The assessment was done on the sorted vector, so we need to reconstruct the split
    threshold_idx = np.searchsorted(fiedler_sorted, threshold)
    
    # Plot Fiedler vector as continuous blue dotted curve in sorted order
    x = np.arange(n)
    ax.plot(x, fiedler_sorted, 'b.', markersize=1.5, linestyle=':', linewidth=0.8, alpha=0.7)
    
    # Draw horizontal dashed lines for plateau means (BOLDER)
    if means[0] is not None and means[1] is not None:
        # Left plateau mean (first part of sorted array)
        if threshold_idx > 0:
            ax.axhline(y=means[0], color='r', linestyle='--', linewidth=3.0, 
                       xmin=0, xmax=threshold_idx/n, alpha=0.9, label='Left plateau mean')
        # Right plateau mean (second part of sorted array)
        if threshold_idx < n:
            ax.axhline(y=means[1], color='r', linestyle='--', linewidth=3.0, 
                       xmin=threshold_idx/n, xmax=1, alpha=0.9, label='Right plateau mean')
    
    # Draw vertical line at threshold
    if threshold_idx > 0 and threshold_idx < n:
        ax.axvline(x=threshold_idx, color='g', linestyle=':', linewidth=1, alpha=0.5)
    
    # Set title with metrics
    conductance_str = f", Conductance={conductance:.3f}" if conductance is not None else ""
    title = f"μ = {mu} | R²={step_R2:.3f}, d={cohens_d:.2f}{conductance_str}, Valid={is_valid}"
    ax.set_title(title, fontsize=10, pad=6)
    
    # Set labels
    ax.set_xlabel("Taxon Index (sorted by Fiedler value)", fontsize=9)
    ax.set_ylabel("Fiedler Vector Value", fontsize=9)
    
    # Set grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add metric box in corner
    info_text = f"Split: {threshold_idx}/{n}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))


def main():
    """Run the mutation rate validation experiment."""
    
    # Create figure with 5 vertically stacked subplots
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), constrained_layout=True)
    
    # If only one subplot returned as scalar
    if not hasattr(axes, '__len__'):
        axes = [axes]
    
    print("Generating Fiedler vectors for different mutation rates...")
    print("="*60)
    
    # Process each mutation rate
    for i, mu in enumerate(MUT_RATES):
        print(f"Processing μ = {mu}... ", end='', flush=True)
        
        # Generate Fiedler vector and assessment
        fiedler_vec, assessment = simulate_fiedler_for_rate(mu, N_TAXA, SEQ_LEN)
        
        # Plot panel
        plot_panel(axes[i], mu, fiedler_vec, assessment)
        
        print(f"Done (R²={assessment['step_R2']:.3f}, Valid={assessment['is_valid']})")
    
    # Overall figure title
    fig.suptitle(f"Fiedler Vector Plateau Validation (n={N_TAXA} taxa, seq_len={SEQ_LEN})", 
                 fontsize=12, fontweight='bold')
    
    print("="*60)
    print("Visualization complete. Displaying plot...")
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), "mutation_rate_fiedler_validation.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Display the figure
    plt.show()


if __name__ == "__main__":
    main()

