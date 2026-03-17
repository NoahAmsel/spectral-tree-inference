"""
Ad-hoc script to check partition quality (σ₂ score) for any tree/similarity matrix.

Usage:
    python check_partition_quality.py --n_taxa 8192 --seq_len 1000 --mutation_rate 0.1
    
Or analyze existing similarity matrix:
    python check_partition_quality.py --matrix_file path/to/matrix.npy
"""
import argparse
import numpy as np
import spectraltree
from utils.utils import generate_sequences, compute_laplacian
from utils.random_entries import compute_fiedler_from_laplacian
from utils.metrics import compute_reference_partition_and_quality
from utils.experiment_config import set_seed


def check_quality_from_tree(n_taxa: int, seq_len: int, mutation_rate: float, 
                            num_gaps: int = 1, min_split: int = 1, seed: int = 42):
    """Generate tree, compute reference partition quality."""
    print(f"Generating tree with {n_taxa} taxa, sequence length {seq_len}, μ={mutation_rate}")
    set_seed(seed)
    
    # Generate data
    tree = spectraltree.balanced_binary(n_taxa)
    model = spectraltree.Jukes_Cantor()
    observations = generate_sequences(tree, seq_len, model, mutation_rate, seed)
    M = spectraltree.utils.similarity_matrix(observations, "JC")
    
    print(f"Similarity matrix: {M.shape}, mean={M.mean():.4f}, std={M.std():.4f}")
    
    # Compute reference Fiedler vector
    L_M = compute_laplacian(M)
    fiedler_ref = compute_fiedler_from_laplacian(L_M)
    
    print(f"Fiedler vector: mean={fiedler_ref.mean():.4f}, std={fiedler_ref.std():.4f}")
    print(f"Sign distribution: {np.sum(fiedler_ref > 0)} positive, {np.sum(fiedler_ref < 0)} negative")
    
    # Compute partition quality
    partition_ref, quality = compute_reference_partition_and_quality(
        fiedler_ref, M, num_gaps, min_split
    )
    
    print(f"\n{'='*60}")
    print(f"Reference Partition Quality (σ₂): {quality:.6f}")
    print(f"{'='*60}")
    print(f"\nPartition:")
    print(f"  - Group A: {np.sum(partition_ref)} taxa")
    print(f"  - Group B: {np.sum(~partition_ref)} taxa")
    print(f"\nInterpretation:")
    print(f"  - Lower σ₂ = better partition (cleaner separation)")
    print(f"  - σ₂ ≈ 0: Perfect rank-1 structure (ideal)")
    print(f"  - σ₂ >> 0: High rank structure (poor partition)")
    if quality == float('inf'):
        print(f"  - WARNING: Partition failed (no valid split found)")
    
    return quality, M, fiedler_ref, partition_ref


def check_quality_from_matrix(matrix_path: str, num_gaps: int = 1, min_split: int = 1):
    """Load matrix, compute reference partition quality."""
    print(f"Loading similarity matrix from {matrix_path}")
    M = np.load(matrix_path)
    
    print(f"Similarity matrix: {M.shape}, mean={M.mean():.4f}, std={M.std():.4f}")
    
    # Compute reference Fiedler vector
    L_M = compute_laplacian(M)
    fiedler_ref = compute_fiedler_from_laplacian(L_M)
    
    print(f"Fiedler vector: mean={fiedler_ref.mean():.4f}, std={fiedler_ref.std():.4f}")
    print(f"Sign distribution: {np.sum(fiedler_ref > 0)} positive, {np.sum(fiedler_ref < 0)} negative")
    
    # Compute partition quality
    partition_ref, quality = compute_reference_partition_and_quality(
        fiedler_ref, M, num_gaps, min_split
    )
    
    print(f"\n{'='*60}")
    print(f"Reference Partition Quality (σ₂): {quality:.6f}")
    print(f"{'='*60}")
    print(f"\nPartition:")
    print(f"  - Group A: {np.sum(partition_ref)} taxa")
    print(f"  - Group B: {np.sum(~partition_ref)} taxa")
    print(f"\nInterpretation:")
    print(f"  - Lower σ₂ = better partition (cleaner separation)")
    print(f"  - σ₂ ≈ 0: Perfect rank-1 structure (ideal)")
    print(f"  - σ₂ >> 0: High rank structure (poor partition)")
    if quality == float('inf'):
        print(f"  - WARNING: Partition failed (no valid split found)")
    
    return quality, M, fiedler_ref, partition_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check partition quality for tree or matrix")
    parser.add_argument("--n_taxa", type=int, help="Number of taxa (for tree generation)")
    parser.add_argument("--seq_len", type=int, help="Sequence length (for tree generation)")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate (default: 0.1)")
    parser.add_argument("--matrix_file", type=str, help="Path to similarity matrix .npy file")
    parser.add_argument("--num_gaps", type=int, default=1, help="Number of gaps to evaluate (default: 1)")
    parser.add_argument("--min_split", type=int, default=1, help="Minimum partition size (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    if args.matrix_file:
        check_quality_from_matrix(args.matrix_file, args.num_gaps, args.min_split)
    elif args.n_taxa and args.seq_len:
        check_quality_from_tree(args.n_taxa, args.seq_len, args.mutation_rate, 
                               args.num_gaps, args.min_split, args.seed)
    else:
        print("Error: Must specify either --matrix_file OR (--n_taxa AND --seq_len)")
        parser.print_help()

