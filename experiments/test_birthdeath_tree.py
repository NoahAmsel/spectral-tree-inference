import spectraltree
import time
import pandas as pd
import argparse

#@profile
def run_method(method, tree, m = 300, kappa = 2, mutation_rate=0.05, threshold = None, verbose = False):
    start_time = time.time()
    observations, taxa_meta = spectraltree.simulate_sequences(m, tree_model=tree, seq_model=spectraltree.HKY(kappa = kappa), mutation_rate=mutation_rate, alphabet="DNA")
    HKY_sim = spectraltree.HKY_similarity_matrix(taxa_meta)
    runtime = time.time() - start_time
    print("Simulation took %s seconds" % runtime)
    
    if method == "RAxML":
        raxml_HKY = spectraltree.RAxML()
        start_time = time.time()
        tree_rec = raxml_HKY(observations, taxa_meta, raxml_args="-T 2 --HKY85 -c 1")      
    if method == "SNJ":
        snj = spectraltree.SpectralNeighborJoining(HKY_sim)
        start_time = time.time()
        tree_rec = snj(observations, taxa_meta)
    if method == "NJ":
        nj = spectraltree.NeighborJoining(HKY_sim)
        start_time = time.time()
        tree_rec = nj(observations, taxa_meta)
    if method == "STDR+NJ":
        spectral_method = spectraltree.STDR(spectraltree.NeighborJoining, HKY_sim)
        start_time = time.time()
        tree_rec = spectral_method(observations, merge_method="global_diff",
                                                            taxa_metadata = taxa_meta,
                                                            threshold = threshold, min_split = 5, verbose = verbose)
    if method == "STDR+SNJ":
        spectral_method = spectraltree.STDR(spectraltree.SpectralNeighborJoining, HKY_sim)
        start_time = time.time()
        tree_rec = spectral_method(observations,
                                   merge_method="global_diff",
                                                            taxa_metadata = taxa_meta, 
                                                            threshold = threshold, min_split = 5, verbose = verbose)
    if method == "STDR+RAxML":
        raxml = spectraltree.RAxML(raxml_args = "-T 2 --HKY85 -c 1")
        spectral_method = spectraltree.STDR(raxml, HKY_sim)
        start_time = time.time()
        tree_rec = spectral_method(observations, merge_method="global_diff",
                                                            taxa_metadata = taxa_meta, 
                                                            threshold = threshold,
                                                            min_split = 5, verbose = verbose)
    runtime = time.time() - start_time
    RF,F1 = spectraltree.compare_trees(tree_rec, tree)
    print(method)
    if threshold is not None: print(threshold)
    print("--- %s seconds ---" % runtime)
    print("RF = ",RF)
    print("F1% = ",F1) 
    return([method, str(threshold), runtime, RF, F1])

#@profile
def get_trees(tree_type, tree_size, tree_path):
    if tree_type == "binary":
        tree = spectraltree.balanced_binary(tree_size)
    elif tree_type == "catepillar":
        tree = spectraltree.lopsided_tree(tree_size)
    elif tree_type == "birthdeath":
        tree = spectraltree.unrooted_birth_death_tree(tree_size)
    elif tree_type == "kingman":
        tree = spectraltree.unrooted_pure_kingman_tree(tree_size)
    elif tree_type == "path":
        tree = dendropy.Tree.get(path=args.path, schema="newick")
    return tree

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run different tree reconstruction methods.', )
    parser.add_argument("type", help="tree type: binary, catepillar, or path (have to provide the path to the tree file).")
    parser.add_argument('method', help='method to run: RaXML, SNJ, NJ, STR+NJ, STR+SNJ, STR+RaXML.')
    parser.add_argument('nrun', type=int, help="Number of times to run the method.")
    parser.add_argument("--size", type=int, help="Size of the tree.")
    parser.add_argument("--path", help="Path to the tree file.")
    parser.add_argument('--threshold', type=int, help='Minimum tree size to run the submethod for STR methods.')
    parser.add_argument("--m", type=int, help="Length of the sequence.", default=300)
    parser.add_argument("--kappa", type=float, help="Transversion/transition rate ratio", default=2)
    parser.add_argument("--mutation_rate", type=float, help="Mutation rate", default=0.05)
    parser.add_argument("--verbose", help="Whether to print the diagnostic messages.")

    args = parser.parse_args()

    tree = get_trees(args.type, args.size, args.path)
    n_runs = args.nrun
    method = args.method
    threshold = args.threshold
    verbose = args.verbose == "True"
    m = args.m
    kappa = args.kappa
    mutation_rate = args.mutation_rate

    ms = []
    ts = []
    rts = []
    rfs = []
    f1s = []

    for i in range(n_runs):
        print(method, threshold)
        res = run_method(method, tree, m, kappa, mutation_rate, threshold = threshold, verbose = verbose)
        ms.append(res[0])
        ts.append(res[1])
        rts.append(res[2])
        rfs.append(res[3])
        f1s.append(res[4])

    perf_metrics = pd.DataFrame({'method': ms, "m": m, "mut_rate": mutation_rate, 'threshold': ts, 'runtime': rts, 'RF': rfs, "F1": f1s})
    perf_metrics.to_csv("/gpfs/ysm/project/kleinstein/mw957/repos/spectral-tree-inference/experiments/results/birthdeath_global_diff_m" + str(m) + "_" + str(method) + "_" + str(threshold) + ".csv", index=False)
