import sys, os

sys.path.append("/gpfs/ysm/project/kleinstein/mw957/repos/spectral-tree-inference/spectraltree")

import generation
import reconstruct_tree
import time
import utils
import pandas as pd
import numpy as np
import argparse
import shutil

def run_method(method, size, run_num, tree, m = 300, kappa = 2, mutation_rate=0.05, threshold = None, verbose = False):
    subtree_folder = "/gpfs/ysm/scratch60/morgan_levine/mw957/tree_merge_test/seqlen_" + str(m) + "_" + method + "_" + str(threshold) + "_" + str(run_num) + "/"
    if os.path.exists(subtree_folder):
        shutil.rmtree(subtree_folder)
    os.mkdir(subtree_folder)
    tree.write(path=subtree_folder + "true_tree.txt", schema="newick")
    subtree_filename = subtree_folder + "subtree_%s.txt"
    start_time = time.time()
    observations, taxa_meta = generation.simulate_sequences(m, tree_model=tree, seq_model=generation.Jukes_Cantor(), mutation_rate=mutation_rate, alphabet="DNA")
    runtime = time.time() - start_time
    print("Simulation took %s seconds" % runtime)
    
    spectral_method = reconstruct_tree.STDR(reconstruct_tree.RAxML, reconstruct_tree.JC_similarity_matrix)
    start_time = time.time()
    tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, reconstruct_tree.JC_similarity_matrix, 
                                                            taxa_metadata = taxa_meta, 
                                                            threshhold = threshold,
                                                            raxml_args = "-T 2 --HKY85 -c 1", min_split = 5, 
                                                            verbose = verbose,
                                                            subtree_filename = subtree_filename)
    runtime = time.time() - start_time
    tree_rec.write(path=subtree_folder + "STDR_tree.txt", schema="newick")
    
    distance = reconstruct_tree.JC_distance_matrix(observations, taxa_meta)
    distance_pd = pd.DataFrame(distance)
    taxa_list = [x.label for x in taxa_meta]

    with open(subtree_folder + 'taxa.txt', 'w') as f:
        for item in taxa_list:
            f.write("%s\n" % item)
    distance_pd.index = taxa_list
    distance_path = subtree_folder + "HKY_distance.txt"
    distance_pd.to_csv(distance_path, sep = "\t", header = False)
    with open(distance_path, 'r') as original: data = original.read()
    with open(distance_path, 'w') as modified: modified.write(str(size) + "\n" + data)
    
    # accuracy of the STDR method
    RF,F1 = reconstruct_tree.compare_trees(tree_rec, tree)
    print(method)
    
    if threshold is not None: print(threshold)
    print("--- %s seconds ---" % runtime)
    print("RF = ",RF)
    print("F1% = ",F1) 
    return([method, str(threshold), runtime, RF, F1])

def get_trees(tree_type, tree_size, tree_path):
    if tree_type == "binary":
        tree = utils.balanced_binary(tree_size)
    elif tree_type == "catepillar":
        tree = utils.lopsided_tree(tree_size)
    elif tree_type == "birthdeath":
        tree = utils.unrooted_birth_death_tree(tree_size)
    elif tree_type == "kingman":
        tree = utils.unrooted_pure_kingman_tree(tree_size)
    elif tree_type == "path":
        tree = dendropy.Tree.get(path=args.path, schema="newick")
    return tree

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run different tree reconstruction methods.', )
    parser.add_argument("type", help="tree type: binary, catepillar, kingman, or path (have to provide the path to the tree file).")
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
    i_s = []

    for i in range(n_runs):
        print(method, threshold)
        res = run_method(method, args.size, i, tree, m, kappa, mutation_rate, threshold = threshold, verbose = verbose)
        ms.append(res[0])
        ts.append(res[1])
        rts.append(res[2])
        rfs.append(res[3])
        f1s.append(res[4])
        i_s.append(i)

    perf_metrics = pd.DataFrame({'method': ms, "m": m, "mut_rate": mutation_rate, 'threshold': ts, 'run_num': i_s, 'runtime': rts, 'RF': rfs, "F1": f1s})
    perf_metrics.to_csv("/home/mw957/project/repos/spectral-tree-inference/experiments/results/kingman_subtrees_m" + str(m) + "_" + str(method) + "_" + str(threshold) + ".csv", index=False)
