import sys, os

sys.path.append("/gpfs/ysm/project/kleinstein/mw957/repos/spectral-tree-inference/spectraltree")

import generation
import reconstruct_tree
import time
import utils
import pandas as pd
import dendropy

def run_method(method, tree, m = 300, kappa = 2, mutation_rate=0.05, threshold = None, verbose = False):
    start_time = time.time()
    observations, taxa_meta = generation.simulate_sequences(m, tree_model=tree, seq_model=generation.HKY(kappa = kappa), mutation_rate=mutation_rate, alphabet="DNA")
    runtime = time.time() - start_time
    print("Simulation took %s seconds" % runtime)
    
    if method == "RaXML":
        raxml_HKY = reconstruct_tree.RAxML()
        start_time = time.time()
        tree_rec = raxml_HKY(observations, taxa_meta, raxml_args="-T 2 --HKY85 -c 1")      
    if method == "SNJ":
        snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.HKY_similarity_matrix)
        start_time = time.time()
        tree_rec = snj(observations, taxa_meta)
    if method == "NJ":
        nj = reconstruct_tree.NeighborJoining(reconstruct_tree.HKY_similarity_matrix)
        start_time = time.time()
        tree_rec = nj(observations, taxa_meta)
    if method == "STR+NJ":
        spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.NeighborJoining, reconstruct_tree.HKY_similarity_matrix)
        start_time = time.time()
        tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, reconstruct_tree.HKY_similarity_matrix, 
                                                            taxa_metadata = taxa_meta,
                                                            threshhold = threshold, min_split = 5, verbose = verbose)
    if method == "STR+SNJ":
        spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.SpectralNeighborJoining, reconstruct_tree.HKY_similarity_matrix)
        start_time = time.time()
        tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, reconstruct_tree.HKY_similarity_matrix, 
                                                            taxa_metadata = taxa_meta, 
                                                            threshhold = threshold, min_split = 5, verbose = verbose)
    if method == "STR+RaXML":
        spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.RAxML, reconstruct_tree.HKY_similarity_matrix)
        start_time = time.time()
        tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, reconstruct_tree.HKY_similarity_matrix, 
                                                            taxa_metadata = taxa_meta, 
                                                            threshhold = threshold,
                                                            raxml_args = "-T 2 --HKY85 -c 1", min_split = 5, verbose = verbose)
    runtime = time.time() - start_time
    RF,F1 = reconstruct_tree.compare_trees(tree_rec, tree)
    print(method)
    if threshold is not None: print(threshold)
    print("--- %s seconds ---" % runtime)
    print("RF = ",RF)
    print("F1% = ",F1) 
    return([method, str(threshold), runtime, RF, F1])




n = 2000
n_runs = 10

methods = ["RaXML", "SNJ", "NJ", "STR+NJ", "STR+NJ", "STR+NJ", "STR+SNJ", "STR+SNJ", "STR+SNJ", "STR+RaXML", "STR+RaXML", "STR+RaXML"]
thresholds = [None, None, None, 32, 64, 128, 32, 64, 128, 32, 64, 128]

ms = []
ts = []
rts = []
rfs = []
f1s = []

for i in range(n_runs):
    for j in range(len(methods)):
        method = methods[j]
        threshold = thresholds[j]
        print(method, threshold)
        kingman_tree = utils.unrooted_pure_kingman_tree(n)
        res = run_method(method, kingman_tree, m = 1000, mutation_rate=0.1, threshold = threshold)
        ms.append(res[0])
        ts.append(res[1])
        rts.append(res[2])
        rfs.append(res[3])
        f1s.append(res[4])

perf_metrics = pd.DataFrame({'method': ms, 'threshold': ts, 'runtime': rts, 'RF': rfs, "F1": f1s})
perf_metrics.to_csv("/gpfs/ysm/project/kleinstein/mw957/repos/spectral-tree-inference/experiments/results/kingman_angle_" + str(n) + ".csv")

