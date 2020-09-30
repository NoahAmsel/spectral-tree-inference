import sys, os

sys.path.append("../spectraltree")

import generation
import reconstruct_tree
import time
import utils
import pandas as pd
import dendropy
import argparse
import numpy as np

def run_method(method, path, threshold = None, verbose = False):
    seqs = dendropy.RnaCharacterMatrix.get(path=path, schema="nexus")
    tree = dendropy.Tree.get(path=path, schema="nexus", taxon_namespace = seqs.taxon_namespace)
    
    leafs_idx = [i.label[0] != " " for i in seqs.taxon_namespace]

    ch_list = list()
    for t in seqs.taxon_namespace:
        ch_list.append([x.symbol for x in seqs[t]])

    observations = np.array(ch_list)
    observations = observations[leafs_idx]
    observations = np.where(observations=='A', 0, observations) 
    observations = np.where(observations=='C', 1, observations) 
    observations = np.where(observations=='G', 2, observations) 
    observations = np.where(observations=='U', 3, observations) 
    observations = np.where(observations=='-', 4, observations) 
    observations = observations.astype('int')

    taxa = np.array(seqs.taxon_namespace._taxa)[leafs_idx]

    taxa_meta = utils.TaxaMetadata(seqs.taxon_namespace, list(taxa), alphabet=dendropy.RNA_STATE_ALPHABET)
    
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run different tree reconstruction methods.', )
    parser.add_argument('method', help='method to run: RaXML, SNJ, NJ, STR+NJ, STR+SNJ, STR+RaXML.')
    parser.add_argument("path", help="Path to the tree file.")
    parser.add_argument('out', help="Path to output data files.")
    parser.add_argument('--threshold', type=int, help='Minimum tree size to run the submethod for STR methods.')
    parser.add_argument("--verbose", help="Whether to print the diagnostic messages.")

    args = parser.parse_args()

    method = args.method
    threshold = args.threshold
    path = args.path
    verbose = args.verbose == "True"

    ms = []
    ts = []
    rts = []
    rfs = []
    f1s = []

    print(method, threshold)
    res = run_method(method, path, threshold = threshold, verbose = verbose)
    ms.append(res[0])
    ts.append(res[1])
    rts.append(res[2])
    rfs.append(res[3])
    f1s.append(res[4])

    perf_metrics = pd.DataFrame({'method': ms, 'threshold': ts, 'runtime': rts, 'RF': rfs, "F1": f1s})
    perf_metrics.to_csv(args.out, index=False)
