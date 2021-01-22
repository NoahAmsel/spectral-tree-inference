import datetime
import pickle
import os.path
import time
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import dendropy
import copy

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))

#import spectraltree
import utils
import generation
import reconstruct_tree
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
import cProfile

if __name__ == '__main__':
    N = 500
    num_taxa = 128
    threshold = 16
    jc = generation.Jukes_Cantor()
    hky = generation.HKY(kappa = 2)
    mutation_rate = 0.05
    #mutation_rate = [jc.p2t(0.95)]
    np.random.seed(0)
    # reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=1)
    # reference_tree = utils.lopsided_tree(num_taxa)
    reference_tree = utils.balanced_binary(num_taxa)
    # reference_tree = utils.unrooted_pure_kingman_tree(num_taxa)
    # for x in reference_tree.preorder_edge_iter():
    #     x.length = 1
    np.random.seed(0)
    t0 = time.time()
    observations,meta = generation.simulate_sequences(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate,rng=np.random, alphabet = 'DNA')
    print("gen time: ", time.time() - t0)
    spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.RAxML,reconstruct_tree.JC_similarity_matrix)   
    #spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.NeighborJoining,reconstruct_tree.JC_similarity_matrix)   

    t0 = time.time()
    # cProfile.run("""tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, reconstruct_tree.JC_similarity_matrix,
    #                                    taxa_metadata= meta,
    #                                    threshhold = threshold, min_split = 5 ,verbose=False)""", filename="temp1.prof")
    # # To view with a nice GUI, run: snakeviz .\temp1.prof
    tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, reconstruct_tree.JC_similarity_matrix,
                                        taxa_metadata= meta,
                                        threshhold = threshold ,min_split = 5, merge_method = "least_square", verbose=False)
    t = time.time() - t0


    RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)
    print("Spectral: ")
    print("time = ", t)

    print("RF = ",RF)
    print("F1% = ",F1)
    print("")



