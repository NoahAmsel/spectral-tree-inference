##########
# This file has been replaced by test_spectral_tree_reconstruction.py
##########

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
import str
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
import cProfile

N = 300
num_taxa = 512
threshold = 32
# jc = generation.Jukes_Cantor()
hky = generation.HKY(kappa = 2)
mutation_rate = 0.05
# mutation_rate = [jc.p2t(0.95)]
np.random.seed(0)
# reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=1)
# reference_tree = utils.lopsided_tree(num_taxa)
reference_tree = utils.balanced_binary(num_taxa)
# for x in reference_tree.preorder_edge_iter():
#     x.length = 1
np.random.seed(0)
t0 = time.time()
observations,meta = generation.simulate_sequences(N, tree_model=reference_tree, seq_model=hky, mutation_rate=mutation_rate,rng=np.random, alphabet = 'DNA')
print("gen time: ", time.time() - t0)
spectral_method = str.STR(reconstruct_tree.RAxML, reconstruct_tree.HKY_similarity_matrix, threshold = threshold, merge_method="least_square", num_gaps = 1, min_split = 5, verbose=False)

t0 = time.time()
#cProfile.run("""tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, reconstruct_tree.JC_similarity_matrix,
#                                    taxa_metadata= meta,
#                                    threshhold = 8, min_split = 3 ,verbose=False)""", filename="temp1.prof")
# # To view with a nice GUI, run: snakeviz .\temp.prof
tree_rec = spectral_method(observations,  taxa_metadata= meta)
t = time.time() - t0


RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)
print("Spectral: ")
print("time = ", t)

print("RF = ",RF)
print("F1% = ",F1)
print("")



