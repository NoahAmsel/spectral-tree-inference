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

N = 5000
num_taxa = 8
jc = generation.Jukes_Cantor(num_classes=2)
mutation_rate = [jc.p2t(0.95)]
np.random.seed(0)
# reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=1)
# reference_tree = utils.lopsided_tree(num_taxa)
reference_tree = utils.balanced_binary(num_taxa)
# for x in reference_tree.preorder_edge_iter():
#     x.length = 1
np.random.seed(0)
t0 = time.time()
observations,meta = generation.simulate_sequences(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate,rng=np.random, alphabet="Binary")
print("gen time: ", time.time() - t0)
rg = reconstruct_tree.RG()

t0 = time.time()
# cProfile.run("""tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, reconstruct_tree.JC_similarity_matrix,
#                                     taxa_metadata= meta,
#                                     threshhold = 4 ,verbose=False)""", filename="temp.prof")
# # To view with a nice GUI, run: snakeviz .\temp.prof
tree_rec = rg(observations, taxa_metadata= meta)
t = time.time() - t0


RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)
print("rg: ")
print("time = ", t)

print("RF = ",RF)
print("F1% = ",F1)
print("")



