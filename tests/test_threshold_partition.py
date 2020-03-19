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

N = 100
num_taxa = 256
jc = generation.Jukes_Cantor()
mutation_rate = [jc.p2t(0.95)]

# reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=1)
# for x in reference_tree.preorder_edge_iter():
#     x.length = 1

reference_tree = utils.balanced_binary(num_taxa)
observations = generation.simulate_sequences_ordered(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate)
spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.NeighborJoining,reconstruct_tree.JC_similarity_matrix)   
tree_rec = spectral_method.deep_spectral_tree_reonstruction(observations, reconstruct_tree.JC_similarity_matrix, taxon_namespace = reference_tree.taxon_namespace, num_gaps = 4,threshhold = 35)
tree_rec_b = spectral_method.deep_spectral_tree_reonstruction(observations, reconstruct_tree.JC_similarity_matrix, taxon_namespace = reference_tree.taxon_namespace, num_gaps = 1,threshhold = 35)

RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)
RF_b,F1_b = reconstruct_tree.compare_trees(tree_rec_b, reference_tree)
print("Spectral deep: ")
print("RF multi gaps= ",RF)
print("RF standard = ",RF_b)
#print("F1% = ",F1)
print("")


