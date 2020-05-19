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

N = 500
num_taxa = 64
jc = generation.Jukes_Cantor()
mutation_rate = [jc.p2t(0.95)]
num_itr = 20
# reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=1)
# for x in reference_tree.preorder_edge_iter():
#     x.length = 1
merging_method_list = ['least_square','angle']
RF = {'least_square': [], 'angle': []}
F1 = {'least_square': [], 'angle': []}
for merge_method in merging_method_list:
    for i in range(num_itr):
        #reference_tree = utils.balanced_binary(num_taxa)
        reference_tree = utils.lopsided_tree(num_taxa)
        observations = generation.simulate_sequences_ordered(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate)
        spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.NeighborJoining,reconstruct_tree.JC_similarity_matrix)   
        tree_rec = spectral_method.deep_spectral_tree_reonstruction(observations, reconstruct_tree.JC_similarity_matrix, 
            taxon_namespace = reference_tree.taxon_namespace, threshhold = 16,merge_method = merge_method)
        RF_i,F1_i = reconstruct_tree.compare_trees(tree_rec, reference_tree)
        RF[merge_method].append(RF_i)
        F1[merge_method].append(F1_i)
        


print("Angle RF: ",np.mean(RF['angle']))
print("LS: ",np.mean(RF['least_square']))
print("")


