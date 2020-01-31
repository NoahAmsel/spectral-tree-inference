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
import spectraltree
import utils
import generation
import reconstruct_tree


N = 400
num_taxa = 128
jc = generation.Jukes_Cantor()
mutation_rate = [jc.p2t(0.95)]

tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=1)
for x in tree.preorder_edge_iter():
    x.length = 1
# tree = utils.lopsided_tree(num_taxa=num_taxa) 
#tree = utils.balanced_binary(num_taxa)


observations = generation.simulate_sequences_ordered(N, tree_model=tree, seq_model=jc, mutation_rate=mutation_rate)
S = reconstruct_tree.JC_similarity_matrix(observations)
TT = reconstruct_tree.spectral_tree_reonstruction(S, namespace = tree.taxon_namespace)

RF,F1 = reconstruct_tree.compare_trees(tree, TT)
print("Spectral: ")
print("RF = ",RF)
print("F1% = ",F1)
print("")
