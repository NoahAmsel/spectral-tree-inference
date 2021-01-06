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
from time import time as _t

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))

#import spectraltree
import utils
import generation
import reconstruct_tree
import metricNearness

from dendropy.model.discrete import simulate_discrete_chars, Jc69, Hky85
from dendropy.calculate.treecompare import symmetric_difference

N = 100
num_taxa = 60

#################################
## Tree generation
#################################
print("Creating tree")
jc = generation.Jukes_Cantor()
hky = generation.HKY(kappa = 2)
mutation_rate = [jc.p2t(0.95)]
# mutation_rate = [0.1]

#reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=0.5)
reference_tree = utils.lopsided_tree(num_taxa)
# reference_tree = utils.balanced_binary(num_taxa)
# for x in reference_tree.preorder_edge_iter():
#     x.length = 0.5
print("Genration observations by JC and HKY")

observationsJC, metaJC = generation.simulate_sequences(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")

#################################
## SNJ - Jukes_Cantor
#################################
t0 = _t()
snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)   
sim = reconstruct_tree.JC_similarity_matrix(observationsJC, metaJC)
inside_log = np.clip(sim, a_min=1e-16, a_max=None)
dis = - np.log(inside_log)
disC = np.array(metricNearness.metricNearness(dis))
simC = np.exp(-disC)
tree_rec = snj.reconstruct_from_similarity(sim, taxa_metadata = metaJC)
tree_recC = snj.reconstruct_from_similarity(simC, taxa_metadata = metaJC)
RFC,F1C = reconstruct_tree.compare_trees(tree_recC, reference_tree)
RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)

print("###################")
print("SNJ - Jukes_Cantor:")
print("time:", _t() - t0)
print("RF = ",RF, "    F1% = ",F1)
print("RFC = ",RFC, "    F1C% = ",F1C)
print("")

