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
from character_matrix import FastCharacterMatrix

from dendropy.model.discrete import simulate_discrete_chars, Jc69, Hky85
from dendropy.calculate.treecompare import symmetric_difference

N = 5000
num_taxa = 32

#################################
## Tree generation
#################################
print("Creating tree")
jc = generation.Jukes_Cantor()
# hky = generation.HKY(kappa = 2)
#mutation_rate = [jc.p2t(0.95)]
mutation_rate = [0.1]

#reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=0.5)
reference_tree = utils.lopsided_tree(num_taxa)
# reference_tree = utils.balanced_binary(num_taxa)
# for x in reference_tree.preorder_edge_iter():
#     x.length = 0.5
print("Genration observations by JC and HKY")
#observationsJC = FastCharacterMatrix(generation.simulate_sequences_ordered(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate))
observationsJC = FastCharacterMatrix(simulate_discrete_chars(N, reference_tree, Jc69(), mutation_rate=mutation_rate[0]))
observationsHKY = FastCharacterMatrix(simulate_discrete_chars(N, reference_tree, Hky85(kappa = 1), mutation_rate=mutation_rate[0]))

#################################
## SNJ - Jukes_Cantor
#################################
t0 = _t()
snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)   
tree_rec = snj(observationsJC,reference_tree.taxon_namespace)
RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)
print("###################")
print("SNJ - Jukes_Cantor:")
print("time:", _t() - t0)
print("RF = ",RF, "    F1% = ",F1)
print("")

#################################
## NJ - Jukes_Cantor
#################################
t0 = _t()
nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)   
tree_rec = nj(observationsJC,reference_tree.taxon_namespace)
RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)

print("###################")
print("NJ - Jukes_Cantor:")
print("time:", _t() - t0)
print("RF = ",RF, "    F1% = ",F1)
print("")



#################################
## SNJ - HKY
#################################
t0 = _t()
snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.HKY_similarity_matrix)   
tree_rec = snj(observationsHKY,reference_tree.taxon_namespace)
RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)
print("###################")
print("SNJ - HKY:")
print("time:", _t() - t0)
print("RF = ",RF, "    F1% = ",F1)
print("")

#################################
## NJ - HKY
#################################
t0 = _t()
nj = reconstruct_tree.NeighborJoining(reconstruct_tree.HKY_similarity_matrix)   
tree_rec = nj(observationsHKY,reference_tree.taxon_namespace)
RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)
print("###################")
print("NJ - HKY:")
print("time:", _t() - t0)
print("RF = ",RF, "    F1% = ",F1)
print("")
