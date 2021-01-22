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

from dendropy.model.discrete import simulate_discrete_chars, Jc69, Hky85
from dendropy.calculate.treecompare import symmetric_difference

N = 500
num_taxa = 8

#################################
## Tree generation
#################################
print("Creating tree")
jc = generation.Jukes_Cantor()
hky = generation.HKY(kappa = 2)
#mutation_rate = [jc.p2t(0.95)]
mutation_rate = [0.1]

reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=0.5)
#reference_tree = utils.lopsided_tree(num_taxa)
#reference_tree = utils.balanced_binary(num_taxa)
# for x in reference_tree.preorder_edge_iter():
#     x.length = 0.5
print("Genration observations by JC and HKY")

observationsJC, metaJC = generation.simulate_sequences(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")
observationsHKY, metaHKY = generation.simulate_sequences(N, tree_model=reference_tree, seq_model=hky, mutation_rate=mutation_rate, alphabet="DNA")

t0 = _t()
forrest = reconstruct_tree.Forrest()
tree_rec = forrest(observationsJC, metaJC)
t1 = _t()
RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)
print("###################")
print("Forrest - Jukes_Cantor:")
print("time:", t1 - t0)
print("RF = ",RF, "    F1% = ",F1)
print("")