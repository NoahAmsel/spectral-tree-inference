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

N = 100000
num_taxa = 16

#################################
## Tree generation
#################################
print("Creating tree")
jc = generation.Jukes_Cantor()
hky = generation.HKY(kappa = 1)
#mutation_rate = [jc.p2t(0.95)]
mutation_rate = [0.1]

#reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=0.5)
# reference_tree = utils.lopsided_tree(num_taxa)
reference_tree = utils.balanced_binary(num_taxa)
for x in reference_tree.preorder_edge_iter():
    x.length = 0.5

print("Genration observations by JC and HKY")
#observationsJC = FastCharacterMatrix(generation.simulate_sequences_ordered(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate))
# observationsHKY = FastCharacterMatrix(generation.simulate_sequences_ordered(N, tree_model=reference_tree, seq_model=hky, mutation_rate=mutation_rate))
t = time.time()
observationsJC = FastCharacterMatrix(simulate_discrete_chars(N, reference_tree, Jc69(), mutation_rate=mutation_rate[0]))
print("Time to generate JC:", time.time()-t)
t = time.time()
observationsHKY = FastCharacterMatrix(simulate_discrete_chars(N, reference_tree, Hky85(kappa = 1), mutation_rate=mutation_rate[0]))
print("Time to generate HKY:", time.time()-t)

t = time.time()
S_JC = reconstruct_tree.JC_similarity_matrix(observationsJC.to_array())
print()
print("Time to compute JC similarity:", time.time()-t)
print("JC similarity")
print(S_JC)

t = time.time()
S_HKY = reconstruct_tree.HKY_similarity_matrix(observationsHKY.to_array())
print()
print("Time to compute HKY similarity:", time.time()-t)
print("HKY similarity")
print(S_HKY)