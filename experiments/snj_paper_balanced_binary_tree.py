import sys, os, platform
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))
import reconstruct_tree
import utils
import time
import generation
import compare_methods
from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
from dendropy.simulate.treesim import birth_death_tree, pure_kingman_tree, mean_kingman_tree 
import igraph
import numpy as np
from itertools import combinations 
from itertools import product
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD 
from sklearn.utils.extmath import randomized_svd
import oct2py
from oct2py import octave
import scipy

tree_list = [utils.balanced_binary(32)]
jc = generation.Jukes_Cantor(num_classes=2)
Ns = [100,200,300]
mutation_rates = [jc.p2t(0.9)]
snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix) 
nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix) 
RG = reconstruct_tree.RG()
CLRG = reconstruct_tree.CLRG()
methods = [nj,snj,RG,CLRG]
num_reps = 1
results = compare_methods.experiment(tree_list = tree_list, 
sequence_model = jc, Ns = Ns, methods=methods, mutation_rates = mutation_rates, 
reps_per_tree=num_reps,savepath='balanced_binary_m_512.pkl',folder = './data/',overwrite=True)
df = compare_methods.results2frame(results)
print(df)
a =1