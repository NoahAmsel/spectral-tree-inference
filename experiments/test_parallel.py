import sys, os, platform
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))
#print(sys.path[0])

import reconstruct_tree
import utils
import time
import generation
import compare_methods
from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
from dendropy.simulate.treesim import birth_death_tree, pure_kingman_tree, mean_kingman_tree 
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp

import time 
  
num_taxa = 100
reference_tree = utils.unrooted_pure_kingman_tree(num_taxa)
jc = generation.Jukes_Cantor(num_classes=4)
mutation_rate = jc.p2t(0.9)
#N_vec = [800,1000,1200,1400]
n = 1000
num_reps = 5

spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.RAxML,reconstruct_tree.JC_similarity_matrix)
observations, taxa_meta = generation.simulate_sequences(n, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")

f = open("delete_me.pkl",'wb')

bipartitions1 = reference_tree.bipartition_edge_map.keys()
pkl.dump(list(bipartitions1)[0],f)



tree_spectral = spectral_method.deep_spectral_tree_reconstruction(observations, reconstruct_tree.JC_similarity_matrix,
                                         taxa_metadata= taxa_meta,
                                        threshhold = 20 ,min_split = 5, merge_method = "least_square", verbose=False)




def square(x): 
    return x * x 
   
if __name__ == '__main__': 
    #pool = multiprocessing.Pool() 
    pool = multiprocessing.Pool(processes=4) 
    inputs = [0,1,2,3,4] 
    outputs = pool.map(square, inputs) 
    print("Input: {}".format(inputs)) 
    print("Output: {}".format(outputs)) 