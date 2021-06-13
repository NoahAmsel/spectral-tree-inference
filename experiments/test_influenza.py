import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))
import numpy as np
import utils
import generation
import reconstruct_tree
import dendropy
import scipy
import time
from itertools import product
import matplotlib.pyplot as plt
import cProfile

from dendropy.model.discrete import simulate_discrete_chars, Jc69, Hky85
from dendropy.calculate.treecompare import symmetric_difference

tree_path = os.path.join(os.path.dirname(sys.path[0]), "data/NY_H3N2.newick")
fasta_path = os.path.join(os.path.dirname(sys.path[0]), "data/NY_H3N2.fasta")
H3N2_tree = dendropy.Tree.get(path=tree_path, schema="newick")
H3N2_dna = dendropy.DnaCharacterMatrix.get(file=open(fasta_path, "r"), schema="fasta")

N = 1000 
data_HKY = simulate_discrete_chars(N, H3N2_tree, Hky85(kappa = 2), mutation_rate=0.1)
ch_list = list()
for t in data_HKY.taxon_namespace:
    ch_list.append([x.symbol for x in data_HKY[t]])
ch_arr = np.array(ch_list)
identical = np.array([np.mean(a == b) for a, b in product(ch_arr, repeat = 2)])

#start_time = time.time()
#cProfile.run('S = HKY_similarity_matrix(ch_arr)')
#compute_s_time = time.time() - start_time
#print("--- %s seconds ---" % compute_s_time)
threshold = 128
t1 = time.time()
spectral_method = reconstruct_tree.STDR(reconstruct_tree.RAxML,
                                                              reconstruct_tree.HKY_similarity_matrix)
tree_rec = spectral_method.deep_spectral_tree_reconstruction(ch_arr, reconstruct_tree.HKY_similarity_matrix, 
                                                            taxon_namespace = H3N2_tree.taxon_namespace, 
                                                            threshhold = threshold,min_split = 30)
runtime = time.time()-t1


Deep_nj_RF, Deep_nj_RF_F1 = reconstruct_tree.compare_trees(tree_rec, H3N2_tree)
print("SNJ: ")
print("RF = ", Deep_nj_RF)
print("F1% = ", Deep_nj_RF_F1)
print("runtime = ", runtime)
print("")



print(tree_path)