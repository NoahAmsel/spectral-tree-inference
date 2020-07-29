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
import pandas as pd
from itertools import combinations 
from itertools import product
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD 
from sklearn.utils.extmath import randomized_svd

def cherry_count_for_tree(tree):
    cherry_count = 0
    for node in tree.leaf_nodes():
        if node.parent_node.child_nodes()[0].is_leaf() and node.parent_node.child_nodes()[1].is_leaf():
            cherry_count += 1
    cherry_count = cherry_count/2
    return cherry_count

num_taxa = 20
num_reps = 10


N_vec = np.arange(100,400,50)
#reference_tree = utils.balanced_binary(num_taxa)
jc = generation.Jukes_Cantor(num_classes=2)

mutation_rate = jc.p2t(0.9)


snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix) 
nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix) 
treesvd = reconstruct_tree.TreeSVD()
methods = [snj,treesvd]
#results = compare_methods.experiment([reference_tree], jc, N_vec, methods=methods,\
#     mutation_rates = [mutation_rate], reps_per_tree=num_reps)
df = pd.DataFrame(columns=['method', 'runtime', 'RF','n','cherries_ref','cherries_res'])
for i in np.arange(num_reps):
    print(i)
    reference_tree = utils.unrooted_pure_kingman_tree(num_taxa)
    ch_ref = cherry_count_for_tree(reference_tree)
    for n in N_vec:
        observations, taxa_meta = generation.simulate_sequences(n, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")

        # Tree svd
        t_s = time.time()    
        tree_svd_rec = treesvd(observations,taxa_meta)   
        runtime_treesvd = time.time()-t_s
        RF_svd,F1 = reconstruct_tree.compare_trees(tree_svd_rec, reference_tree)
        ch_svd = cherry_count_for_tree(tree_svd_rec)
        df = df.append({'method': 'treesvd', 'runtime': runtime_treesvd, 'RF': RF_svd,'n': n,'cherries_ref': ch_ref,'cherries_res':ch_svd}, ignore_index=True)

        # SNJ        
        t_s = time.time()    
        tree_snj = snj(observations, taxa_meta)
        runtime_snj = time.time()-t_s
        RF_snj,F1 = reconstruct_tree.compare_trees(tree_snj, reference_tree)
        ch_snj = cherry_count_for_tree(tree_snj)
        df = df.append({'method': 'snj', 'runtime': runtime_snj, 'RF': RF_snj,'n': n,'cherries_ref': ch_ref,'cherries_res':ch_snj}, ignore_index=True)
        
        #NJ
        t_s = time.time()    
        tree_nj = nj(observations, taxa_meta)
        runtime_nj = time.time()-t_s
        RF_nj,F1 = reconstruct_tree.compare_trees(tree_nj, reference_tree)
        ch_nj = cherry_count_for_tree(tree_nj)
        df = df.append({'method': 'nj', 'runtime': runtime_nj, 'RF': RF_nj,'n': n,'cherries_ref': ch_ref,'cherries_res':ch_nj}, ignore_index=True)



a = 1        

 
        
        #print('SNJ:  RF, ',RF,' runtime ',runtime_snj)
