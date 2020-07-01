# %% imports and funcs
import sys, os
import pickle

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))

import copy
import numpy as np
import utils
import generation
import reconstruct_tree
import dendropy
import scipy
import time
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dendropy.model.discrete import simulate_discrete_chars, Jc69, Hky85
from dendropy.calculate.treecompare import symmetric_difference
from sklearn.decomposition import TruncatedSVD

def check_is_bipartition(tree, parent_split,bool_partition):
    if sum(bool_partition) == 0:
        return False
    if sum(bool_partition) == len(bool_partition):
        return False
    bipartitions = [str(x)[::-1] for x in tree.encode_bipartitions()]

    #ariel add - remove parent list
    idx_parent = [i for i, element in enumerate(parent_split) if element]
    bipartitions = ["".join([p[i] for i in idx_parent]) for p in bipartitions]

    partition_1 = "".join(list(bool_partition.astype('int').astype('str')))
    partition_2 = "".join(list((1 - bool_partition).astype('int').astype('str')))
    is_bipartition = (partition_1 in bipartitions) or (partition_2 in bipartitions)
    return is_bipartition

def check_is_bipartition_subtree(tree, parent_split,child_split):
    if sum(child_split) == 0:
        return False
    if sum(child_split) == len(child_split):
        return False
    bipartitions = [str(x)[::-1] for x in tree.encode_bipartitions()]

    #ariel add - remove parent list
    idx_parent = [i for i, element in enumerate(parent_split) if element]
    bipartitions = ["".join([p[i] for i in idx_parent]) for p in bipartitions]
    child_split = child_split[parent_split]

    partition_1 = "".join(list(child_split.astype('int').astype('str')))
    partition_2 = "".join(list((1 - child_split).astype('int').astype('str')))
    is_bipartition = (partition_1 in bipartitions) or (partition_2 in bipartitions)
    return is_bipartition


def to_bool(partition_str):
    return np.array(list(partition_str)) == '1'

def min_partition_size(bipartition_encoding):
    n_ones = np.sum(np.array(list(bipartition_encoding)) == '1')
    n_zeros = np.sum(np.array(list(bipartition_encoding)) == '0')
    return(min(n_ones, n_zeros))

# def check_mearge_all_bp(tree, data_len, min_bipartition_length = 0):
#     all_bipartitions = np.array([str(x)[::-1] for x in tree.encode_bipartitions()][0:-1])
#     min_bipar = np.array([min_partition_size(x) for x in all_bipartitions])
#     filtered_bipar = all_bipartitions[np.where(min_bipar > min_bipartition_length)[0]]

######################################################################
######################################################################
# %% Mearging

# tree_path = "/home/mw957/project/repos/spec_tree/data/skygrid_J2.newick"
# fasta_path = "/home/mw957/project/repos/spec_tree/data/H3N2_NewYork.fasta"
# tree = utils.lopsided_tree(128)
# # tree = dendropy.Tree.get(path=tree_path, schema="newick")
# all_bipartitions = np.array([str(x)[::-1] for x in tree.encode_bipartitions()][0:-1])

# taxon_namespace_label = np.array([x.label for x in tree.taxon_namespace])



# min_bipar = np.array([min_partition_size(x) for x in all_bipartitions])
# filtered_bipar = all_bipartitions[np.where(min_bipar > 50)[0]]


# N = [50, 100, 400, 600, 800, 1000]

# Ns = []
# par1s = []
# par2s = []
# RFs = []
# F1s = []
# rts = []

# for n in N:
#     print(n)
#     data_HKY = simulate_discrete_chars(n, tree, Hky85(kappa = 2), mutation_rate=0.1)
#     ch_list = list()
#     for t in data_HKY.taxon_namespace: 
#         ch_list.append([x.symbol for x in data_HKY[t]])
#     ch_arr = np.array(ch_list)
#     HKY_sim = reconstruct_tree.HKY_similarity_matrix(ch_arr)
    
#     for partition in filtered_bipar:
#         partition = to_bool(partition)
#         par1_size = np.sum(partition)
#         par2_size = np.sum(np.logical_not(partition))
#         print("Partition size: ", par1_size, " vs ", par2_size)
#         left_namespace = list(taxon_namespace_label[np.where(partition)[0]])
#         left_taxa = dendropy.TaxonNamespace([taxon for taxon in tree.taxon_namespace
#             if taxon.label in left_namespace])

#         T_left = copy.deepcopy(tree).extract_tree_with_taxa_labels(labels = left_namespace)
#         T_left.purge_taxon_namespace()
#         s = T_left.as_string(schema = "newick")
#         T_left = dendropy.Tree.get(data=s, schema="newick", taxon_namespace = left_taxa)
#         right_namespace = list(taxon_namespace_label[np.where(np.logical_not(partition))[0]])
#         right_taxa = dendropy.TaxonNamespace([taxon for taxon in tree.taxon_namespace
#             if taxon.label in right_namespace])
#         T_right = copy.deepcopy(tree).extract_tree_with_taxa_labels(labels = right_namespace)
#         T_right.purge_taxon_namespace()
#         s = T_right.as_string(schema = "newick")
#         T_right = dendropy.Tree.get(data=s,
#         schema="newick", taxon_namespace = right_taxa)
        
#         start_time = time.time()
#         joined_tree = reconstruct_tree.join_trees_with_spectral_root_finding_ls(
#             HKY_sim, T_left, T_right, merge_method="least_square", taxon_namespace = tree.taxon_namespace)
#         runtime = time.time() - start_time
        
#         RF,F1 = reconstruct_tree.compare_trees(joined_tree, tree)
        
#         Ns.append(n)
#         par1s.append(par1_size)
#         par2s.append(par2_size)
#         RFs.append(RF)
#         F1s.append(F1)
#         rts.append(runtime)
        
# perf_metrics = pd.DataFrame({'seqlength': Ns, 'par1_size': par1s, 'par2_size': par2s, 
#                              'RF': RFs, "F1": F1s, "runtime": rts})
# #perf_metrics.to_csv("/gpfs/ysm/project/kleinstein/mw957/repos/spec_tree/script/rooting_metrics_ls_2.csv")

######################################################################
######################################################################
# %% partitioning
#
def check_single_splits(tree,parent_split,child_split):
    # Assume that parent split is indded a partition of tree 
    # get subtree that includes only parent split
    namespace = tree.taxon_namespace
    sub_namespace =  np.array(list(tree.taxon_namespace))[~parent_split]
    taxa_label_list = [i.label for i in sub_namespace]
    subtree = copy.deepcopy(tree)
    subtree.prune_taxa_with_labels(taxa_label_list)
    #child_split_cut = child_split[parent_split]
    return check_is_bipartition_subtree(subtree,parent_split,child_split)
     
def check_all_splits(tree,splits,parent_idx):
    results = []
    for c_idx in np.arange(1,len(splits)):
        child_split = splits[c_idx]
        parent_split = splits[parent_idx[c_idx]]
        results.append(check_single_splits(tree,parent_split,child_split))    
    return results
     
def generate_figure(results,N,threshold_methods):
    #results = pd.read_csv("influenza_split_test.csv")
    #for res in results.iterrows():
        #t_vec = res["split_results"][1:-1].split(',')
        #b_vec = [ t=='True' for t in t_vec]
        #s = any(not b_vec)
    #success_vec = [any(~res['split results']) for idx,res in results.iterrows()]
    #results['success'] = success_vec
    res_summary = pd.DataFrame({'N':[],'method':[],'acc':[]})
    for n in N:
        for method in threshold_methods:
            acc = np.mean(results.loc[(results['method']==method) & (results['N']==n)]['success'])
            res_summary = res_summary.append({'N':n,'method':method,'acc':acc},ignore_index=True)
    sns.lineplot(x="N", y="acc", hue="method", style="method",data=res_summary)

    

    
    #for n in N:
    #    for method in threshold_methods:

    #for res in results:
    #    success_vec.append()
    #sns.set()
    #sns.relplot(x="N", y="tip", col="time",
    #        hue="smoker", style="smoker", size="size",
    #        data=results)

#    for method in threshold_methods:
#        for n in N:

def recursive_partition_taxa(S,threshold,threshold_method):
    #splits = [np.array([True] * S.shape[0])]
    splits = []
    
    # Obtain an split of the taxa by one of four methods
    if (threshold_method == 'similarity_zero'):
        _, eigvec = np.linalg.eigh(S)
        partition = eigvec[:,-2]<0        
    if (threshold_method == 'similarity_gap'):
        _, eigvec = np.linalg.eigh(S)
        partition = reconstruct_tree.partition_taxa(eigvec[:,-2], S, 1,1)
    if (threshold_method == 'laplacian_zero'):        
        lap = np.diag(np.sum(S, axis = 0)) - S
        _, eigvec = np.linalg.eigh(lap)
        partition = eigvec[:,1]<0                
    if (threshold_method == 'laplacian_gap'):        
        lap = np.diag(np.sum(S, axis = 0)) - S
        _, eigvec = np.linalg.eigh(lap)
        partition = reconstruct_tree.partition_taxa(eigvec[:,1], S, 1,1)
    if (threshold_method == 'laplacian_rw_zero'):        
        lap = np.matmul(np.diag(np.sum(S,axis=0)**-1),S)
        _, eigvec = np.linalg.eig(lap)
        partition = eigvec[:,1]<0
    if (threshold_method == 'laplacian_rw_gap'):        
        lap = np.matmul(np.diag(np.sum(S,axis=0)**-1),S)
        _, eigvec = np.linalg.eig(lap)
        partition = reconstruct_tree.partition_taxa(eigvec[:,1], S, 1,1)

    splits.append(partition)
    

    # Check if splitting obtained threshold for positive elements
    p_list = [0]
    if np.sum(partition)>threshold:
        
        # Continue recursive split until obtain threshold
        r_p_list,partition_list = recursive_partition_taxa(S[partition,:][:,partition],threshold,threshold_method)        
        r_p_list = [r_p_idx+1 for r_p_idx in r_p_list]
        p_list = p_list + r_p_list

        # Add False to elements not participated in the splitting action
        for p in partition_list:
            par_p = copy.deepcopy(partition)
            par_p[partition] = p
            splits.append(par_p)            
    
    

    # Check if splitting obtained threshold for negative elements
    splits.append(~partition)    
    p_list = p_list + [0]
    if np.sum(~partition)>threshold:

        # Continue recursive split until obtain threshold
        l_p_list,partition_list = recursive_partition_taxa(S[~partition,:][:,~partition],threshold,threshold_method)
        l_p_list = [l_p_idx+len(p_list) for l_p_idx in l_p_list]
        p_list = p_list + l_p_list

        # Add False to elements not participated in the splitting action
        for p in partition_list:
            par_p = copy.deepcopy(~partition)
            par_p[~partition] = p
            splits.append(par_p)            

    return p_list,splits

#tree = utils.lopsided_tree(32)
tree_path = os.path.join(os.path.dirname(sys.path[0]), "data/NY_H3N2.newick")
#fasta_path = os.path.join(os.path.dirname(sys.path[0]), "data/NY_H3N2.fasta")
H3N2_tree = dendropy.Tree.get(path=tree_path, schema="newick")
#H3N2_dna = dendropy.DnaCharacterMatrix.get(file=open(fasta_path, "r"), schema="fasta")

tree_H1N1_path = os.path.join(os.path.dirname(sys.path[0]), "data/NY_H1N1.newick")
#fasta_H1N1_path = os.path.join(os.path.dirname(sys.path[0]), "data/NY_H1N1.fasta")
H1N1_tree = dendropy.Tree.get(path=tree_H1N1_path, schema="newick")
#H1N1_dna = dendropy.DnaCharacterMatrix.get(file=open(fasta_H1N1_path, "r"), schema="fasta")


caterpillar_tree_512 = utils.lopsided_tree(512)
binary_tree_1024= utils.balanced_binary(1024)
tree_list = [H1N1_tree,H3N2_tree,binary_tree_1024,caterpillar_tree_512]
tree_str_list = ['H1N1','H3N2','binary_1024','caterpillar_512']

B = 20
N_H3N2 = [500,1000,1500,2000]
N_H1N1 = [300,500,700,1000]
N_1024 = [500,1000,1500,2000] 
N_cat_256 =  [300,500,700,1000]
threshold = 128
#N_list = [N_H1N1,N_H3N2]
N = [300,500,700,900,1000]

# test params
#B = 2
#tree_list = [utils.lopsided_tree(32), utils.balanced_binary(32)]
#tree_str_list = ['caterpillar', 'balanced']
#N = [100, 200]
#threshold = 16



threshold_methods = ['similarity_zero','similarity_gap','laplacian_zero','laplacian_gap','laplacian_rw_zero','laplacian_rw_gap']
#generate_figure(N,threshold_methods)
results = pd.DataFrame({'tree':[],'N':[],'method':[],'split size':[],'split results':[],'success':[]})

for (tree,tree_str) in zip(tree_list,tree_str_list):
    for n in N:
        print(n)
    
        start_time = time.time()
        for b in range(B):
            print(b)
            data_HKY = simulate_discrete_chars(n, tree, Hky85(kappa = 2), mutation_rate=0.05)
            ch_list = list()
            for t in data_HKY.taxon_namespace: 
                ch_list.append([x.symbol for x in data_HKY[t]])
            ch_arr = np.array(ch_list)
        
            # Compute similarity matrix
            HKY_sim = reconstruct_tree.HKY_similarity_matrix(ch_arr)
        
            # scan over methods       
            for method in threshold_methods:
                parent_list,splits = recursive_partition_taxa(HKY_sim,threshold,method)
                splits = [np.array([True] * HKY_sim.shape[0])] + splits
                parent_list = [-1]+parent_list
                split_results = check_all_splits(tree, splits, parent_list)
                success = all(split_results)            
                split_size = [np.sum(spl) for spl in splits]            
                results = results.append({'tree':tree_str,'N':n,'method':method,'split size':split_size,\
                    'success':success,'split results':split_results},ignore_index=True)
            output = open('splitting_multiple_trees.pkl', 'wb')
            pickle.dump(results, output)
            output.close()

#results.to_csv('influenza_h1n1_split_test.csv',index=False)
print(results)
print('')
# # %%
