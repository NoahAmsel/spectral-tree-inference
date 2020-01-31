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

# def join_trees_with_spectral_root_finding(similarity_matrix, T1, T2, namespace=None):
#     # sub_idx  - internal variable. Do not change when calling the function
#     m, m2 = similarity_matrix.shape
#     sub_idx = list(range(m))
#     #print("size of data:", m, )
#     assert m == m2, "Distance matrix must be square"
#     if namespace is None:
#         namespace = utils.default_namespace(m)
#     else:
#         assert len(namespace) >= m, "Namespace too small for distance matrix"
    
#     T = dendropy.Tree(taxon_namespace=namespace)
    
#     half1_idx = list(range(len(T1.leaf_nodes())))
#     half1_idx_array = np.array(half1_idx)
#     T1.is_rooted = True

#     half2_idx = list(range(len(T1.leaf_nodes()), len(namespace)))
#     half2_idx_array = np.array(half2_idx)
#     T2.is_rooted = True
    
#     # finding roots
    
#     # find root of half 1
#     T1.update_bipartitions()
#     if len(half1_idx)>2:
#         bipartitions1 = T1.bipartition_edge_map
#         min_ev2 = float("inf")
#         for bp in bipartitions1.keys():
#             if bp.leafset_as_bitstring().find('0') == -1:
#                 continue
#             # Slice similarity matrix of half2 + part of half1 VS the other part of half 1
#             bool_array =np.array(list(map(bool,[int(i) for i in bp.leafset_as_bitstring()]))[::-1])
#             h1_idx_A = half1_idx_array[bool_array]
#             h1_idx_B = half1_idx_array[~bool_array]
#             other_idx = np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])), sub_idx)
        
#             A_h2_other_idx = list(np.concatenate([h1_idx_A, half2_idx_array, other_idx]))
#             B_h2_other_idx = list(np.concatenate([h1_idx_B, half2_idx_array, other_idx]))

#             A_other_idx = list(np.concatenate([h1_idx_A, other_idx]))
#             B_other_idx = list(np.concatenate([h1_idx_B, other_idx]))
            
#             A_h2_idx = list(np.concatenate([h1_idx_A, half2_idx_array]))
#             B_h2_idx = list(np.concatenate([h1_idx_B, half2_idx_array]))

#             #all_minus_h1_idx_cur = list(np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])),h1_idx_cur))
#             #all_minus_h1_idx_cur_complement = list(np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])),h1_idx_cur_complement))
            
#             ####################### example of indexes ##################
#             # FFFF FFFF TTTT TTTT - idx            - these are the element we are "solving for" at theis step
#             # FFFF FFFF 1111 FFFF - half1_idx      - these are the elements assignd to half 1 (and described by T1)
#             # FFFF FFFF FFFF 1111 - half2_idx      - these are the elements assignd to half 1 (and described by T1)
#             # FFFF FFFF 0110 FFFF - h1_idx_A       - these are the elements chosen at the current partition (bp)
#             # FFFF FFFF 1001 FFFF - h1_idx_B       - these are the elements chosen at the current partition (bp)
#             # 1111 1111 FFFF FFFF - other_idx      - 
            
#             # 1111 1111 1001 1111 - B_h2_other_idx - 
#             # 1111 1111 0110 1111 - A_h2_other_idx - 
#             # 1111 1111 0110 0000 - A_other_idx    - 
#             # 1111 1111 1001 0000 - B_other_idx    - 
#             # 0000 0000 0110 1111 - A_h2_idx       - 
#             # 0000 0000 1001 1111 - B_h2_idx       - 
            
#             ## Case 1: Other is connected to A
#             sliced_sim_mat_try1_1 = similarity_matrix[A_other_idx, ...]
#             sliced_sim_mat_try1_1 = sliced_sim_mat_try1_1[...,B_h2_idx]
#             score_try1_1 = 0 if (sliced_sim_mat_try1_1.shape[0] == 1) | (sliced_sim_mat_try1_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try1_1, compute_uv = False)[1]

#             sliced_sim_mat_try1_2 = similarity_matrix[A_h2_other_idx, ...]
#             sliced_sim_mat_try1_2 = sliced_sim_mat_try1_2[...,h1_idx_B]
#             score_try1_2 = 0 if (sliced_sim_mat_try1_2.shape[0] == 1) | (sliced_sim_mat_try1_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try1_2, compute_uv = False)[1]

#             score_try1 = np.max([score_try1_1,score_try1_2])
#             ## Case 2: Other is connected to B
#             sliced_sim_mat_try2_1 = similarity_matrix[h1_idx_A, ...]
#             sliced_sim_mat_try2_1 = sliced_sim_mat_try2_1[...,B_h2_other_idx]
#             score_try2_1 = 0 if (sliced_sim_mat_try2_1.shape[0] == 1) | (sliced_sim_mat_try2_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try2_1, compute_uv = False)[1]
            
#             sliced_sim_mat_try2_2 = similarity_matrix[A_h2_idx, ...]
#             sliced_sim_mat_try2_2 = sliced_sim_mat_try2_2[...,B_other_idx]
#             score_try2_2 = 0 if (sliced_sim_mat_try2_2.shape[0] == 1) | (sliced_sim_mat_try2_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try2_2, compute_uv = False)[1]
            
#             score_try2 = np.max([score_try2_1,score_try2_2])
            
#             ## Case 3: Other is connected to h2
#             sliced_sim_mat_try3_1 = similarity_matrix[h1_idx_A, ...]
#             sliced_sim_mat_try3_1 = sliced_sim_mat_try3_1[...,B_h2_other_idx]
#             score_try3_1 = 0 if (sliced_sim_mat_try3_1.shape[0] == 1) | (sliced_sim_mat_try3_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try3_1, compute_uv = False)[1]
            
#             sliced_sim_mat_try3_2 = similarity_matrix[A_h2_other_idx, ...]
#             sliced_sim_mat_try3_2 = sliced_sim_mat_try3_2[...,h1_idx_B]
#             score_try3_2 = 0 if (sliced_sim_mat_try3_2.shape[0] == 1) | (sliced_sim_mat_try3_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try3_2, compute_uv = False)[1]
            
#             score_try3 = np.max([score_try3_1,score_try3_2])
            
#             if np.min([score_try1, score_try2, score_try3]) <min_ev2:
#                 min_ev2 = np.min([score_try1, score_try2, score_try3])
#                 bp_min = bp
#         # if sum([int(i) for i in bp_min.leafset_as_bitstring()]) != len([int(i) for i in bp_min.leafset_as_bitstring()])/2:
#         #     print("ERROR: ")
#         #     print("size of data:", m)
#         #     print("Half sizes:", sum([int(i) for i in bp_min.leafset_as_bitstring()]), sum([-1*int(i)+1 for i in bp_min.leafset_as_bitstring()]))

#         T1.reroot_at_edge(bipartitions1[bp_min])
    
#     # find root of half 2
#     if len(half2_idx)>2:
#         bipartitions2 = T2.bipartition_edge_map
#         min_ev2 = float("inf")
#         for bp in bipartitions2.keys():
#             if bp.leafset_as_bitstring().find('0') == -1:
#                 continue
#             # Slice similarity matrix of half2 + part of half1 VS the other part of half 1
            
#             bool_array = np.array(list(map(bool,[int(i) for i in bp.leafset_as_bitstring()]))[::-1])
#             h2_idx_A = half2_idx_array[bool_array]
#             h2_idx_B = half2_idx_array[~bool_array]
#             other_idx = np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])), sub_idx)

#             A_h1_other_idx = list(np.concatenate([h2_idx_A, half1_idx_array, other_idx]))
#             B_h1_other_idx = list(np.concatenate([h2_idx_B, half1_idx_array, other_idx]))

#             A_other_idx = list(np.concatenate([h2_idx_A, other_idx]))
#             B_other_idx = list(np.concatenate([h2_idx_B, other_idx]))

#             A_h1_idx = list(np.concatenate([h2_idx_A, half1_idx_array]))
#             B_h1_idx = list(np.concatenate([h2_idx_B, half1_idx_array]))

#             ####################### example of indexes ##################
#             # FFFF FFFF TTTT TTTT - idx            - these are the element we are "solving for" at theis step
#             # FFFF FFFF 1111 FFFF - half1_idx      - these are the elements assignd to half 1 (and described by T1)
#             # FFFF FFFF FFFF 1111 - half2_idx      - these are the elements assignd to half 1 (and described by T1)
#             # FFFF FFFF FFFF 1100 - h1_idx_A       - these are the elements chosen at the current partition (bp)
#             # FFFF FFFF FFFF 0011 - h1_idx_B       - these are the elements chosen at the current partition (bp)
#             # 1111 1111 FFFF FFFF - other_idx      - 
            
#             # 1111 1111 1111 0011 - B_h1_other_idx - 
#             # 1111 1111 1111 1100 - A_h1_other_idx - 
#             # 1111 1111 0000 1100 - A_other_idx    - 
#             # 1111 1111 0000 0011 - B_other_idx    - 
#             # 0000 0000 1111 1100 - A_h1_idx       - 
#             # 0000 0000 1111 0011 - B_h1_idx       - 
            
#             ## Case 1: Other is connected to A
#             sliced_sim_mat_try1_1 = similarity_matrix[A_other_idx, ...]
#             sliced_sim_mat_try1_1 = sliced_sim_mat_try1_1[...,B_h1_idx]
#             score_try1_1 = 0 if (sliced_sim_mat_try1_1.shape[0] == 1) | (sliced_sim_mat_try1_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try1_1, compute_uv = False)[1]

#             sliced_sim_mat_try1_2 = similarity_matrix[A_h1_other_idx, ...]
#             sliced_sim_mat_try1_2 = sliced_sim_mat_try1_2[...,h2_idx_B]
#             score_try1_2 = 0 if (sliced_sim_mat_try1_2.shape[0] == 1) | (sliced_sim_mat_try1_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try1_2, compute_uv = False)[1]

#             score_try1 = np.max([score_try1_1,score_try1_2])
#             ## Case 2: Other is connected to B
#             sliced_sim_mat_try2_1 = similarity_matrix[h2_idx_A, ...]
#             sliced_sim_mat_try2_1 = sliced_sim_mat_try2_1[...,B_h1_other_idx]
#             score_try2_1 = 0 if (sliced_sim_mat_try2_1.shape[0] == 1) | (sliced_sim_mat_try2_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try2_1, compute_uv = False)[1]
            
#             sliced_sim_mat_try2_2 = similarity_matrix[A_h1_idx, ...]
#             sliced_sim_mat_try2_2 = sliced_sim_mat_try2_2[...,B_other_idx]
#             score_try2_2 = 0 if (sliced_sim_mat_try2_2.shape[0] == 1) | (sliced_sim_mat_try2_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try2_2, compute_uv = False)[1]
            
#             score_try2 = np.max([score_try2_1,score_try2_2])
            
#             ## Case 3: Other is connected to h2
#             sliced_sim_mat_try3_1 = similarity_matrix[h2_idx_A, ...]
#             sliced_sim_mat_try3_1 = sliced_sim_mat_try3_1[...,B_h1_other_idx]
#             score_try3_1 = 0 if (sliced_sim_mat_try3_1.shape[0] == 1) | (sliced_sim_mat_try3_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try3_1, compute_uv = False)[1]
            
#             sliced_sim_mat_try3_2 = similarity_matrix[A_h1_other_idx, ...]
#             sliced_sim_mat_try3_2 = sliced_sim_mat_try3_2[...,h2_idx_B]
#             score_try3_2 = 0 if (sliced_sim_mat_try3_2.shape[0] == 1) | (sliced_sim_mat_try3_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try3_2, compute_uv = False)[1]
            
#             score_try3 = np.max([score_try3_1,score_try3_2])
            
#             if np.min([score_try1, score_try2, score_try3]) <min_ev2:
#                 min_ev2 = np.min([score_try1, score_try2, score_try3])
#                 bp_min = bp
#         # if sum([int(i) for i in bp_min.leafset_as_bitstring()]) != len([int(i) for i in bp_min.leafset_as_bitstring()])/2:
#         #     print("ERROR: ")
#         #     print("size of data:", m)
#         #     print("Half sizes:", sum([int(i) for i in bp_min.leafset_as_bitstring()]), sum([-1*int(i)+1 for i in bp_min.leafset_as_bitstring()]))
#         T2.reroot_at_edge(bipartitions2[bp_min])
        
#     T.seed_node.set_child_nodes([T1.seed_node,T2.seed_node])
#     #T.seed_node.add_child(T1.seed_node)
#     #T.seed_node.add_child(T2.seed_node)
#     if len(sub_idx) != similarity_matrix.shape[0]:
#         T.is_rooted = True
#     return T



N = 1000
num_taxa = 256
jc = generation.Jukes_Cantor()
mutation_rate = [jc.p2t(0.95)]
# Construct data of type 'tree' from class dendropy
namespace =  utils.default_namespace(num_taxa)
tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=1)
for x in tree.preorder_edge_iter():
    x.length = 1
#tree = utils.lopsided_tree(num_taxa=num_taxa, namespace=namespace) 
#tree = utils.balanced_binary(num_taxa, namespace=namespace)
tree.is_rooted = True

for i in tree.bipartition_edge_map:
    if tree.bipartition_edge_map[i] == tree.seed_node.child_edges()[0]:
        taxa_half1 =  set(i.leafset_taxa(tree.taxon_namespace) )
    if tree.bipartition_edge_map[i] == tree.seed_node.child_edges()[1]:
        taxa_half2 =  set(i.leafset_taxa(tree.taxon_namespace) )
tree.is_rooted = False        

#taxa_half1 = set([taxon for taxon in tree.taxon_namespace if int(taxon.label)<num_taxa/2])
#taxa_half2 = set([taxon for taxon in tree.taxon_namespace if int(taxon.label)>=num_taxa/2])
tree1 = tree.extract_tree_with_taxa(taxa=taxa_half1)
tree2 = tree.extract_tree_with_taxa(taxa=taxa_half2)

namespace1 = [l.taxon for l in tree1.leaf_nodes()]
namespace1.sort(key = lambda x: int(x.label))
namespace1 = dendropy.TaxonNamespace(namespace1)
namespace2 = [l.taxon for l in tree2.leaf_nodes()]
namespace2.sort(key = lambda x: int(x.label))
namespace2 = dendropy.TaxonNamespace(namespace2)

tree1.taxon_namespace = namespace1
tree2.taxon_namespace = namespace2

# namespace1 = namespace
# T1 = utils.lopsided_tree(num_taxa=int(num_taxa/2), namespace=dendropy.TaxonNamespace(namespace[:int(num_taxa/2)])) 
# T2 = utils.lopsided_tree(num_taxa=int(num_taxa/2), namespace=dendropy.TaxonNamespace(namespace[int(num_taxa/2):])) 
# T1.print_plot
# TT1 = T1.extract_tree_with_taxa(namespace)
# TT2 = T2.extract_tree_with_taxa(namespace)
# tree.seed_node.set_child_nodes([TT1.seed_node,TT2.seed_node])
# create sequences for each node, not sure what is th

observations = generation.simulate_sequences_ordered(N, tree_model=tree, seq_model=jc, mutation_rate=mutation_rate)

t1= time.time()
S = reconstruct_tree.JC_similarity_matrix(observations)
t2 = time.time()
TT = reconstruct_tree.join_trees_with_spectral_root_finding(S,tree1, tree2, namespace = tree.taxon_namespace)
# TT.print_plot(width = 80)
# T.print_plot(width = 80)
t3= time.time()

RF,F1 = reconstruct_tree.compare_trees(tree, TT)
print("Spectral: ")
print("RF = ",RF)
print("F1% = ",F1)
print("")
print("time:")
print("    Build Sim.: ", t2-t1)
print("    Find roots: ", t3-t2)