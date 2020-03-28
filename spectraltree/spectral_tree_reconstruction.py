import platform
from abc import ABC, abstractmethod
from functools import partial
from itertools import combinations
import os, sys

import numpy as np
import scipy.spatial.distance
from sklearn.decomposition import TruncatedSVD
import dendropy     #should this library be independent of dendropy? is that even possible?
from dendropy.interop import raxml
import utils
from character_matrix import FastCharacterMatrix

RECONSTRUCT_TREE_PATH = os.path.abspath(__file__)
RECONSTRUCT_TREE__DIR_PATH = os.path.dirname(RECONSTRUCT_TREE_PATH)

##########################################################
##               Reconstruction methods
##########################################################

def correlation_distance_matrix(observations):
    """Correlation Distance"""
    corr = np.abs(np.corrcoef(observations))
    corr = np.clip(corr, a_min=1e-16, a_max=None)
    return -np.log(corr)

def partition_taxa(v,similarity,num_gaps):
    m = len(v)
    #idx_vec = 
    v_sort = np.sort(v)
    gaps = v_sort[1:m]-v_sort[0:m-1]
    ind_partition = np.argpartition(gaps, -num_gaps)[-num_gaps:]
    smin = 1000
    for p_idx in ind_partition:        
        threshold = (v_sort[p_idx]+v_sort[p_idx+1])/2
        if (p_idx>0) & (p_idx<m-2):
            bool_bipartition = v<threshold
            s_sliced = similarity[bool_bipartition,:]
            s_sliced = s_sliced[:,~bool_bipartition]
            s2 = svd2(s_sliced)
            if s2<smin:
                partition_min = bool_bipartition
                smin = s2
    return partition_min   
    #max_idx = np.argmax(gaps)
    #threshold = (v_sort[max_idx]+v_sort[max_idx+1])/2
    #bool_bipartition = v<threshold
    #return bool_bipartition

def join_trees_with_spectral_root_finding(similarity_matrix, T1, T2, taxon_namespace=None):
    m, m2 = similarity_matrix.shape
    assert m == m2, "Distance matrix must be square"
    if taxon_namespace is None:
        taxon_namespace = utils.default_namespace(m)
    else:
        assert len(taxon_namespace) >= m, "Namespace too small for distance matrix"
    
    T = dendropy.Tree(taxon_namespace=taxon_namespace)

    # Extracting inecies from namespace    
    T1_labels = [x.taxon.label for x in T1.leaf_nodes()]
    T2_labels = [x.taxon.label for x in T2.leaf_nodes()]

    half1_idx_bool = [x.label in T1_labels for x in taxon_namespace]
    half1_idx = [i for i, x in enumerate(half1_idx_bool) if x]
    half1_idx_array = np.array(half1_idx)
    T1.is_rooted = True
    
    half2_idx_bool = [x.label in T2_labels for x in taxon_namespace]
    half2_idx = [i for i, x in enumerate(half2_idx_bool) if x]
    half2_idx_array = np.array(half2_idx)
    T2.is_rooted = True
    
    # finding roots
    
    # find root of half 1
    bipartitions1 = T1.bipartition_edge_map
    min_ev2 = float("inf")
    for bp in bipartitions1.keys():
        if bp.leafset_as_bitstring().find('0') == -1:
            continue
        # Slice similarity matrix of half2 + part of half1 VS the other part of half 1
        bool_array = np.array(list(map(bool,[int(i) for i in bp.leafset_as_bitstring()]))[::-1])
        h1_idx_A = half1_idx_array[bool_array]
        h1_idx_B = half1_idx_array[~bool_array]
        other_idx = np.array([], dtype = np.int32) #np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])), sub_idx)
    
        A_h2_other_idx = list(np.concatenate([h1_idx_A, half2_idx_array, other_idx]))
        B_h2_other_idx = list(np.concatenate([h1_idx_B, half2_idx_array, other_idx]))

        A_other_idx = list(np.concatenate([h1_idx_A, other_idx]))
        B_other_idx = list(np.concatenate([h1_idx_B, other_idx]))
        
        A_h2_idx = list(np.concatenate([h1_idx_A, half2_idx_array]))
        B_h2_idx = list(np.concatenate([h1_idx_B, half2_idx_array]))

        
        #all_minus_h1_idx_cur = list(np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])),h1_idx_cur))
        #all_minus_h1_idx_cur_complement = list(np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])),h1_idx_cur_complement))
        
        ####################### example of indexes ##################
        # FFFF FFFF TTTT TTTT - idx            - these are the element we are "solving for" at theis step
        # FFFF FFFF 1111 FFFF - half1_idx      - these are the elements assignd to half 1 (and described by T1)
        # FFFF FFFF FFFF 1111 - half2_idx      - these are the elements assignd to half 1 (and described by T1)
        # FFFF FFFF 0110 FFFF - h1_idx_A       - these are the elements chosen at the current partition (bp)
        # FFFF FFFF 1001 FFFF - h1_idx_B       - these are the elements chosen at the current partition (bp)
        # 1111 1111 FFFF FFFF - other_idx      - 
        
        # 1111 1111 1001 1111 - B_h2_other_idx - 
        # 1111 1111 0110 1111 - A_h2_other_idx - 
        # 1111 1111 0110 0000 - A_other_idx    - 
        # 1111 1111 1001 0000 - B_other_idx    - 
        # 0000 0000 0110 1111 - A_h2_idx       - 
        # 0000 0000 1001 1111 - B_h2_idx       - 
        
        ## Case 1: Other is connected to A
        sliced_sim_mat_try1_1 = similarity_matrix[A_other_idx, ...]
        sliced_sim_mat_try1_1 = sliced_sim_mat_try1_1[...,B_h2_idx]
        score_try1_1 = svd2(sliced_sim_mat_try1_1)

        sliced_sim_mat_try1_2 = similarity_matrix[A_h2_other_idx, ...]
        sliced_sim_mat_try1_2 = sliced_sim_mat_try1_2[...,h1_idx_B]
        score_try1_2 = svd2(sliced_sim_mat_try1_2)

        score_try1 = np.max([score_try1_1,score_try1_2])
        ## Case 2: Other is connected to B
        sliced_sim_mat_try2_1 = similarity_matrix[h1_idx_A, ...]
        sliced_sim_mat_try2_1 = sliced_sim_mat_try2_1[...,B_h2_other_idx]
        score_try2_1 = svd2(sliced_sim_mat_try2_1)
        
        sliced_sim_mat_try2_2 = similarity_matrix[A_h2_idx, ...]
        sliced_sim_mat_try2_2 = sliced_sim_mat_try2_2[...,B_other_idx]
        score_try2_2 = svd2(sliced_sim_mat_try2_2)
        
        score_try2 = np.max([score_try2_1,score_try2_2])
        
        ## Case 3: Other is connected to h2
        sliced_sim_mat_try3_1 = similarity_matrix[h1_idx_A, ...]
        sliced_sim_mat_try3_1 = sliced_sim_mat_try3_1[...,B_h2_other_idx]
        score_try3_1 = svd2(sliced_sim_mat_try3_1)
        
        sliced_sim_mat_try3_2 = similarity_matrix[A_h2_other_idx, ...]
        sliced_sim_mat_try3_2 = sliced_sim_mat_try3_2[...,h1_idx_B]
        score_try3_2 = svd2(sliced_sim_mat_try3_2)
        
        score_try3 = np.max([score_try3_1,score_try3_2])
        
        if np.min([score_try1, score_try2, score_try3]) <min_ev2:
            min_ev2 = np.min([score_try1, score_try2, score_try3])
            bp_min = bp
    # if sum([int(i) for i in bp_min.leafset_as_bitstring()]) != len([int(i) for i in bp_min.leafset_as_bitstring()])/2:
    #     print("ERROR: ")
    #     print("size of data:", m)
    #     print("Half sizes:", sum([int(i) for i in bp_min.leafset_as_bitstring()]), sum([-1*int(i)+1 for i in bp_min.leafset_as_bitstring()]))

    T1.reroot_at_edge(bipartitions1[bp_min])

    # find root of half 2
    bipartitions2 = T2.bipartition_edge_map
    min_ev2 = float("inf")
    for bp in bipartitions2.keys():
        if bp.leafset_as_bitstring().find('0') == -1:
            continue
        # Slice similarity matrix of half2 + part of half1 VS the other part of half 1
        
        bool_array = np.array(list(map(bool,[int(i) for i in bp.leafset_as_bitstring()]))[::-1])
        h2_idx_A = half2_idx_array[bool_array]
        h2_idx_B = half2_idx_array[~bool_array]
        other_idx = np.array([], dtype = np.int32)#np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])), sub_idx)

        A_h1_other_idx = list(np.concatenate([h2_idx_A, half1_idx_array, other_idx]))
        B_h1_other_idx = list(np.concatenate([h2_idx_B, half1_idx_array, other_idx]))

        A_other_idx = list(np.concatenate([h2_idx_A, other_idx]))
        B_other_idx = list(np.concatenate([h2_idx_B, other_idx]))

        A_h1_idx = list(np.concatenate([h2_idx_A, half1_idx_array]))
        B_h1_idx = list(np.concatenate([h2_idx_B, half1_idx_array]))

        ####################### example of indexes ##################
        # FFFF FFFF TTTT TTTT - idx            - these are the element we are "solving for" at theis step
        # FFFF FFFF 1111 FFFF - half1_idx      - these are the elements assignd to half 1 (and described by T1)
        # FFFF FFFF FFFF 1111 - half2_idx      - these are the elements assignd to half 1 (and described by T1)
        # FFFF FFFF FFFF 1100 - h1_idx_A       - these are the elements chosen at the current partition (bp)
        # FFFF FFFF FFFF 0011 - h1_idx_B       - these are the elements chosen at the current partition (bp)
        # 1111 1111 FFFF FFFF - other_idx      - 
        
        # 1111 1111 1111 0011 - B_h1_other_idx - 
        # 1111 1111 1111 1100 - A_h1_other_idx - 
        # 1111 1111 0000 1100 - A_other_idx    - 
        # 1111 1111 0000 0011 - B_other_idx    - 
        # 0000 0000 1111 1100 - A_h1_idx       - 
        # 0000 0000 1111 0011 - B_h1_idx       - 
        
        ## Case 1: Other is connected to A
        sliced_sim_mat_try1_1 = similarity_matrix[A_other_idx, ...]
        sliced_sim_mat_try1_1 = sliced_sim_mat_try1_1[...,B_h1_idx]
        score_try1_1 = svd2(sliced_sim_mat_try1_1)

        sliced_sim_mat_try1_2 = similarity_matrix[A_h1_other_idx, ...]
        sliced_sim_mat_try1_2 = sliced_sim_mat_try1_2[...,h2_idx_B]
        score_try1_2 =svd2(sliced_sim_mat_try1_2)

        score_try1 = np.max([score_try1_1,score_try1_2])
        ## Case 2: Other is connected to B
        sliced_sim_mat_try2_1 = similarity_matrix[h2_idx_A, ...]
        sliced_sim_mat_try2_1 = sliced_sim_mat_try2_1[...,B_h1_other_idx]
        score_try2_1 = svd2(sliced_sim_mat_try2_1)
        
        sliced_sim_mat_try2_2 = similarity_matrix[A_h1_idx, ...]
        sliced_sim_mat_try2_2 = sliced_sim_mat_try2_2[...,B_other_idx]
        score_try2_2 = svd2(sliced_sim_mat_try2_2)
        
        score_try2 = np.max([score_try2_1,score_try2_2])
        
        ## Case 3: Other is connected to h2
        sliced_sim_mat_try3_1 = similarity_matrix[h2_idx_A, ...]
        sliced_sim_mat_try3_1 = sliced_sim_mat_try3_1[...,B_h1_other_idx]
        score_try3_1 = svd2(sliced_sim_mat_try3_1)
        
        sliced_sim_mat_try3_2 = similarity_matrix[A_h1_other_idx, ...]
        sliced_sim_mat_try3_2 = sliced_sim_mat_try3_2[...,h2_idx_B]
        score_try3_2 = svd2(sliced_sim_mat_try3_2)
        
        score_try3 = np.max([score_try3_1,score_try3_2])
        
        if np.min([score_try1, score_try2, score_try3]) <min_ev2:
            min_ev2 = np.min([score_try1, score_try2, score_try3])
            bp_min = bp
    # if sum([int(i) for i in bp_min.leafset_as_bitstring()]) != len([int(i) for i in bp_min.leafset_as_bitstring()])/2:
    #     print("ERROR: ")
    #     print("size of data:", m)
    #     print("Half sizes:", sum([int(i) for i in bp_min.leafset_as_bitstring()]), sum([-1*int(i)+1 for i in bp_min.leafset_as_bitstring()]))
    T2.reroot_at_edge(bipartitions2[bp_min])
        
    T.seed_node.set_child_nodes([T1.seed_node,T2.seed_node])
    #T.seed_node.add_child(T1.seed_node)
    #T.seed_node.add_child(T2.seed_node)
    # if len(sub_idx) != similarity_matrix.shape[0]:
    #     T.is_rooted = True
    return T


def spectral_tree_reonstruction_old(similarity_matrix, namespace=None, sub_idx = None):
    # sub_idx  - internal variable. Do not change when calling the function
    m, m2 = similarity_matrix.shape
    if sub_idx == None:
        sub_idx = list(range(m))
    #print("size of data:", m, )
    assert m == m2, "Distance matrix must be square"
    if namespace is None:
        namespace = utils.default_namespace(m)
    else:
        assert len(namespace) >= m, "Namespace too small for distance matrix"
    
    similarity_matrix_small = similarity_matrix[sub_idx,...]
    similarity_matrix_small = similarity_matrix_small[...,sub_idx]
    if len(sub_idx) != similarity_matrix.shape[0]:
        T = dendropy.Tree(taxon_namespace= dendropy.TaxonNamespace([namespace[i] for i in sub_idx]))
    else:
        T = dendropy.Tree(taxon_namespace=namespace)
    
    #Reconstract tree with NJ for small tree. SHOULD BE CHANGED TO MAXIMUM LIKLHOOD
    if len(sub_idx)<=64:
        #distance_matrix = np.clip(similarity_matrix_small, a_min=1e-16, a_max=None)
        T = estimate_tree_topology(similarity_matrix_small, dendropy.TaxonNamespace([namespace[i] for i in sub_idx]))
        #T = neighbor_joining(distance_matrix, dendropy.TaxonNamespace([namespace[i] for i in sub_idx]))
        T.reroot_at_node(T.seed_node)
        return T
        

    w,v = scipy.linalg.eigh(similarity_matrix_small)
    eigv2 =v[:,-2]
    
    cut_val = np.median(eigv2)
    #cut_val = 0
    half1_idx = [sub_idx[i] for i, val in enumerate(eigv2<cut_val) if val] 
    half1_idx_array = np.array(half1_idx)
    #half1 = eigv2[half1_idx]
    # similarity_matrix1 = similarity_matrix[half1_idx,...]
    # similarity_matrix1 = similarity_matrix1[...,half1_idx]
    # namespace1 = dendropy.TaxonNamespace([namespace[i] for i in half1_idx])
    T1 = spectral_tree_reonstruction_old(similarity_matrix, namespace,sub_idx=half1_idx)
    T1.is_rooted = True

    half2_idx = [sub_idx[i] for i, val in enumerate(eigv2>cut_val) if val] 
    half2_idx_array = np.array(half2_idx)
    #half2 = eigv2[half2_idx]
    # similarity_matrix2 = similarity_matrix[half2_idx,...]
    # similarity_matrix2 = similarity_matrix2[...,half2_idx]
    # namespace2 = dendropy.TaxonNamespace([namespace[i] for i in half2_idx])
    T2 = spectral_tree_reonstruction_old(similarity_matrix, namespace, sub_idx=half2_idx)
    T2.is_rooted = True
    # finding roots
    
    # find root of half 1
    if len(half1_idx)>2:
        bipartitions1 = T1.bipartition_edge_map
        min_ev2 = float("inf")
        for bp in bipartitions1.keys():
            if bp.leafset_as_bitstring().find('0') == -1:
                continue
            # Slice similarity matrix of half2 + part of half1 VS the other part of half 1
            bool_array =np.array(list(map(bool,[int(i) for i in bp.leafset_as_bitstring()]))[::-1])
            h1_idx_A = half1_idx_array[bool_array]
            h1_idx_B = half1_idx_array[~bool_array]
            other_idx = np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])), sub_idx)
        
            A_h2_other_idx = list(np.concatenate([h1_idx_A, half2_idx_array, other_idx]))
            B_h2_other_idx = list(np.concatenate([h1_idx_B, half2_idx_array, other_idx]))

            A_other_idx = list(np.concatenate([h1_idx_A, other_idx]))
            B_other_idx = list(np.concatenate([h1_idx_B, other_idx]))
            
            A_h2_idx = list(np.concatenate([h1_idx_A, half2_idx_array]))
            B_h2_idx = list(np.concatenate([h1_idx_B, half2_idx_array]))

            #all_minus_h1_idx_cur = list(np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])),h1_idx_cur))
            #all_minus_h1_idx_cur_complement = list(np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])),h1_idx_cur_complement))
            
            ####################### example of indexes ##################
            # FFFF FFFF TTTT TTTT - idx            - these are the element we are "solving for" at theis step
            # FFFF FFFF 1111 FFFF - half1_idx      - these are the elements assignd to half 1 (and described by T1)
            # FFFF FFFF FFFF 1111 - half2_idx      - these are the elements assignd to half 1 (and described by T1)
            # FFFF FFFF 0110 FFFF - h1_idx_A       - these are the elements chosen at the current partition (bp)
            # FFFF FFFF 1001 FFFF - h1_idx_B       - these are the elements chosen at the current partition (bp)
            # 1111 1111 FFFF FFFF - other_idx      - 
            
            # 1111 1111 1001 1111 - B_h2_other_idx - 
            # 1111 1111 0110 1111 - A_h2_other_idx - 
            # 1111 1111 0110 0000 - A_other_idx    - 
            # 1111 1111 1001 0000 - B_other_idx    - 
            # 0000 0000 0110 1111 - A_h2_idx       - 
            # 0000 0000 1001 1111 - B_h2_idx       - 
            
            ## Case 1: Other is connected to A
            sliced_sim_mat_try1_1 = similarity_matrix[A_other_idx, ...]
            sliced_sim_mat_try1_1 = sliced_sim_mat_try1_1[...,B_h2_idx]
            score_try1_1 = 0 if (sliced_sim_mat_try1_1.shape[0] == 1) | (sliced_sim_mat_try1_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try1_1, compute_uv = False)[1]

            sliced_sim_mat_try1_2 = similarity_matrix[A_h2_other_idx, ...]
            sliced_sim_mat_try1_2 = sliced_sim_mat_try1_2[...,h1_idx_B]
            score_try1_2 = 0 if (sliced_sim_mat_try1_2.shape[0] == 1) | (sliced_sim_mat_try1_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try1_2, compute_uv = False)[1]

            score_try1 = np.max([score_try1_1,score_try1_2])
            ## Case 2: Other is connected to B
            sliced_sim_mat_try2_1 = similarity_matrix[h1_idx_A, ...]
            sliced_sim_mat_try2_1 = sliced_sim_mat_try2_1[...,B_h2_other_idx]
            score_try2_1 = 0 if (sliced_sim_mat_try2_1.shape[0] == 1) | (sliced_sim_mat_try2_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try2_1, compute_uv = False)[1]
            
            sliced_sim_mat_try2_2 = similarity_matrix[A_h2_idx, ...]
            sliced_sim_mat_try2_2 = sliced_sim_mat_try2_2[...,B_other_idx]
            score_try2_2 = 0 if (sliced_sim_mat_try2_2.shape[0] == 1) | (sliced_sim_mat_try2_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try2_2, compute_uv = False)[1]
            
            score_try2 = np.max([score_try2_1,score_try2_2])
            
            ## Case 3: Other is connected to h2
            sliced_sim_mat_try3_1 = similarity_matrix[h1_idx_A, ...]
            sliced_sim_mat_try3_1 = sliced_sim_mat_try3_1[...,B_h2_other_idx]
            score_try3_1 = 0 if (sliced_sim_mat_try3_1.shape[0] == 1) | (sliced_sim_mat_try3_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try3_1, compute_uv = False)[1]
            
            sliced_sim_mat_try3_2 = similarity_matrix[A_h2_other_idx, ...]
            sliced_sim_mat_try3_2 = sliced_sim_mat_try3_2[...,h1_idx_B]
            score_try3_2 = 0 if (sliced_sim_mat_try3_2.shape[0] == 1) | (sliced_sim_mat_try3_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try3_2, compute_uv = False)[1]
            
            score_try3 = np.max([score_try3_1,score_try3_2])
            
            if np.min([score_try1, score_try2, score_try3]) <min_ev2:
                min_ev2 = np.min([score_try1, score_try2, score_try3])
                bp_min = bp
        # if sum([int(i) for i in bp_min.leafset_as_bitstring()]) != len([int(i) for i in bp_min.leafset_as_bitstring()])/2:
        #     print("ERROR: ")
        #     print("size of data:", m)
        #     print("Half sizes:", sum([int(i) for i in bp_min.leafset_as_bitstring()]), sum([-1*int(i)+1 for i in bp_min.leafset_as_bitstring()]))

        T1.reroot_at_edge(bipartitions1[bp_min])
    
    # find root of half 2
    if len(half2_idx)>2:
        bipartitions2 = T2.bipartition_edge_map
        min_ev2 = float("inf")
        for bp in bipartitions2.keys():
            if bp.leafset_as_bitstring().find('0') == -1:
                continue
            # Slice similarity matrix of half2 + part of half1 VS the other part of half 1
            
            bool_array = np.array(list(map(bool,[int(i) for i in bp.leafset_as_bitstring()]))[::-1])
            h2_idx_A = half2_idx_array[bool_array]
            h2_idx_B = half2_idx_array[~bool_array]
            other_idx = np.setdiff1d(np.array(range(np.shape(similarity_matrix)[0])), sub_idx)

            A_h1_other_idx = list(np.concatenate([h2_idx_A, half1_idx_array, other_idx]))
            B_h1_other_idx = list(np.concatenate([h2_idx_B, half1_idx_array, other_idx]))

            A_other_idx = list(np.concatenate([h2_idx_A, other_idx]))
            B_other_idx = list(np.concatenate([h2_idx_B, other_idx]))

            A_h1_idx = list(np.concatenate([h2_idx_A, half1_idx_array]))
            B_h1_idx = list(np.concatenate([h2_idx_B, half1_idx_array]))

            ####################### example of indexes ##################
            # FFFF FFFF TTTT TTTT - idx            - these are the element we are "solving for" at theis step
            # FFFF FFFF 1111 FFFF - half1_idx      - these are the elements assignd to half 1 (and described by T1)
            # FFFF FFFF FFFF 1111 - half2_idx      - these are the elements assignd to half 1 (and described by T1)
            # FFFF FFFF FFFF 1100 - h1_idx_A       - these are the elements chosen at the current partition (bp)
            # FFFF FFFF FFFF 0011 - h1_idx_B       - these are the elements chosen at the current partition (bp)
            # 1111 1111 FFFF FFFF - other_idx      - 
            
            # 1111 1111 1111 0011 - B_h1_other_idx - 
            # 1111 1111 1111 1100 - A_h1_other_idx - 
            # 1111 1111 0000 1100 - A_other_idx    - 
            # 1111 1111 0000 0011 - B_other_idx    - 
            # 0000 0000 1111 1100 - A_h1_idx       - 
            # 0000 0000 1111 0011 - B_h1_idx       - 
            
            ## Case 1: Other is connected to A
            sliced_sim_mat_try1_1 = similarity_matrix[A_other_idx, ...]
            sliced_sim_mat_try1_1 = sliced_sim_mat_try1_1[...,B_h1_idx]
            score_try1_1 = 0 if (sliced_sim_mat_try1_1.shape[0] == 1) | (sliced_sim_mat_try1_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try1_1, compute_uv = False)[1]

            sliced_sim_mat_try1_2 = similarity_matrix[A_h1_other_idx, ...]
            sliced_sim_mat_try1_2 = sliced_sim_mat_try1_2[...,h2_idx_B]
            score_try1_2 = 0 if (sliced_sim_mat_try1_2.shape[0] == 1) | (sliced_sim_mat_try1_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try1_2, compute_uv = False)[1]

            score_try1 = np.max([score_try1_1,score_try1_2])
            ## Case 2: Other is connected to B
            sliced_sim_mat_try2_1 = similarity_matrix[h2_idx_A, ...]
            sliced_sim_mat_try2_1 = sliced_sim_mat_try2_1[...,B_h1_other_idx]
            score_try2_1 = 0 if (sliced_sim_mat_try2_1.shape[0] == 1) | (sliced_sim_mat_try2_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try2_1, compute_uv = False)[1]
            
            sliced_sim_mat_try2_2 = similarity_matrix[A_h1_idx, ...]
            sliced_sim_mat_try2_2 = sliced_sim_mat_try2_2[...,B_other_idx]
            score_try2_2 = 0 if (sliced_sim_mat_try2_2.shape[0] == 1) | (sliced_sim_mat_try2_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try2_2, compute_uv = False)[1]
            
            score_try2 = np.max([score_try2_1,score_try2_2])
            
            ## Case 3: Other is connected to h2
            sliced_sim_mat_try3_1 = similarity_matrix[h2_idx_A, ...]
            sliced_sim_mat_try3_1 = sliced_sim_mat_try3_1[...,B_h1_other_idx]
            score_try3_1 = 0 if (sliced_sim_mat_try3_1.shape[0] == 1) | (sliced_sim_mat_try3_1.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try3_1, compute_uv = False)[1]
            
            sliced_sim_mat_try3_2 = similarity_matrix[A_h1_other_idx, ...]
            sliced_sim_mat_try3_2 = sliced_sim_mat_try3_2[...,h2_idx_B]
            score_try3_2 = 0 if (sliced_sim_mat_try3_2.shape[0] == 1) | (sliced_sim_mat_try3_2.shape[1] == 1) else scipy.linalg.svd(sliced_sim_mat_try3_2, compute_uv = False)[1]
            
            score_try3 = np.max([score_try3_1,score_try3_2])
            
            if np.min([score_try1, score_try2, score_try3]) <min_ev2:
                min_ev2 = np.min([score_try1, score_try2, score_try3])
                bp_min = bp
        # if sum([int(i) for i in bp_min.leafset_as_bitstring()]) != len([int(i) for i in bp_min.leafset_as_bitstring()])/2:
        #     print("ERROR: ")
        #     print("size of data:", m)
        #     print("Half sizes:", sum([int(i) for i in bp_min.leafset_as_bitstring()]), sum([-1*int(i)+1 for i in bp_min.leafset_as_bitstring()]))
        T2.reroot_at_edge(bipartitions2[bp_min])
        
    T.seed_node.set_child_nodes([T1.seed_node,T2.seed_node])
    #T.seed_node.add_child(T1.seed_node)
    #T.seed_node.add_child(T2.seed_node)
    if len(sub_idx) != similarity_matrix.shape[0]:
        T.is_rooted = True
    return T

##########################################################
##               Testing
##########################################################


def compare_trees(reference_tree, inferred_tree):
    inferred_tree.update_bipartitions()
    reference_tree.update_bipartitions()
    false_positives, false_negatives = dendropy.calculate.treecompare.false_positives_and_negatives(reference_tree, inferred_tree, is_bipartitions_updated=True)
    total_reference = len(reference_tree.bipartition_encoding)
    total_inferred = len(inferred_tree.bipartition_encoding)
    
    true_positives = total_inferred - false_positives
    precision = true_positives / total_inferred
    true_positives = total_inferred - false_positives
    recall = true_positives / total_reference


    F1 = 100* 2*(precision * recall)/(precision + recall)
    RF = false_positives + false_negatives
    return RF, F1

def check_partition_in_tree(tree,partition):
    
    # UGLY!! The bipartitions in 'tree' is given by order of taxonomy (T1,T2 etc)
    # The input 'partition' is via order of the appearnace in the matrix 
    # The first step is to switch partition such that it is of the same order of tree
    ns = []
    for leaf_ix, leaf in enumerate(tree.leaf_node_iter()):    
        ns.append(int(leaf.taxon.label))
    ns_idx = np.argsort(np.array(ns))
    partition = partition[ns_idx]
    partition_inv = [not i for i in partition]
    

    #print("*********************")
    tree_bipartitions = tree.bipartition_edge_map
    flag = False
    for bp in tree_bipartitions.keys():
        bool_array =np.array(list(map(bool,[int(i) for i in bp.leafset_as_bitstring()]))[::-1])
        #bool_array = bool_array[ns_idx]
        #print(bool_array)
        if np.array_equal(bool_array,partition) or np.array_equal(bool_array,partition_inv):
            flag = True
    return flag  

class SpectralTreeReconstruction(ReconstructionMethod):
    def __init__(self, inner_method, similarity_metric):
        self.inner_method = inner_method
        self.similarity_metric = similarity_metric
        if issubclass(inner_method, DistanceReconstructionMethod):
            self.reconstruction_alg = inner_method(similarity_metric)
        else:
            self.reconstruction_alg = inner_method()
            
    def __call__(self, sequences, taxon_namespace=None):
        return self.spectral_tree_reonstruction(sequences, self.similarity_metric, taxon_namespace=taxon_namespace, reconstruction_alg = self.inner_method)
    
    def __repr__(self):
        return "spectralTree"

    def deep_spectral_tree_reonstruction(self, sequences, similarity_metric,taxon_namespace = None, num_gaps =1,threshhold = 100, **kargs):
        self.sequences = sequences
        self.similarity_matrix = similarity_metric(sequences)
        m, m2 = self.similarity_matrix.shape
        assert m == m2, "Distance matrix must be square"
        self.taxon_namespace = taxon_namespace
        if self.taxon_namespace is None:
            self.taxon_namespace = utils.default_namespace(m)
        else:
            assert len(self.taxon_namespace) >= m, "Namespace too small for distance matrix"

        partitioning_tree = MyTree([True]*len(self.taxon_namespace))
        cur_node = partitioning_tree.root
        while True:
            if (cur_node.right != None) and (cur_node.right.tree !=None) and (cur_node.left.tree != None):
                cur_node.tree = self.margeTreesLeftRight(cur_node)
                if cur_node.parent == None:
                    break
                if cur_node.parent.right == cur_node:
                    cur_node = cur_node.parent.left
                else:
                    cur_node = cur_node.parent
            elif sum(cur_node.bitmap) > threshhold:
                L1,L2 = self.splitTaxa(cur_node,num_gaps)
                cur_node.setLeft(MyNode(L1))
                cur_node.setRight(MyNode(L2))
                cur_node = cur_node.right
            else:
                cur_node.tree = self.reconstruct_alg_wrapper(cur_node, **kargs)
                if cur_node.parent == None:
                    break
                if cur_node.parent.right == cur_node:
                    cur_node = cur_node.parent.left
                else:
                    cur_node = cur_node.parent
        partitioning_tree.root.tree.taxon_namespace = self.taxon_namespace
        return partitioning_tree.root.tree
        
    def splitTaxa(self,node,num_gaps):
        cur_similarity = self.similarity_matrix[node.bitmap,:]
        cur_similarity = cur_similarity[:,node.bitmap]
        [D,V] = np.linalg.eigh(cur_similarity)
        bool_bipartition = partition_taxa(V[:,-2],cur_similarity,num_gaps)
        
        #Building partitioning bitmaps from partial bitmaps
        ll = np.array([i for i, x in enumerate(node.bitmap) if x])
        ll1 = ll[bool_bipartition]
        not_bool_bipartition = [~i for i in bool_bipartition]
        ll2 = ll[not_bool_bipartition]
        bitmap1 = [True if i in ll1 else False for i in range(len(self.taxon_namespace))]
        bitmap2 = [True if i in ll2 else False for i in range(len(self.taxon_namespace))]
        return bitmap1, bitmap2

    def margeTreesLeftRight(self, node):
        cur_namespace = dendropy.TaxonNamespace([self.taxon_namespace[i] for i in [i for i, x in enumerate(node.bitmap) if x]])  
        cur_similarity = self.similarity_matrix[node.bitmap,:]
        cur_similarity = cur_similarity[:,node.bitmap]
        return join_trees_with_spectral_root_finding(cur_similarity, node.left.tree, node.right.tree, taxon_namespace=cur_namespace)
    
    def reconstruct_alg_wrapper(self, node, **kargs):
        namespace1 = dendropy.TaxonNamespace([self.taxon_namespace[i] for i in [i for i, x in enumerate(node.bitmap) if x]]) 
        if issubclass(self.inner_method, DistanceReconstructionMethod):
            similarity_matrix1 = self.similarity_matrix[node.bitmap,:]
            similarity_matrix1 = similarity_matrix1[:,node.bitmap]
            return self.reconstruction_alg.reconstruct_from_similarity(similarity_matrix1, taxon_namespace = namespace1)
        else:
            # print(type(self.sequences))
            sequences1 = self.sequences[node.bitmap,:]
            return self.reconstruction_alg(sequences1, taxon_namespace = namespace1, **kargs)

    def spectral_tree_reonstruction(self, sequences, similarity_metric, taxon_namespace=None, reconstruction_alg = SpectralNeighborJoining(None)):
        similarity_matrix = similarity_metric(sequences)
        m, m2 = similarity_matrix.shape
        assert m == m2, "Distance matrix must be square"
        if taxon_namespace is None:
            taxon_namespace = utils.default_namespace(m)
        else:
            assert len(taxon_namespace) >= m, "Namespace too small for distance matrix"
        
        # Partitioning
        [D,V] = np.linalg.eigh(similarity_matrix)            
        bool_bipartition = partition_taxa(V[:,-2],similarity_matrix,1)
        
        similarity_matrix1 = similarity_matrix[bool_bipartition,:]
        similarity_matrix1 = similarity_matrix1[:, bool_bipartition]

        not_bool_bipartition = [~i for i in bool_bipartition]
        similarity_matrix2 = similarity_matrix[not_bool_bipartition,:]
        similarity_matrix2 = similarity_matrix2[:, not_bool_bipartition]
        
        namespace1 = dendropy.TaxonNamespace([taxon_namespace[i] for i in [i for i, x in enumerate(bool_bipartition) if x]])
        namespace2 = dendropy.TaxonNamespace([taxon_namespace[i] for i in [i for i, x in enumerate(not_bool_bipartition) if x]])

        #reconstructing each part
        if issubclass(reconstruction_alg, DistanceReconstructionMethod):
            method = reconstruction_alg(similarity_metric)
            T1 = method.reconstruct_from_similarity(similarity_matrix1, taxon_namespace = namespace1)
            T2 = method.reconstruct_from_similarity(similarity_matrix2, taxon_namespace = namespace2)
        else:
            sequences1 = sequences[bool_bipartition,:]
            sequences2 = sequences[not_bool_bipartition,:]
            method = reconstruction_alg()
            T1 = method(sequences1,  taxon_namespace = namespace1)
            T2 = method(sequences2,  taxon_namespace = namespace2)

        # Finding roots and merging trees
        T = join_trees_with_spectral_root_finding(similarity_matrix, T1, T2, taxon_namespace=taxon_namespace)
        return T
    
class MyNode(object):
    def __init__(self, data):
        self.bitmap = data
        self.tree = None
        self.left = None
        self.right = None
        self.parent = None
    def setLeft(self,node):
        self.left = node
        node.parent = self
    def setRight(self,node):
        self.right = node
        node.parent = self
class MyTree(object):
    def __init__(self, data = None):
        self.root = MyNode(data)
    def setLeft(self,node):
        self.left = node
        node.parent = self
    def setRight(self,node):
        self.right = node
        node.parent = self

# %%
if __name__ == "__main__":
    pass