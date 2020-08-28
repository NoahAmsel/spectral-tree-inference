import platform
from abc import ABC, abstractmethod
from functools import partial
from itertools import combinations
import os, sys
import numpy as np
import scipy.spatial.distance
from sklearn.decomposition import TruncatedSVD
from itertools import product
from itertools import combinations
import dendropy     #should this library be independent of dendropy? is that even possible?
from dendropy.interop import raxml
import utils
import time
import os
import psutil
import reconstruct_tree
from numba import jit
from sklearn.utils.extmath import randomized_svd
import oct2py
from oct2py import octave
import scipy
import multiprocessing as mp

RECONSTRUCT_TREE_PATH = os.path.abspath(__file__)
RECONSTRUCT_TREE__DIR_PATH = os.path.dirname(RECONSTRUCT_TREE_PATH)


def compute_score(bp,taxa_metadata,T1,T2_mask,similarity_matrix, u_12,sigma_12, v_12, O1, merge_method):      
    mask1A = taxa_metadata.bipartition2mask(bp)
    mask1B = (T1.mask ^ mask1A)
    score = reconstruct_tree.compute_merge_score(mask1A, mask1B, T2_mask, similarity_matrix, 
        u_12[:,0],sigma_12[0], v_12[0,:], O1, merge_method)
    return score

def join_trees_with_spectral_root_finding_par(similarity_matrix, T1, T2, merge_method, taxa_metadata, verbose = False):
    
           
    
    m, m2 = similarity_matrix.shape
    assert m == m2, "Distance matrix must be square"
    assert T1.taxon_namespace == T2.taxon_namespace == taxa_metadata.taxon_namespace
    assert len(taxa_metadata) == m, "TaxaMetadata must correspond to the similarity matrix"
    


    # the tree we get after combining T1 and T2
    T = dendropy.Tree(taxon_namespace=taxa_metadata.taxon_namespace)

    """
    # Extracting inecies from namespace    
    T1_labels = [x.taxon.label for x in T1.leaf_nodes()]
    T2_labels = [x.taxon.label for x in T2.leaf_nodes()]

    half1_idx_bool = [x.label in T1_labels for x in taxa_metadata]
    half1_idx = [i for i, x in enumerate(half1_idx_bool) if x]
    T1.is_rooted = True
    
    half2_idx_bool = [x.label in T2_labels for x in taxa_metadata]
    half2_idx = [i for i, x in enumerate(half2_idx_bool) if x]  
    T2.is_rooted = True
    
    # Get sbmatrix of siilarities between nodes in subset 1 
    S_11 = similarity_matrix[half1_idx,:]
    S_11 = S_11[:,half1_idx]
    # Get sbmatrix of siilarities between nodes in subset 2
    S_22 = similarity_matrix[half2_idx,:]
    S_22 = S_22[:,half2_idx]
    # Get sbmatrix of cross similarities between nodes in subsets 1 and 2
    S_12 = similarity_matrix[half1_idx,:]
    S_12 = S_12[:,half2_idx]
    """

    T1_mask = taxa_metadata.tree2mask(T1)
    T2_mask = taxa_metadata.tree2mask(T2)
    # Make sure this is necessary
    T1.is_rooted = True
    T2.is_rooted = True

    S_12 = similarity_matrix[np.ix_(T1_mask, T2_mask)]

    # TODO: truncated svd?
    # u_12 is matrix
    [u_12,sigma_12,v_12] = np.linalg.svd(S_12)
    O1 = np.outer(u_12[:,0],u_12[:,0])
    O2 = np.outer(v_12[0,:],v_12[0,:])

    # find root of half 1
    bipartitions1 = T1.bipartition_edge_map.keys()

    if verbose: print("len(bipartitions1)", len(bipartitions1))

    if len(bipartitions1) ==2:
        print("NOOOOO")
    if len(bipartitions1) > 1:
        min_score = float("inf")
        results = {'sizeA': [], 'sizeB': [], 'score': []}
        
        time_s = time.time()
        pool = mp.Pool(mp.cpu_count())
        score_list = [pool.apply(compute_score, args=(bp,taxa_metadata,T1,T2_mask,
            similarity_matrix, u_12,sigma_12, v_12, O1, merge_method)) for bp in bipartitions1]
        #score_list = pool.map(compute_score(), [bp for bp in bipartitions1])        
        pool.close()
        min_score_par = np.min(score_list)
        min_idx_par = np.argmin(score_list)
        runtime_par = time.time()-time_s

        time_s = time.time()
        for bp in bipartitions1:
            mask1A = taxa_metadata.bipartition2mask(bp)
            mask1B = (T1.mask ^ mask1A)

            score = reconstruct_tree.compute_merge_score(mask1A, mask1B, T2_mask, similarity_matrix, u_12[:,0],sigma_12[0], v_12[0,:], O1, merge_method)
            # DELETE ME bool_array = taxa_metadata.bipartition2mask(bp)
            #DELETE ME score = compute_merge_score(bool_array,S_11,S_12,u_12[:,0],sigma_12[0],v_12[0,:],O1,merge_method)
            
            #results['sizeA'].append(mask1A.sum())
            #results['sizeB'].append(mask1B.sum())
            #results['score'].append(score)
            #results.append([sum(bool_array),sum(~bool_array), score])
            if score <min_score:
                min_score = score
                bp_min = bp
                min_mask1A = mask1A

        runtime_seq = time.time()-time_s
        print('score par: ',min_score_par, 'score_seq: ',min_score)
        print('runtime par: ',runtime_par, 'runtime_seq: ',runtime_seq)

        #bool_array = np.array(list(map(bool,[int(i) for i in bp_min.leafset_as_bitstring()]))[::-1])
        if verbose:
            if min_mask1A.sum()==1:
                print('one')
        if verbose: print("one - merging: ",min_mask1A.sum(), " out of: ", T1_mask.sum())
        
        T1.reroot_at_edge(T1.bipartition_edge_map[bp_min])
    #if len(bipartitions) > 1: 
    #    T1.reroot_at_edge(T1.bipartition_edge_map[bp_min])

    # find root of half 2
    #[u_12,s,v_12] = np.linalg.svd(S_12.T)
    
    bipartitions2 = T2.bipartition_edge_map.keys()
    if verbose: print("len(bipartitions2)", len(bipartitions2))
    # if len(bipartitions2) ==2:
    #     print("NOOOOO")
    if len(bipartitions2) > 1:
        min_score = float("inf")
        results2 = {'sizeA': [], 'sizeB': [], 'score': []}
        for bp in bipartitions2:
            mask2A = taxa_metadata.bipartition2mask(bp)
            mask2B = (T2.mask ^ mask2A)

            score = compute_merge_score(mask2A, mask2B, T1_mask, similarity_matrix, v_12[0,:],sigma_12[0], u_12[:,0], O2, merge_method)
            # DELETE ME bool_array = taxa_metadata.bipartition2mask(bp)
            #DELETE ME score = compute_merge_score(bool_array,S_11,S_12,u_12[:,0],sigma_12[0],v_12[0,:],O1,merge_method)
            
            results2['sizeA'].append(mask2A.sum())
            results2['sizeB'].append(mask2B.sum())
            results2['score'].append(score)
            #results.append([sum(bool_array),sum(~bool_array), score])
            if score <min_score:
                min_score = score
                bp_min2 = bp
                min_mask2A = mask2A
        if verbose:
            if min_mask2A.sum()==1:
                print('one')
        if verbose: print("one - merging: ",min_mask2A.sum(), " out of: ", T2_mask.sum())
        T2.reroot_at_edge(T2.bipartition_edge_map[bp_min2])
        #if len(bipartitions2) > 1: 
        #    T2.reroot_at_edge(T2.bipartition_edge_map[bp_min2])

    T.seed_node.set_child_nodes([T1.seed_node,T2.seed_node])
    return T
