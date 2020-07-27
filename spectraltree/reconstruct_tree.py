import platform
from abc import ABC, abstractmethod
from functools import partial
from itertools import combinations
import os, sys

import numpy as np
import scipy.spatial.distance
from sklearn.decomposition import TruncatedSVD
from itertools import product
import dendropy     #should this library be independent of dendropy? is that even possible?
from dendropy.interop import raxml
import utils
import time
import os
import psutil
from numba import jit
from sklearn.utils.extmath import randomized_svd

RECONSTRUCT_TREE_PATH = os.path.abspath(__file__)
RECONSTRUCT_TREE__DIR_PATH = os.path.dirname(RECONSTRUCT_TREE_PATH)

def sv2(A1, A2, M):
    """Second Singular Value"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        return s[1] # the second eigenvalue

def sum_squared_quartets(A1, A2, M):
    """Normalized Sum of Squared Quartets"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    norm_sq = np.linalg.norm(M_A)**2
    num = norm_sq**2 - np.linalg.norm(M_A.T.dot(M_A))**2
    return num / norm_sq

def paralinear_similarity(observations, taxa_metadata=None):
    # build the similarity matrix M -- this is pretty optimized, at least assuming k << n
    # setting classes allows you to ignore missing data. e.g. set class to 1,2,3,4 and it will
    # treat np.nan, -1, and 5 as missing data
    
    # TODO: clean and then delete these two
    #if classes is None:
    #    classes = np.unique(observations)
    classes = np.unique(observations)

    observations_one_hot = np.array([observations==cls for cls in classes], dtype='int')
    # this command makes an (m x m) array of (k x k) confusion matrices
    # where m = number of leaves and k = number of classes
    confusion_matrices = np.einsum("jik,mlk->iljm", observations_one_hot, observations_one_hot)
    # valid_observations properly accounts for missing data I think
    valid_observations = confusion_matrices.sum(axis=(2,3), keepdims=True)
    M = np.linalg.det(confusion_matrices/valid_observations)     # same as `confusion_matrices/n' when there aren't null entries
    #M = np.linalg.det(confusion_matrices/sqrt(n))

    diag = M.diagonal()
    # assert np.count_nonzero(diag) > 0, "Sequence was so short that some characters never appeared"
    similarity = M / np.sqrt(np.outer(diag,diag))
    return similarity

def paralinear_distance(observations, taxa_metadata=None):
    similarity = paralinear_similarity(observations)
    similarity = np.clip(similarity, a_min=1e-20, a_max=None)
    return -np.log(similarity)

def JC_similarity_matrix(observations, taxa_metadata=None):
    """Jukes-Cantor Corrected Similarity"""
    classes = np.unique(observations)
    if classes.dtype == np.dtype('<U1'):
        # needed to use hamming distance with string arrays
        vord = np.vectorize(ord)
        observations = vord(observations)
    k = len(classes)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    inside_log = 1 - hamming_matrix*k/(k-1)
    return inside_log**(k-1)

def JC_distance_matrix(observations, taxa_metadata=None):
    """Jukes-Cantor Corrected Distance"""
    inside_log = JC_similarity_matrix(observations, taxa_metadata)
    inside_log = np.clip(inside_log, a_min=1e-16, a_max=None)
    return - np.log(inside_log)

def estimate_edge_lengths(tree, distance_matrix):
    # check the PAUP* documentation
    pass


def hamming_dist_missing_values(vals, missing_val):
    classnames, indices = np.unique(vals, return_inverse=True)
    num_arr = indices.reshape(vals.shape)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(num_arr, metric='hamming'))
    missing_array = (vals==missing_val)
    pdist_xor = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_xor(u,v))))
    pdist_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_or(u,v))))
    
    return (hamming_matrix*vals.shape[1] - pdist_xor) / (np.ones_like(hamming_matrix) * vals.shape[1] - pdist_or)

def HKY_similarity_matrix(observations, taxa_metadata, verbose = False):
    m, N = observations.shape

    assert taxa_metadata.alphabet in [dendropy.DNA_STATE_ALPHABET, dendropy.RNA_STATE_ALPHABET]

    if verbose: print("Computing the average base frequency for each pair of sequences...")
    g = {}

    # TODO: ignore other indice too, not just "-"
    not_missing = observations != 4
    not_missing_sum = np.sum(not_missing, axis = 1) # # not missing for each taxon
    not_missing_pair = np.array([a + b for a, b in product(not_missing_sum, repeat = 2)]).reshape((m, m))

    for x, ix in [("A",0), ("C",1), ("G",2), ("T",3)]:
        obs_x = (observations == ix)
        g[x] = np.array([np.sum(np.hstack([a, b])) for a, b in product(obs_x, repeat = 2)]).reshape((m, m))
        g[x] = g[x] / not_missing_pair

    # in DNA_STATE_ALPHABET, 0 => A, 1 => C, 2 => G, 3 => T (or U in RNA)
    # Purines are A and G
    # Pyrimidines are C and T (or U)

    g["purines"] = g["A"] + g["G"]
    g["pyrimidines"] = g["C"] + g["T"]

    # compute transition and transversion proportion
    if verbose: print("Computing transition and transversion proportion for each pair of sequences...")

    P_1 = np.zeros((m,m))
    P_2 = np.zeros((m,m))

    A = hamming_dist_missing_values(observations, missing_val = 4)

    for i in range(m):
        for j in range(i + 1, m):
            neither_missing = np.logical_and(not_missing[i,:], not_missing[j,:])
            a = observations[i,:][neither_missing]
            b = observations[j,:][neither_missing]
            
            # in DNA_STATE_ALPHABET, 0 => A, 1 => C, 2 => G, 3 => T (or U in RNA)
            A_G = np.mean(np.logical_and(a == 0, b == 2) + np.logical_and(a == 2, b == 0))
            P_1[i, j] = P_1[j, i] = A_G
            
            C_T = np.mean(np.logical_and(a == 1, b == 3) + np.logical_and(a == 3, b == 1))
            P_2[i, j] = P_2[j, i] = C_T
            
    Q = A - P_1 - P_2
    
    #print("P", P_1, P_2)
    #print("Q", Q)
    # compute the similarity (formula 7)
    if verbose: print("Computing similarity matrix")
    R = (1 - g["purines"]/(2 * g["A"] * g["G"]) * P_1 - 1 / (2 * g["purines"]) * Q)
    Y = (1 - g["pyrimidines"]/(2 * g["T"] * g["C"]) * P_2 - 1 / (2 * g["pyrimidines"]) * Q)
    T = (1 - 1/(2 * g["purines"] * g["pyrimidines"]) * Q)
    S = np.sign(R) * (np.abs(R))**(8 * g["A"] * g["G"] / g["purines"])
    S *= np.sign(Y) * (np.abs(Y))**(8 * g["T"] * g["C"] / g["pyrimidines"])
    S *= np.sign(T) * (np.abs(T))**(8 * (g["purines"] * g["pyrimidines"] - g["A"] * g["G"] * g["pyrimidines"] / g["purines"] - g["T"] * g["C"] * g["purines"] / g["pyrimidines"]))

    return np.maximum(S,np.zeros_like(S))

"""
def HKY_similarity_matrix(observations, classes=None, verbose = False):
    m, N = observations.shape
    if classes is None:
        classes = np.unique(observations)
    k = len(classes)
    # From Tamura, K., and M. Nei. 1993
    # for each pair of sequences, 
    # 1. estimate the average base frequency for pairs of sequences
    # 2. compute purine transition proportion P1 (A <-> G)
    # 3. compute pyrimidine transition proportion P2 (T <-> C)
    # 3. compute transversion proportion Q (A <-> C, A <-> T, G <-> C, G <-> T)

    if verbose: print("Computing the average base frequency for each pair of sequences...")
    g = {}
    
    not_missing = observations != "-"
    not_missing_sum = np.sum(not_missing, axis = 1) # # not missing for each taxon
    not_missing_pair = np.array([a + b for a, b in product(not_missing_sum, repeat = 2)]).reshape((m, m))
    
    for x in classes:
        obs_x = observations == x 
        g[x] = np.array([np.sum(np.hstack([a, b])) for a, b in product(obs_x, repeat = 2)]).reshape((m, m))
        g[x] = g[x] / not_missing_pair

    
    g["R"] = g["A"] + g["G"]
    g["Y"] = g["T"] + g["C"]
    
    # compute transition and transversion proportion
    if verbose: print("Computing transition and transversion proportion for each pair of sequences...")
        
    P_1 = np.zeros((m,m))
    P_2 = np.zeros((m,m))
    
    A = hamming_dist_missing_values(observations, missing_val = "-")
    
    for i in range(m):
        for j in range(i + 1, m):
            neither_missing = np.logical_and(not_missing[i,:], not_missing[j,:])
            a = observations[i,:][neither_missing]
            b = observations[j,:][neither_missing]
            
            A_G = np.mean(np.logical_and(a == "A", b == "G") + np.logical_and(a == "G", b == "A"))
            P_1[i, j] = P_1[j, i] = A_G
            
            C_T = np.mean(np.logical_and(a == "C", b == "T") + np.logical_and(a == "T", b == "C"))
            P_2[i, j] = P_2[j, i] = C_T
            
    Q = A - P_1 - P_2
    
    #print("P", P_1, P_2)
    #print("Q", Q)
    # compute the similarity (formula 7)
    if verbose: print("Computing similarity matrix")
    R = (1 - g["R"]/(2 * g["A"] * g["G"]) * P_1 - 1 / (2 * g["R"]) * Q)
    Y = (1 - g["Y"]/(2 * g["T"] * g["C"]) * P_2 - 1 / (2 * g["Y"]) * Q)
    T = (1 - 1/(2 * g["R"] * g["Y"]) * Q)
    S = np.sign(R) * (np.abs(R))**(8 * g["A"] * g["G"] / g["R"])
    S *= np.sign(Y) * (np.abs(Y))**(8 * g["T"] * g["C"] / g["Y"])
    S *= np.sign(T) * (np.abs(T))**(8 * (g["R"] * g["Y"] - g["A"] * g["G"] * g["Y"] / g["R"] - g["T"] * g["C"] * g["R"] / g["Y"]))

    return np.maximum(S,np.zeros_like(S))

"""

##########################################################
##               Reconstruction methods
##########################################################

def correlation_distance_matrix(observations):
    """Correlation Distance"""
    corr = np.abs(np.corrcoef(observations))
    corr = np.clip(corr, a_min=1e-16, a_max=None)
    return -np.log(corr)

def raxml_reconstruct():
    pass

def partition_taxa(v,similarity,num_gaps = 1, min_split = 1):
    # partition_taxa2(v,similarity,num_gaps = 1, min_split = 1) partitions the vector vusing the threshold 0
    # or the best threshold (out of num_gaps posibbilitys, picked by the gaps in v) according to 
    # the second singular value of the partition.
    m = len(v)
    partition_min = v>0
    if np.minimum(np.sum(partition_min),np.sum(~partition_min))<min_split:
        if num_gaps == 0:
            raise Exception("Error: partition smaller than min_split. Increase num_gaps, or decrese min_split")
        else:
            smin = np.inf
    if num_gaps > 0:
        
        if np.minimum(np.sum(partition_min),np.sum(~partition_min))>=min_split:
            s_sliced = similarity[partition_min,:]
            s_sliced = s_sliced[:,~partition_min]
            smin = svd2(s_sliced)

        v_sort = np.sort(v)
        gaps = v_sort[min_split:m-min_split+1]-v_sort[min_split-1:m-min_split]
        sort_idx = np.argsort(gaps)
        for i in range(1, num_gaps+1):
            threshold = (v_sort[sort_idx[-i]+min_split-1]+v_sort[sort_idx[-i]+min_split])/2
            bool_bipartition = v<threshold
            if np.minimum(np.sum(bool_bipartition),np.sum(~bool_bipartition))>=min_split:
                s_sliced = similarity[bool_bipartition,:]
                s_sliced = s_sliced[:,~bool_bipartition]
                s2 = svd2(s_sliced)
                if s2<smin:
                    partition_min = bool_bipartition
                    smin = s2
    if smin == np.inf:
        raise Exception("Error: partition smaller than min_split. Increase num_gaps, or decrese min_split")
    return partition_min

SVD2_OBJ = TruncatedSVD(n_components=2, n_iter=7)
def svd2(mat, normalized = False):
    if (mat.shape[0] == 1) | (mat.shape[1] == 1):
        return 0
    elif (mat.shape[0] == 2) | (mat.shape[1] == 2):
        return np.linalg.svd(mat,False,False)[1]
    else:
        sigmas = SVD2_OBJ.fit(mat).singular_values_
        if normalized:
            return sigmas[1]**2/(sigmas[1]**2 + sigmas[0]**2)
        else: 
            return sigmas[1]

def compute_alpha_tensor(S_11,S_12,u_12,v_12,bool_array,sigma):
    # set indices
    S_AB = S_11[bool_array,:]
    S_AB = S_AB[:,~bool_array]
    S_A2 = S_12[bool_array,:]
    S_B2 = S_12[~bool_array,:]
    u_A = u_12[bool_array]
    u_B = u_12[~bool_array]

    m = len(u_A)*len(u_B)*len(v_12)
    U_T = np.zeros((m,1))
    S_T = np.zeros((m,1))
    
    ctr = 0
    for i in range(len(u_A)):
        for j in range(len(u_B)):
            for k in range(len(v_12)):
                U_T[ctr] = u_A[i]*u_B[j]*v_12[k]
                S_T[ctr] = S_AB[i,j]*S_A2[i,k]*S_B2[j,k]
                ctr = ctr+1

    alpha_square = np.linalg.lstsq( (U_T*sigma)**2,S_T)
    return alpha_square[0]

#@jit(nopython=True)
def compute_merge_score(mask1A, mask1B, mask2, similarity_matrix, u_12,sigma_12, v_12, O, merge_method= 'angle'):
    
    mask1 = np.logical_or(mask1A, mask1B)

    # submatrix of similarities betweeb potential subgroups of 1:
    #S_11_AB = similarity_matrix[np.ix_(mask1A, mask1B)]
    S_11_AB = similarity_matrix[mask1A, :]
    S_11_AB = S_11_AB[:, mask1B]

    #submatrix of outer product
    # bool_array is indexer into O, which is size (# nodes in T1) x (# nodes in T1)
    bool_array = mask1A[mask1]
    if np.sum(bool_array) == len(bool_array):
        return np.inf
    O_AB = O[bool_array,:]
    O_AB = O_AB[:,~bool_array]

    # flattern submatrices
    S_11_AB = np.reshape(S_11_AB,(-1,1))        
    O_AB = np.reshape(O_AB,(-1,1))

    if merge_method=='least_square':
        # merge_method 0 is least square alpha        
        alpha = np.linalg.lstsq(O_AB,S_11_AB,rcond=None)
        if len(alpha[1]) == 0:
            score = 0
        else:
            score = alpha[1]/(np.linalg.norm(S_11_AB)**2)
    if merge_method=='normalized_least_square':
        O_AB_n = O_AB / S_11_AB
        S_11_AB_n = np.ones(len(S_11_AB))
        alpha = np.linalg.lstsq(O_AB_n, S_11_AB_n, rcond=None)
        if len(alpha[1]) == 0:
            score = 0
        else:
            score = alpha[1]
    if merge_method=='angle':
        # merge_method 1 is angle between        
        O_AB_n = O_AB/np.linalg.norm(O_AB)
        S_11_AB_n = S_11_AB/np.linalg.norm(S_11_AB)        
        score = 1-np.matmul(S_11_AB_n.T,O_AB_n)
    # if merge_method=='tensor':
    #     # merge method 2 is tensor method
    #     S_11 = S[np.ix_(mask1, mask1)]
    #     S_12 = S[np.ix_(mask1, mask2)]
    #     alpha_square = compute_alpha_tensor(S_11,S_12,u_12,v_12,bool_array,sigma_12)
    #     score = np.linalg.norm(S_11_AB-alpha_square*O_AB)/np.linalg.norm(S_11_AB)
    else:
        Exception("Illigal method: choose least_square, normalized_least_square, angle or tensor")
    return score

def join_trees_with_spectral_root_finding_ls(similarity_matrix, T1, T2, merge_method, taxa_metadata, verbose = False):
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
        for bp in bipartitions1:
            mask1A = taxa_metadata.bipartition2mask(bp)
            mask1B = (T1.mask ^ mask1A)

            score = compute_merge_score(mask1A, mask1B, T2_mask, similarity_matrix, u_12[:,0],sigma_12[0], v_12[0,:], O1, merge_method)
            # DELETE ME bool_array = taxa_metadata.bipartition2mask(bp)
            #DELETE ME score = compute_merge_score(bool_array,S_11,S_12,u_12[:,0],sigma_12[0],v_12[0,:],O1,merge_method)
            
            results['sizeA'].append(mask1A.sum())
            results['sizeB'].append(mask1B.sum())
            results['score'].append(score)
            #results.append([sum(bool_array),sum(~bool_array), score])
            if score <min_score:
                min_score = score
                bp_min = bp
                min_mask1A = mask1A

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



class ReconstructionMethod(ABC):
    @abstractmethod
    def __call__(self, sequences, taxon_namespace=None):
        pass

class RAxML(ReconstructionMethod):
    def __call__(self, sequences, taxa_metadata=None, raxml_args = "-T 2 --JC69 -c 1"):
        if not isinstance(sequences, dendropy.DnaCharacterMatrix):
            # data = FastCharacterMatrix(sequences, taxon_namespace=taxon_namespace).to_dendropy()
            data = utils.array2charmatrix(sequences, taxa_metadata=taxa_metadata) 
            data.taxon_namespace = dendropy.TaxonNamespace(taxa_metadata)
        else:
            data = sequences
            
        if platform.system() == 'Windows':
            # Windows version:
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(RECONSTRUCT_TREE__DIR_PATH, 'raxmlHPC-SSE3.exe'))
        elif platform.system() == 'Darwin':
            #MacOS version:
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(RECONSTRUCT_TREE__DIR_PATH,'raxmlHPC-macOS'))
        elif platform.system() == 'Linux':
            #Linux version
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(RECONSTRUCT_TREE__DIR_PATH,'raxmlHPC-SSE3-linux'))

        tree = rx.estimate_tree(char_matrix=data, raxml_args=[raxml_args])
        tree.is_rooted = False
        if taxa_metadata != None:
            tree.taxon_namespace = taxa_metadata.taxon_namespace
        return tree
    def __repr__(self):
        return "RAxML"

class TreeSVD(ReconstructionMethod):
    
    def __call__(self, observations, taxa_metadata=None):        
        return self.estimate_tree_topology(observations, taxa_metadata)

    #def reconstruct_from_similarity(self, similarity_matrix, taxa_metadata=None):
    #    return self.estimate_tree_topology(similarity_matrix, taxa_metadata)
    #change to spectral_neighbor_joining

    def compute_cnt_matrix(self,data,d):
        m = data.shape[0]
        #comb_arr = np.array(list(product(state_vec, repeat = 4)))
        cnt_mtx = np.zeros(d**m)
        pow_vec = d**(np.arange(m-1,-1,-1))
        for i in data.T:
            loc = np.sum(pow_vec*i)
                #loc = (4**3)*i[0]+(4**2)*i[1]+(4**1)*i[2]+(4**0)*i[3]            
            cnt_mtx[loc] = cnt_mtx[loc]+1
        return cnt_mtx

    def compute_flattening_score(self,cnt_mtx,partition,comb_arr,d):
        # input: 
        num_taxa = self.observations.shape[0]
        row_ind = np.where(np.sum(comb_arr[:,partition],axis=1)==0)[0]
        skip_ind = np.where(np.diff(row_ind)>1)[0]
        if  len(skip_ind)==0:
            skip = d**(num_taxa-len(partition))
        else:
            skip = skip_ind[0]+1
    
        # create flattening matrix
        flattened_mtx = np.zeros((d**len(partition),d**(num_taxa-len(partition))))
        for i in np.arange(d**len(partition)):
            flattened_mtx[i,:] = cnt_mtx[skip*i + row_ind]

        # compute score
        U,sig,V = randomized_svd(flattened_mtx,n_components=d) 
        #score = (np.linalg.norm(sig)**2)/(np.linalg.norm(flattened_mtx)**2)
        score = (np.linalg.norm(flattened_mtx)**2) - (np.linalg.norm(sig)**2)
        return score

    def estimate_tree_topology(self, observations, taxa_metadata=None,bifurcating=False):
        self.observations = observations
        d = len(np.unique(observations))
        m = observations.shape[0]
        
        if taxa_metadata is None:
            taxa_metadata = utils.TaxaMetadata.default(m)
        else:
            assert len(taxa_metadata) >= m, "Namespace too small for distance matrix"
        
        # initialize leaf nodes
        G = taxa_metadata.all_leaves()

        available_clades = set(range(len(G)))   # len(G) == m

        # compute count matrix
        comb_arr = np.array(list(product(np.arange(d), repeat = m)))
        cnt_mtx = self.compute_cnt_matrix(observations,d)
        

        # initialize Sigma
        Score = np.full((2*m,2*m), np.nan)  # we should only use entries that we set later; init to nan so they'll throw an error if we do
        #sv1 = np.full((2*m,2*m), np.nan)
        for i,j in combinations(available_clades, 2):
            Score[i,j] = self.compute_flattening_score(cnt_mtx,[i,j],comb_arr,d)                       
            Score[j,i] = Score[i,j]    # necessary b/c sets have unstable order, so `combinations' could return either one

        # merge
        while len(available_clades) > (2 if bifurcating else 3): # this used to be 1
            left, right = min(combinations(available_clades, 2), key=lambda pair: Score[pair])
            G.append(utils.merge_children((G[left], G[right])))
            new_ix = len(G) - 1
            available_clades.remove(left)
            available_clades.remove(right)
            for other_ix in available_clades:
                partition = np.where(np.logical_or(G[new_ix].mask,G[other_ix].mask))[0]
                Score[other_ix, new_ix] = self.compute_flattening_score(cnt_mtx,partition,comb_arr,d)
                #Sigma[other_ix, new_ix] = scorer(G[other_ix].mask, G[new_ix].mask, similarity_matrix)
                Score[new_ix, other_ix] = Score[other_ix, new_ix]    # necessary b/c sets have unstable order, so `combinations' could return either one
            available_clades.add(new_ix)

        # HEYYYY why does ariel do something special for the last THREE groups? (rather than last two)
        # think about what would happen if at the end we have three groups: 1 leaf, 1 leaf, n-2 leaves
        # how would we know which leaf to attach, since score would be 0 for both??

        # return Phylo.BaseTree.Tree(G[-1])

        # for a bifurcating tree we're combining the last two available clades
        # for an unrooted one it's the last three because
        # if we're making unrooted comparisons it doesn't really matter which order we attach the last three
        return dendropy.Tree(taxon_namespace=taxa_metadata.taxon_namespace, seed_node=utils.merge_children((G[i] for i in available_clades)), is_rooted=False)
    def __repr__(self):
        return "TreeSVD"



class DistanceReconstructionMethod(ReconstructionMethod):
    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric

    def __call__(self, sequences, taxa_metadata=None):
        similarity_matrix = self.similarity_metric(sequences, taxa_metadata)
        return self.reconstruct_from_similarity(similarity_matrix, taxa_metadata)

    @abstractmethod
    def reconstruct_from_similarity(self, similarity_matrix, taxa_metadata=None):
        pass

class NeighborJoining(DistanceReconstructionMethod):
    def reconstruct_from_similarity(self, similarity_matrix, taxa_metadata=None):
        return self.neighbor_joining(similarity_matrix, taxa_metadata)
    
    def neighbor_joining(self, similarity_matrix, taxa_metadata=None):
        similarity_matrix = np.clip(similarity_matrix, a_min=1e-20, a_max=None)
        distance_matrix = -np.log(similarity_matrix)
        T = utils.array2distance_matrix(distance_matrix, taxa_metadata).nj_tree()
        return T
    def __repr__(self):
        return "NJ"
        
class SpectralNeighborJoining(DistanceReconstructionMethod):
    def reconstruct_from_similarity(self, similarity_matrix, taxa_metadata=None):
        return self.estimate_tree_topology(similarity_matrix, taxa_metadata)
    #change to spectral_neighbor_joining

    def estimate_tree_topology(self, similarity_matrix, taxa_metadata=None, scorer=sv2, scaler=1.0, bifurcating=False):
        m, m2 = similarity_matrix.shape
        assert m == m2, "Distance matrix must be square"
        if taxa_metadata is None:
            taxa_metadata = utils.TaxaMetadata.default(m)
        else:
            assert len(taxa_metadata) >= m, "Namespace too small for distance matrix"
        
        # initialize leaf nodes
        G = taxa_metadata.all_leaves()

        available_clades = set(range(len(G)))   # len(G) == m
        # initialize Sigma
        Sigma = np.full((2*m,2*m), np.nan)  # we should only use entries that we set later; init to nan so they'll throw an error if we do
        sv1 = np.full((2*m,2*m), np.nan)
        for i,j in combinations(available_clades, 2):
            Sigma[i,j] = scorer(G[i].mask, G[j].mask, similarity_matrix)
            Sigma[j,i] = Sigma[i,j]    # necessary b/c sets have unstable order, so `combinations' could return either one

        # merge
        while len(available_clades) > (2 if bifurcating else 3): # this used to be 1
            left, right = min(combinations(available_clades, 2), key=lambda pair: Sigma[pair])
            G.append(utils.merge_children((G[left], G[right])))
            new_ix = len(G) - 1
            available_clades.remove(left)
            available_clades.remove(right)
            for other_ix in available_clades:
                Sigma[other_ix, new_ix] = scorer(G[other_ix].mask, G[new_ix].mask, similarity_matrix)
                Sigma[new_ix, other_ix] = Sigma[other_ix, new_ix]    # necessary b/c sets have unstable order, so `combinations' could return either one
            available_clades.add(new_ix)

        # HEYYYY why does ariel do something special for the last THREE groups? (rather than last two)
        # think about what would happen if at the end we have three groups: 1 leaf, 1 leaf, n-2 leaves
        # how would we know which leaf to attach, since score would be 0 for both??

        # return Phylo.BaseTree.Tree(G[-1])

        # for a bifurcating tree we're combining the last two available clades
        # for an unrooted one it's the last three because
        # if we're making unrooted comparisons it doesn't really matter which order we attach the last three
        return dendropy.Tree(taxon_namespace=taxa_metadata.taxon_namespace, seed_node=utils.merge_children((G[i] for i in available_clades)), is_rooted=False)
    def __repr__(self):
        return "SNJ"


class SpectralTreeReconstruction(ReconstructionMethod):
    def __init__(self, inner_method, similarity_metric):
        self.inner_method = inner_method
        self.similarity_metric = similarity_metric
        if issubclass(inner_method, DistanceReconstructionMethod):
            self.reconstruction_alg = inner_method(similarity_metric)
        else:
            self.reconstruction_alg = inner_method()
            
    def __call__(self, sequences, taxa_metadata=None):
        return self.deep_spectral_tree_reconstruction(sequences, self.similarity_metric, taxa_metadata=taxa_metadata, reconstruction_alg = self.inner_method)
    def __repr__(self):
        return "spectralTree" + " + " + self.inner_method.__repr__(self.inner_method)

    def deep_spectral_tree_reconstruction(self, sequences, similarity_metric, taxa_metadata = None, num_gaps =1,threshhold = 100, min_split = 1,merge_method = "angle", verbose = False, **kargs):
        self.verbose = verbose
        self.sequences = sequences
        self.similarity_matrix = similarity_metric(sequences, taxa_metadata = taxa_metadata)
        m, m2 = self.similarity_matrix.shape
        assert m == m2, "Distance matrix must be square"
        self.taxa_metadata = taxa_metadata
        self.taxon_namespace = self.taxa_metadata.taxon_namespace
        if self.taxon_namespace is None:
            self.taxon_namespace = utils.default_namespace(m)
        else:
            assert len(self.taxon_namespace) >= m, "Namespace too small for distance matrix"

        partitioning_tree = MyTree([True]*len(self.taxon_namespace))
        cur_node = partitioning_tree.root
        #process = psutil.Process(os.getpid())
        while True:
            #print(process.memory_info().rss/1024, "KB")  # in bytes    
            if (cur_node.right != None) and (cur_node.right.tree !=None) and (cur_node.left.tree != None):
                cur_node.tree = self.margeTreesLeftRight(cur_node, merge_method)
                cur_node.tree.mask = np.logical_or(cur_node.right.tree.mask, cur_node.left.tree.mask)
                if cur_node.parent == None:
                    break
                if cur_node.parent.right == cur_node:
                    cur_node = cur_node.parent.left
                else:
                    cur_node = cur_node.parent
            elif np.sum(cur_node.bitmap) > threshhold:
                L1,L2 = self.splitTaxa(cur_node,num_gaps,min_split)
                if verbose: print("partition")
                if verbose: print("L1 size: ", np.sum(L1))
                if verbose: print("L2 size: ", np.sum(L2))
                cur_node.setLeft(MyNode(L1))
                cur_node.setRight(MyNode(L2))
                cur_node = cur_node.right
            else:
                start_time = time.time()
                cur_node.tree = self.reconstruct_alg_wrapper(cur_node, **kargs)
                cur_node.tree.mask = taxa_metadata.tree2mask(cur_node.tree)
                runtime = time.time() - start_time
                if verbose: print("--- %s seconds ---" % runtime)
                if cur_node.parent == None:
                    break
                if cur_node.parent.right == cur_node:
                    cur_node = cur_node.parent.left
                else:
                    cur_node = cur_node.parent
        #DELETE MEpartitioning_tree.root.tree.taxon_namespace = self.taxon_namespace
        return partitioning_tree.root.tree
        
    def splitTaxa(self,node,num_gaps,min_split):
        cur_similarity = self.similarity_matrix[node.bitmap,:]
        cur_similarity = cur_similarity[:,node.bitmap]
        laplacian = np.diag(np.sum(cur_similarity, axis = 0)) - cur_similarity
        _, V = np.linalg.eigh(laplacian)
        bool_bipartition = partition_taxa(V[:,1],cur_similarity,num_gaps,min_split)

        """# %%
        if np.minimum(sum(bool_bipartition),sum(~bool_bipartition))<min_split:
            print("????")
            bool_bipartition = partition_taxa(V[:,1],cur_similarity,num_gaps,min_split)
        """

        #Building partitioning bitmaps from partial bitmaps
        ll = np.array([i for i, x in enumerate(node.bitmap) if x])
        ll1 = ll[bool_bipartition]
        not_bool_bipartition = [~i for i in bool_bipartition]
        ll2 = ll[not_bool_bipartition]

        # TODO: use TaxaMetadata.taxa2mask here
        bitmap1 = [True if i in ll1 else False for i in range(len(self.taxon_namespace))]
        bitmap2 = [True if i in ll2 else False for i in range(len(self.taxon_namespace))]
        return bitmap1, bitmap2

    def margeTreesLeftRight(self, node, merge_method):
        # DELETE ME cur_meta = self.taxa_metadata.mask2sub_taxa_metadata(np.array(node.bitmap))

        # cur_similarity = self.similarity_matrix[node.bitmap,:]
        # cur_similarity = cur_similarity[:,node.bitmap]
        return join_trees_with_spectral_root_finding_ls(self.similarity_matrix, node.left.tree, node.right.tree, merge_method, self.taxa_metadata, verbose=self.verbose)
    
    def reconstruct_alg_wrapper(self, node, **kargs):
        # DELETE namespace1 = dendropy.TaxonNamespace([self.taxon_namespace[i] for i in [i for i, x in enumerate(node.bitmap) if x]]) 
        metadata1 = self.taxa_metadata.mask2sub_taxa_metadata(np.array(node.bitmap))
        
        if issubclass(self.inner_method, DistanceReconstructionMethod):
            similarity_matrix1 = self.similarity_matrix[node.bitmap,:]
            similarity_matrix1 = similarity_matrix1[:,node.bitmap]
            return self.reconstruction_alg.reconstruct_from_similarity(similarity_matrix1, taxa_metadata=metadata1)
        else:
            sequences1 = self.sequences[node.bitmap,:]
            return self.reconstruction_alg(sequences1, taxa_metadata=metadata1, **kargs)
    
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
    import dendropy
    ref = utils.balanced_binary(8)
    #all_data = utils.temp_dataset_maker(ref, 1000, 0.01)[0]
    all_data = dendropy.model.discrete.simulate_discrete_chars(1000, ref, dendropy.model.discrete.Jc69(), mutation_rate=0.05)
    observations, _ = utils.charmatrix2array(all_data)
    dist = paralinear_distance(observations)
    dist
    inf = neighbor_joining(dist)
    print('ref:')
    ref.print_plot()
    print("inf:")
    inf.print_plot()

    print([leaf.distance_from_root() for leaf in ref.leaf_nodes()])


    dist = paralinear_distance(observations)
    dist[:5,:5]

    dist = JC_similarity_matrix(observations)
    np.log(dist[:5,:5])

    scipy.spatial.distance.pdist(observations, metric='hamming')

    mm = np.array([[1,2],[-3, -1]])
    np.linalg.norm(mm)**2
