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
from character_matrix import FastCharacterMatrix
import time

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

def paralinear_similarity(observations, classes=None):
    # build the similarity matrix M -- this is pretty optimized, at least assuming k << n
    # setting classes allows you to ignore missing data. e.g. set class to 1,2,3,4 and it will
    # treat np.nan, -1, and 5 as missing data
    if classes is None:
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

def paralinear_distance(observations, classes=None):
    similarity = paralinear_similarity(observations, classes)
    similarity = np.clip(similarity, a_min=1e-20, a_max=None)
    return -np.log(similarity)

def JC_similarity_matrix(observations, classes=None):
    """Jukes-Cantor Corrected Similarity"""
    assert classes is None
    if classes is None:
        classes = np.unique(observations)
    if classes.dtype == np.dtype('<U1'):
        vord = np.vectorize(ord)
        observations = vord(observations)
    k = len(classes)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    inside_log = 1 - hamming_matrix*k/(k-1)
    return inside_log**(k-1)

def JC_distance_matrix(observations, classes=None):
    """Jukes-Cantor Corrected Distance"""
    inside_log = JC_similarity_matrix(observations,classes)
    inside_log = np.clip(inside_log, a_min=1e-16, a_max=None)
    return - np.log(inside_log)

def estimate_edge_lengths(tree, distance_matrix):
    # check the PAUP* documentation
    pass


def hamming_dist_missing_values(vals, missing_val = "-"):
    classnames, indices = np.unique(vals, return_inverse=True)
    num_arr = indices.reshape(vals.shape)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(num_arr, metric='hamming'))
    missing_array = (vals==missing_val)
    pdist_xor = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_xor(u,v))))
    pdist_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_or(u,v))))
    
    return (hamming_matrix*vals.shape[1] - pdist_xor) / (np.ones_like(hamming_matrix) * vals.shape[1] - pdist_or)


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



##########################################################
##               Reconstruction methods
##########################################################

def correlation_distance_matrix(observations):
    """Correlation Distance"""
    corr = np.abs(np.corrcoef(observations))
    corr = np.clip(corr, a_min=1e-16, a_max=None)
    return -np.log(corr)

def estimate_tree_topology(similarity_matrix, taxon_namespace=None, scorer=sv2, scaler=1.0, bifurcating=False):
    m, m2 = similarity_matrix.shape
    assert m == m2, "Distance matrix must be square"
    if taxon_namespace is None:
        taxon_namespace = utils.default_namespace(m)
    else:
        assert len(taxon_namespace) >= m, "Namespace too small for distance matrix"
    
    # initialize leaf nodes
    G = [utils.leaf(i, taxon_namespace) for i in range(m)]

    available_clades = set(range(len(G)))   # len(G) == m
    # initialize Sigma
    Sigma = np.full((2*m,2*m), np.nan)  # we should only use entries that we set later; init to nan so they'll throw an error if we do
    sv1 = np.full((2*m,2*m), np.nan)
    for i,j in combinations(available_clades, 2):
        Sigma[i,j] = scorer(G[i].taxa_set, G[j].taxa_set, similarity_matrix)
        Sigma[j,i] = Sigma[i,j]    # necessary b/c sets have unstable order, so `combinations' could return either one

    # merge
    while len(available_clades) > (2 if bifurcating else 3): # this used to be 1
        left, right = min(combinations(available_clades, 2), key=lambda pair: Sigma[pair])
        G.append(utils.merge_children((G[left], G[right])))
        new_ix = len(G) - 1
        available_clades.remove(left)
        available_clades.remove(right)
        for other_ix in available_clades:
            Sigma[other_ix, new_ix] = scorer(G[other_ix].taxa_set, G[new_ix].taxa_set, similarity_matrix)
            Sigma[new_ix, other_ix] = Sigma[other_ix, new_ix]    # necessary b/c sets have unstable order, so `combinations' could return either one
        available_clades.add(new_ix)

    # HEYYYY why does ariel do something special for the last THREE groups? (rather than last two)
    # think about what would happen if at the end we have three groups: 1 leaf, 1 leaf, n-2 leaves
    # how would we know which leaf to attach, since score would be 0 for both??

    # return Phylo.BaseTree.Tree(G[-1])

    # for a bifurcating tree we're combining the last two available clades
    # for an unrooted one it's the last three because
    # if we're making unrooted comparisons it doesn't really matter which order we attach the last three
    return dendropy.Tree(taxon_namespace=namespace, seed_node=utils.merge_children((G[i] for i in available_clades)), is_rooted=False)

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
                
def compute_merge_score(bool_array,S_11,S_12,u_12,sigma_12,v_12,O,merge_method= 'least_square'):

    # submatrix of similarities betweeb potential subgroups of 1:
    S_11_AB = S_11[bool_array,:]
    S_11_AB = S_11_AB[:,~bool_array]

    #submatrix of outer product
    O_AB = O[bool_array,:]
    O_AB = O_AB[:,~bool_array]

    # flattern submatrices
    S_11_AB = np.reshape(S_11_AB,(-1,1))        
    O_AB = np.reshape(O_AB,(-1,1))

    if merge_method=='least_square':
        # merge_method 0 is least square alpha        
        alpha = np.linalg.lstsq(O_AB,S_11_AB,rcond=None)
        score = alpha[1]/(np.linalg.norm(S_11_AB)**2)
    if merge_method=='angle':
        # merge_method 1 is angle between        
        O_AB_n = O_AB/np.linalg.norm(O_AB)
        S_11_AB_n = S_11_AB/np.linalg.norm(S_11_AB)        
        score = 1-np.matmul(S_11_AB_n.T,O_AB_n)
    if merge_method=='tensor':
        # merge method 2 is tensor method
        alpha_square = compute_alpha_tensor(S_11,S_12,u_12,v_12,bool_array,sigma_12)
        score = np.linalg.norm(S_11_AB-alpha_square*O_AB)/np.linalg.norm(S_11_AB)
    else:
        Exception("Illigal method: choose least_square, angle or tensor")
    return score

def join_trees_with_spectral_root_finding_ls(similarity_matrix, T1, T2, merge_method,taxon_namespace=None, verbose = False):
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
    T1.is_rooted = True
    
    half2_idx_bool = [x.label in T2_labels for x in taxon_namespace]
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
    [u_12,sigma_12,v_12] = np.linalg.svd(S_12)
    O = np.outer(u_12[:,0],u_12[:,0])
    
    # find root of half 1
    bipartitions1 = T1.bipartition_edge_map
    min_score = float("inf")
    results = []
    results = {'sizeA': [], 'sizeB': [], 'score': []}
    if verbose: print("len(bipartitions1)", len(bipartitions1))
    if len(bipartitions1.keys()) ==2:
        print("NOOOOO") 
    for bp in bipartitions1.keys():
        bool_array = np.array(list(map(bool,[int(i) for i in bp.leafset_as_bitstring()]))[::-1])

        score = compute_merge_score(bool_array,S_11,S_12,u_12[:,0],sigma_12[0],v_12[0,:],O,merge_method)
        
        results['sizeA'].append(np.sum(bool_array))
        results['sizeB'].append(np.sum(~bool_array))
        results['score'].append(score)
        #results.append([sum(bool_array),sum(~bool_array), score])
        if score <min_score:
            min_score = score
            bp_min = bp

    bool_array = np.array(list(map(bool,[int(i) for i in bp_min.leafset_as_bitstring()]))[::-1])
    if verbose: 
        if np.sum(bool_array)==1:
            print('one')
    if verbose: print("one - merging: ",np.sum(bool_array), " out of: ", len(bool_array))
    if len(bipartitions1.keys()) > 1: 
        T1.reroot_at_edge(bipartitions1[bp_min])

    # find root of half 2
    #[u_12,s,v_12] = np.linalg.svd(S_12.T)
    O = np.outer(v_12[0,:],v_12[0,:])
    #O = np.outer(v_12[:,0],v_12[:,0])
    bipartitions2 = T2.bipartition_edge_map
    min_score = float("inf")
    results2 = []
    if verbose: print("len(bipartitions2)", len(bipartitions2))
    if len(bipartitions2.keys()) ==2:
        print("NOOOOO")
    for bp in bipartitions2.keys():
        bool_array = np.array(list(map(bool,[int(i) for i in bp.leafset_as_bitstring()]))[::-1])
        #h2_idx_A = half2_idx_array[bool_array]
        #h2_idx_B = half2_idx_array[~bool_array]
        
        score = compute_merge_score(bool_array,S_22,S_12.T,v_12[0,:],sigma_12[0],u_12[:,0],O,merge_method)
        
        results2.append([np.sum(bool_array),np.sum(~bool_array), score])
        if score <min_score:
            min_score = score
            bp_min = bp

    bool_array = np.array(list(map(bool,[int(i) for i in bp_min.leafset_as_bitstring()]))[::-1])
    if verbose:
        if np.sum(bool_array)==1:
            print('one')
    if verbose: print("one - merging: ",np.sum(bool_array), " out of: ", len(bool_array))
    if len(bipartitions2.keys()) > 1: 
        T2.reroot_at_edge(bipartitions2[bp_min])

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
    def __call__(self, sequences, taxon_namespace=None, raxml_args = "-T 2 --JC69 -c 1"):
        if not isinstance(sequences, dendropy.DnaCharacterMatrix):
            data = FastCharacterMatrix(sequences, taxon_namespace=taxon_namespace).to_dendropy()
        else:
            data = sequences     
        if platform.system() == 'Windows':
            # Windows version:
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(RECONSTRUCT_TREE__DIR_PATH, 'raxmlHPC-SSE3.exe'))
        elif platform.system() == 'Darwin':
            #MacOS version:
            rx = raxml.RaxmlRunner()
        elif platform.system() == 'Linux':
            #Linux version
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(RECONSTRUCT_TREE__DIR_PATH,'raxmlHPC-SSE3-linux'))

        tree = rx.estimate_tree(char_matrix=data, raxml_args=[raxml_args])
        tree.is_rooted = False
        return tree
    def __repr__():
        return "RAxML"

class DistanceReconstructionMethod(ReconstructionMethod):
    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric

    def __call__(self, sequences, taxon_namespace=None):
        if isinstance(sequences, FastCharacterMatrix):
            sequences = sequences.to_array()
        similarity_matrix = self.similarity_metric(sequences)
        return self.reconstruct_from_similarity(similarity_matrix, taxon_namespace)

    @abstractmethod
    def reconstruct_from_similarity(self, similarity_matrix, taxon_namespace=None):
        pass

class NeighborJoining(DistanceReconstructionMethod):
    def reconstruct_from_similarity(self, similarity_matrix, taxon_namespace=None):
        return self.neighbor_joining(similarity_matrix, taxon_namespace)
    
    def neighbor_joining(self, similarity_matrix, taxon_namespace=None):
        similarity_matrix = np.clip(similarity_matrix, a_min=1e-20, a_max=None)
        distance_matrix = -np.log(similarity_matrix)
        T = utils.array2distance_matrix(distance_matrix, taxon_namespace).nj_tree()
        return T
    def __repr__():
        return "NJ"
        
class SpectralNeighborJoining(DistanceReconstructionMethod):
    def reconstruct_from_similarity(self, similarity_matrix, taxon_namespace=None):
        return self.estimate_tree_topology(similarity_matrix, taxon_namespace)
    #change to spectral_neighbor_joining

    def estimate_tree_topology(self, similarity_matrix, taxon_namespace=None, scorer=sv2, scaler=1.0, bifurcating=False):
        m, m2 = similarity_matrix.shape
        assert m == m2, "Distance matrix must be square"
        if taxon_namespace is None:
            taxon_namespace = utils.default_namespace(m)
        else:
            assert len(taxon_namespace) >= m, "Namespace too small for distance matrix"
        
        # initialize leaf nodes
        G = [utils.leaf(i, taxon_namespace) for i in range(m)]

        available_clades = set(range(len(G)))   # len(G) == m
        # initialize Sigma
        Sigma = np.full((2*m,2*m), np.nan)  # we should only use entries that we set later; init to nan so they'll throw an error if we do
        sv1 = np.full((2*m,2*m), np.nan)
        for i,j in combinations(available_clades, 2):
            Sigma[i,j] = scorer(G[i].taxa_set, G[j].taxa_set, similarity_matrix)
            Sigma[j,i] = Sigma[i,j]    # necessary b/c sets have unstable order, so `combinations' could return either one

        # merge
        while len(available_clades) > (2 if bifurcating else 3): # this used to be 1
            left, right = min(combinations(available_clades, 2), key=lambda pair: Sigma[pair])
            G.append(utils.merge_children((G[left], G[right])))
            new_ix = len(G) - 1
            available_clades.remove(left)
            available_clades.remove(right)
            for other_ix in available_clades:
                Sigma[other_ix, new_ix] = scorer(G[other_ix].taxa_set, G[new_ix].taxa_set, similarity_matrix)
                Sigma[new_ix, other_ix] = Sigma[other_ix, new_ix]    # necessary b/c sets have unstable order, so `combinations' could return either one
            available_clades.add(new_ix)

        # HEYYYY why does ariel do something special for the last THREE groups? (rather than last two)
        # think about what would happen if at the end we have three groups: 1 leaf, 1 leaf, n-2 leaves
        # how would we know which leaf to attach, since score would be 0 for both??

        # return Phylo.BaseTree.Tree(G[-1])

        # for a bifurcating tree we're combining the last two available clades
        # for an unrooted one it's the last three because
        # if we're making unrooted comparisons it doesn't really matter which order we attach the last three
        return dendropy.Tree(taxon_namespace=taxon_namespace, seed_node=utils.merge_children((G[i] for i in available_clades)), is_rooted=False)
def __repr__():
        return "SNJ"

class SpectralTreeReconstruction(ReconstructionMethod):
    def __init__(self, inner_method, similarity_metric):
        self.inner_method = inner_method
        self.similarity_metric = similarity_metric
        if issubclass(inner_method, DistanceReconstructionMethod):
            self.reconstruction_alg = inner_method(similarity_metric)
        else:
            self.reconstruction_alg = inner_method()
            
    def __call__(self, sequences, taxon_namespace=None):
        return self.deep_spectral_tree_reconstruction(sequences, self.similarity_metric, taxon_namespace=taxon_namespace, reconstruction_alg = self.inner_method)
    def __repr__():
        return "spectralTree"

    def deep_spectral_tree_reconstruction(self, sequences, similarity_metric,taxon_namespace = None, num_gaps =1,threshhold = 100, min_split = 1,merge_method = "angle", verbose = False, **kargs):
        self.verbose = verbose
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
                cur_node.tree = self.margeTreesLeftRight(cur_node,merge_method)
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
                runtime = time.time() - start_time
                if verbose: print("--- %s seconds ---" % runtime)
                if cur_node.parent == None:
                    break
                if cur_node.parent.right == cur_node:
                    cur_node = cur_node.parent.left
                else:
                    cur_node = cur_node.parent
        partitioning_tree.root.tree.taxon_namespace = self.taxon_namespace
        return partitioning_tree.root.tree
        
    def splitTaxa(self,node,num_gaps,min_split):
        cur_similarity = self.similarity_matrix[node.bitmap,:]
        cur_similarity = cur_similarity[:,node.bitmap]
        laplacian = np.diag(np.sum(cur_similarity, axis = 0)) - cur_similarity
        _, V = np.linalg.eigh(laplacian)
        bool_bipartition = partition_taxa(V[:,1],cur_similarity,num_gaps,min_split)
        # %%
        if np.minimum(np.sum(bool_bipartition),np.sum(~bool_bipartition))<min_split:
            print("????")
            bool_bipartition = partition_taxa(V[:,1],cur_similarity,num_gaps,min_split)

        #Building partitioning bitmaps from partial bitmaps
        ll = np.array([i for i, x in enumerate(node.bitmap) if x])
        ll1 = ll[bool_bipartition]
        not_bool_bipartition = [~i for i in bool_bipartition]
        ll2 = ll[not_bool_bipartition]
        bitmap1 = [True if i in ll1 else False for i in range(len(self.taxon_namespace))]
        bitmap2 = [True if i in ll2 else False for i in range(len(self.taxon_namespace))]
        return bitmap1, bitmap2

    def margeTreesLeftRight(self, node,merge_method):
        cur_namespace = dendropy.TaxonNamespace([self.taxon_namespace[i] for i in [i for i, x in enumerate(node.bitmap) if x]])  
        cur_similarity = self.similarity_matrix[node.bitmap,:]
        cur_similarity = cur_similarity[:,node.bitmap]
        return join_trees_with_spectral_root_finding_ls(cur_similarity, node.left.tree, node.right.tree, merge_method,taxon_namespace=cur_namespace, verbose=self.verbose)
    
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


class Reconstruction_Method:
    def __init__(self, core=estimate_tree_topology, similarity=paralinear_similarity, **kwargs):
        self.core = core
        self.similarity = similarity
        if self.core == estimate_tree_topology:
            kwargs["scorer"] = kwargs.get("scorer", sv2)
            kwargs["scaler"] = kwargs.get("scaler", 1.0)
        self.kwargs = kwargs

    @property
    def scorer(self):
        return self.kwargs.get('scorer', None)

    @property
    def scaler(self):
        return self.kwargs.get('scaler', None)

    def __call__(self, observations, namespace=None):
        T = self.reconstruct_from_array(observations, namespace)
        #T.print_plot()
        return T

    def reconstruct_from_array(self, observations, namespace=None):
        similarity_matrix = self.similarity(observations)
        tree = self.core(similarity_matrix, namespace=namespace, **self.kwargs)
        return tree

    def reconstruct_from_charmatrix(self, char_matrix, namespace=None):
        """
        takes dendropy.datamodel.charmatrixmodel.CharacterMatrix
        """
        observations, alphabet = utils.charmatrix2array(char_matrix)
        return self(observations)

    def __str__(self):
        if self.core == estimate_tree_topology:
            s = self.scorer.__name__
            if self.scaler != 1.0:
                s += " x{:.2}".format(self.scaler)
        elif self.core == neighbor_joining:
            s = "NJ"
        else:
            s = self.core.__name__
        return s

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
