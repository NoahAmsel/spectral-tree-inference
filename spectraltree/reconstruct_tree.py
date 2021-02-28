import platform
from abc import ABC, abstractmethod
from functools import partial
from itertools import combinations
import os, sys
import numpy as np
import scipy.spatial.distance
import scipy.linalg
from sklearn.decomposition import TruncatedSVD
from itertools import product
from itertools import combinations
import dendropy     #should this library be independent of dendropy? is that even possible?
from dendropy.interop import raxml
import subprocess
from . import utils
import time
import os
import psutil
from numba import jit
from sklearn.utils.extmath import randomized_svd
import multiprocessing as mp
from joblib import Parallel, delayed
num_cores = mp.cpu_count()
import scipy

def sv2(A1, A2, M):
    """Second Singular Value"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        return s[1] # the second eigenvalue

def max_quartet(A1,A2,M):
    "maximal quartet criterion"
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]
    m_1,m_2 = M_A.shape
    max_w = 0
    for x in combinations(range(m_1), 2):
        for y in combinations(range(m_2),2):
            w = np.abs(M_A[x[0],y[0]]*M_A[x[1],y[1]]-M_A[x[0],y[1]]*M_A[x[1],y[0]])
            if w>max_w:
                max_w = w
    return max_w
    #for (i,j) in  

def sum_sigular_values(A1, A2, M):
    """Normalized Sum of Squared Quartets"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    return np.linalg.norm(s[1:])

def sum_squared_quartets(A1, A2, M):
    """Normalized Sum of Squared Quartets"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    norm_sq = np.linalg.norm(M_A)**2
    num = norm_sq**2 - np.linalg.norm(M_A.T.dot(M_A))**2
    return num / norm_sq

def average_quartets(A1, A2, M):
    """Normalized Sum of Squared Quartets"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    #norm_sq = np.linalg.norm(M_A)**2
    m_1 = np.sum(A1)+np.sum(A2)
    m_2 = len(A1)-m_1
    norm_sq = m_1*(m_1-1)*m_2*(m_2-1)
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

def JC_similarity_matrix(observations, taxa_metadata=None,params=None):
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
    #return inside_log**(3)

def gamma_func(P,a):
    return (3/4)*a*( (1-(4/3)*P)**(-1/a)-1 )

def JC_gamma_similarity_matrix(observations, taxa_metadata=None,params = None):
    """Jukes-Cantor Corrected Similarity"""
    classes = np.unique(observations)
    if classes.dtype == np.dtype('<U1'):
        # needed to use hamming distance with string arrays
        vord = np.vectorize(ord)
        observations = vord(observations)
    k = len(classes)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    #inside_log = 1 - hamming_matrix*k/(k-1)
    a = params.alpha
    return np.exp(gamma_func(hamming_matrix,a))

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

    def __call__(self, sequences, taxa_metadata=None,params = None):
        # params are other needed parameters. i.e. params.spectral_criterion = 'sv2', params.alpha = 1
        similarity_matrix = self.similarity_metric(sequences, taxa_metadata,params)
        return self.reconstruct_from_similarity(similarity_matrix, taxa_metadata, params=params)

    @abstractmethod
    def reconstruct_from_similarity(self, similarity_matrix, taxa_metadata=None, params = None):
        pass

class NeighborJoining(DistanceReconstructionMethod):
    def reconstruct_from_similarity(self, similarity_matrix, taxa_metadata=None, params = None):
        return self.neighbor_joining(similarity_matrix, taxa_metadata)
    
    def neighbor_joining(self, similarity_matrix, taxa_metadata=None):
        similarity_matrix = np.clip(similarity_matrix, a_min=1e-20, a_max=None)
        distance_matrix = -np.log(similarity_matrix)
        T = utils.array2distance_matrix(distance_matrix, taxa_metadata).nj_tree()
        return T
    def __repr__(self):
        return "NJ"
        
class SpectralNeighborJoining(DistanceReconstructionMethod):    
    def reconstruct_from_similarity(self, similarity_matrix, taxa_metadata=None, params = None):
        return self.estimate_tree_topology(similarity_matrix, taxa_metadata, params = params)
    #change to spectral_neighbor_joining

    def estimate_tree_topology(self, similarity_matrix, taxa_metadata=None, scorer=sv2, scaler=1.0, bifurcating=False, params = None):
        if hasattr(params,'alpha'):
            similarity_matrix = similarity_matrix**params.alpha
        m, m2 = similarity_matrix.shape
        assert m == m2, "Distance matrix must be square"
        if taxa_metadata is None:
            taxa_metadata = utils.TaxaMetadata.default(m)
        else:
            assert len(taxa_metadata) >= m, "Namespace too small for distance matrix"
        
        # initialize leaf nodes
        G = taxa_metadata.all_leaves()

        available_clades = set(range(len(G)))   # len(G) == m
        
        # choose spectral score function
        if hasattr(params,'score_func'):
            scorer = params.score_func
        else:
            scorer = sv2

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

# %%
if __name__ == "__main__":
    import dendropy
    ref = utils.balanced_binary(8)
    #all_data = utils.temp_dataset_maker(ref, 1000, 0.01)[0]
    all_data = dendropy.model.discrete.simulate_discrete_chars(1000, ref, dendropy.model.discrete.Jc69(), mutation_rate=0.05)
    observations, taxon_meta = utils.charmatrix2array(all_data)
    
    dist = paralinear_distance(observations)
    dist2 = raxml_gamma_corrected_distance_matrix(observations, taxon_meta)
    
    dist
    inf = NeighborJoining(lambda x: x).reconstruct_from_similarity(np.exp(-dist), taxon_meta)
    dist2
    inf2 = NeighborJoining(lambda x: x).reconstruct_from_similarity(np.exp(-dist2), taxon_meta)
    print('ref:')
    ref.print_plot()
    print("inf:")
    inf.print_plot()
    inf2.print_plot()

    print([leaf.distance_from_root() for leaf in ref.leaf_nodes()])


    dist = paralinear_distance(observations)
    dist[:5,:5]

    dist = JC_similarity_matrix(observations)
    np.log(dist[:5,:5])

    scipy.spatial.distance.pdist(observations, metric='hamming')

    mm = np.array([[1,2],[-3, -1]])
    np.linalg.norm(mm)**2

