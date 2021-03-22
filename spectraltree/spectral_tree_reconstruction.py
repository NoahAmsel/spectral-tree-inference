import os
from itertools import product
import time

import dendropy
# Having trouble installing numba (https://github.com/numba/llvmlite/issues/527)
# Omitting for now since we aren't currently using it
# from numba import jit
import numpy as np
import scipy.linalg
from sklearn.decomposition import TruncatedSVD

from . import utils
from .reconstruct_tree import ReconstructionMethod, DistanceReconstructionMethod

def correlation_distance_matrix(observations):
    """Correlation Distance"""
    corr = np.abs(np.corrcoef(observations))
    corr = np.clip(corr, a_min=1e-16, a_max=None)
    return -np.log(corr)

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

def compute_score_test(bp):
    return bp


def compute_score_test2(mask1A,T1,T2_mask,similarity_matrix, u_12,sigma_12, v_12, O1, merge_method):      
    mask1B = (T1.mask ^ mask1A)
    score = compute_merge_score(mask1A, mask1B, T2_mask, similarity_matrix, 
        u_12[:,0],sigma_12[0], v_12[0,:], O1, merge_method)
    return score

def compute_score(bp,taxa_metadata,T1,T2_mask,similarity_matrix, u_12,sigma_12, v_12, O1, merge_method):      
    mask1A = taxa_metadata.bipartition2mask(bp)
    mask1B = (T1.mask ^ mask1A)
    score = compute_merge_score(mask1A, mask1B, T2_mask, similarity_matrix, 
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
    bipartitions1 = list(T1.bipartition_edge_map.keys())

    if verbose: print("len(bipartitions1)", len(bipartitions1))

    if len(bipartitions1) ==2:
        print("NOOOOO")
    if len(bipartitions1) > 1:
        min_score = float("inf")
        results = {'sizeA': [], 'sizeB': [], 'score': []}
        
        time_s = time.time()
        #pool = mp.Pool(4)
        #score_list = pool.map(compute_score,product(bipartitions1,[taxa_metadata],[T1],[T2_mask],
        #    [similarity_matrix], [u_12],[sigma_12], [v_12], [O1], [merge_method]))
        #parameters = product(bipartitions1,[taxa_metadata],[T1],[T2_mask],
        #    [similarity_matrix], [u_12],[sigma_12], [v_12], [O1], [merge_method]
        
        # score_list = Parallel(n_jobs=4)(delayed(compute_score)(bp,taxa_metadata,T1,T2_mask,
        #     similarity_matrix, u_12,sigma_12, v_12, O1, merge_method) for bp in bipartitions1)
        bp_mask = [taxa_metadata.bipartition2mask(bp) for bp in bipartitions1]
        score_list = Parallel(n_jobs=4)(delayed(compute_score_test2)(bp_mask[i], T1, T2_mask, similarity_matrix, u_12, sigma_12, v_12, O1, merge_method) for i in range(len(bp_mask)))
        
        #score_list = [pool.apply(compute_score, args=(bp,taxa_metadata,T1,T2_mask,
        #    similarity_matrix, u_12,sigma_12, v_12, O1, merge_method)) for bp in bipartitions1]
        #score_list = pool.map(compute_score(), [bp for bp in bipartitions1])        
        #pool.close()
        min_score_par = np.min(score_list)
        min_idx_par = np.argmin(score_list)
        runtime_par = time.time()-time_s

        time_s = time.time()
        for bp in bipartitions1:
            mask1A = taxa_metadata.bipartition2mask(bp)
            mask1B = (T1.mask ^ mask1A)

            score = compute_merge_score(mask1A, mask1B, T2_mask, similarity_matrix, u_12[:,0],sigma_12[0], v_12[0,:], O1, merge_method)
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

class STDR(ReconstructionMethod):
    def __init__(self, inner_method, similarity_metric):
        self.inner_method = inner_method
        self.similarity_metric = similarity_metric
        if issubclass(inner_method, DistanceReconstructionMethod):
            self.reconstruction_alg = inner_method(similarity_metric)
        else:
            self.reconstruction_alg = inner_method()
            
    def __call__(self, sequences, taxa_metadata=None, num_gaps =1,threshold = 100,min_split = 1,merge_method = "angle",verbose = False):
        return self.deep_spectral_tree_reconstruction(sequences, self.similarity_metric, taxa_metadata=taxa_metadata, 
             num_gaps = num_gaps,threshold = threshold, merge_method = merge_method, min_split = min_split, verbose = verbose)
    def __repr__(self):
        return "STDR" + " + " + self.inner_method.__repr__(self.inner_method)

    def deep_spectral_tree_reconstruction(self, sequences, similarity_metricx, taxa_metadata = None, num_gaps =1,threshold = 100, 
        alpha = 1,min_split = 1,merge_method = "angle", verbose = False, **kargs):
        self.verbose = verbose
        self.sequences = sequences
        if callable(similarity_metricx):
            self.similarity_matrix = similarity_metricx(sequences)**alpha
        else:
            self.similarity_matrix =similarity_metricx
        m, m2 = self.similarity_matrix.shape
        assert m == m2, "Distance matrix must be square"
        self.taxa_metadata = taxa_metadata
        self.taxon_namespace = self.taxa_metadata.taxon_namespace
        if self.taxon_namespace is None:
            self.taxon_namespace = utils.default_namespace(m)
        else:
            assert len(self.taxa_metadata) >= m, "Namespace too small for distance matrix"

        partitioning_tree = MyTree([True]*len(self.taxa_metadata))
        cur_node = partitioning_tree.root
        if self.verbose: process = psutil.Process(os.getpid())
        while True:
            if self.verbose: print(process.memory_info().rss/1024, "KB")  # in bytes    
            if (cur_node.right != None) and (cur_node.right.tree !=None) and (cur_node.left.tree != None):
                cur_node.tree = self.margeTreesLeftRight(cur_node, merge_method)
                cur_node.tree.mask = np.logical_or(cur_node.right.tree.mask, cur_node.left.tree.mask)
                if cur_node.parent == None:
                    break
                if cur_node.parent.right == cur_node:
                    cur_node = cur_node.parent.left
                else:
                    cur_node = cur_node.parent
            elif np.sum(cur_node.bitmap) > threshold:
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
        # t= time.time()
        e,V = scipy.linalg.eigh(laplacian, eigvals = (0,1))
        # tt = time.time() - t
        # t= time.time()
        # _, V = np.linalg.eigh(laplacian)
        # tt = time.time() - t
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
        bitmap1 = [True if i in ll1 else False for i in range(len(self.taxa_metadata))]
        bitmap2 = [True if i in ll2 else False for i in range(len(self.taxa_metadata))]
        return bitmap1, bitmap2

    def margeTreesLeftRight(self, node, merge_method):
        # DELETE ME cur_meta = self.taxa_metadata.mask2sub_taxa_metadata(np.array(node.bitmap))

        # cur_similarity = self.similarity_matrix[node.bitmap,:]
        # cur_similarity = cur_similarity[:,node.bitmap]
        return join_trees_with_spectral_root_finding_ls(self.similarity_matrix, node.left.tree, node.right.tree, merge_method, self.taxa_metadata, verbose=self.verbose)
        #return join_trees_with_spectral_root_finding_par(self.similarity_matrix, node.left.tree, node.right.tree, merge_method, self.taxa_metadata, verbose=self.verbose)
    
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