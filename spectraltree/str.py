import numpy as np
import dendropy
from sklearn.decomposition import TruncatedSVD

from .reconstruct_tree import ReconstructionMethod, DistanceReconstructionMethod

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

class STR(ReconstructionMethod):
    def __init__(self, inner_method, similarity_metric, threshold, merge_method, num_gaps = 1, min_split = 1, verbose=False):
        self.inner_method = inner_method
        self.similarity_metric = similarity_metric
        if issubclass(inner_method, DistanceReconstructionMethod):
            self.reconstruction_alg = inner_method(similarity_metric)
        else:
            self.reconstruction_alg = inner_method()

        self.threshold = threshold
        self.merge_method = merge_method
        self.num_gaps = num_gaps
        self.min_split = min_split
        self.verbose = verbose

    def __repr__():
        return "spectralTreeReconstruction"

    def __call__(self, sequences, taxa_metadata=None):
        num_taxa = sequences.shape[0]
        similarity_matrix = self.similarity_metric(sequences)

        return self.__reconstruction_helper(np.ones(num_taxa, dtype=bool), similarity_matrix, sequences, taxa_metadata)

    def __reconstruction_helper(self, parent_mask, similarity, sequences, metadata):
        if parent_mask.sum() <= self.threshold:
            subtree = self.__reconstruct_alg_wrapper(parent_mask, similarity, sequences, metadata)
            subtree.mask = parent_mask
            return subtree
        else:
            similarity_submat = similarity[np.ix_(parent_mask, parent_mask)]
            submeta = metadata.mask2sub_taxa_metadata(parent_mask)
            laplacian = np.diag(np.sum(similarity_submat, axis = 0)) - similarity_submat
            _, V = np.linalg.eigh(laplacian)

            sub_bipartition_mask = partition_taxa(V[:,1], similarity_submat, self.num_gaps, self.min_split)

            left_sub_meta = submeta.mask2sub_taxa_metadata(sub_bipartition_mask)
            right_sub_meta = submeta.mask2sub_taxa_metadata(~sub_bipartition_mask)

            left_mask = metadata.taxa2mask(left_sub_meta)
            right_mask = metadata.taxa2mask(right_sub_meta)

            left_child = self.__reconstruction_helper(left_mask, similarity, sequences, metadata)
            right_child = self.__reconstruction_helper(right_mask, similarity, sequences, metadata)

            merged = self.__merge_subtrees(left_child, right_child, similarity, metadata)
            merged.mask = np.logical_or(left_child.mask, right_child.mask)
            return merged

    def __merge_subtrees(self, left, right, similarity_matrix, taxa_metadata):
        # make sure this is necessary
        left.is_rooted = True
        right.is_rooted = True

        # TODO: use truncated svd here
        # u_12 is matrix
        S_12 = similarity_matrix[np.ix_(left.mask, right.mask)]
        [u_12,sigma_12,v_12] = np.linalg.svd(S_12)
        left.u_12 = u_12
        right.u_12 = v_12

        left.O = np.outer(u_12[:,0],u_12[:,0])
        right.O = np.outer(v_12[0,:],v_12[0,:])               

        for T1, T2 in [(left, right), (right, left)]:
            bipartitions1 = T1.bipartition_edge_map.keys()

            if len(bipartitions1) > 1:
                min_score = np.inf

                for bp in bipartitions1:
                    mask1A = taxa_metadata.bipartition2mask(bp)
                    mask1B = (T1.mask ^ mask1A)                
            
                    score = compute_merge_score(mask1A, mask1B, T2.mask, similarity_matrix, T1.u_12[:,0], sigma_12[0], T2.u_12[0,:], T1.O, self.merge_method)
                    if score < min_score:
                        min_score = score
                        bp_min = bp

                T1.reroot_at_edge(T1.bipartition_edge_map[bp_min])

        T = dendropy.Tree(taxon_namespace=taxa_metadata.taxon_namespace)
        T.seed_node.set_child_nodes([T1.seed_node, T2.seed_node])
        return T        

    def __reconstruct_alg_wrapper(self, mask, similarity, sequences, metadata):
        submeta = metadata.mask2sub_taxa_metadata(mask)
        if issubclass(self.inner_method, DistanceReconstructionMethod):
            subsimilarity = similarity[np.ix_(mask, mask)]
            return self.reconstruction_alg.reconstruct_from_similarity(subsimilarity, taxa_metadata=submeta)
        else:
            subseq = sequences[mask, :]
            return self.reconstruction_alg(subseq, taxa_metadata=submeta)


def partition_taxa(v, similarity, num_gaps, min_split):
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


def compute_merge_score(mask1A, mask1B, mask2, similarity_matrix, u_12,sigma_12, v_12, O, merge_method= 'angle'):
    
    mask1 = np.logical_or(mask1A, mask1B)

    # submatrix of similarities betweeb potential subgroups of 1:
    S_11_AB = similarity_matrix[np.ix_(mask1A, mask1B)]

    #submatrix of outer product
    # bool_array is indexer into O, which is size (# nodes in T1) x (# nodes in T1)
    bool_array = mask1A[mask1]
    if sum(bool_array) == len(bool_array):
        return float("inf")
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
    if merge_method=='tensor':
        # merge method 2 is tensor method
        S_11 = S[np.ix_(mask1, mask1)]
        S_12 = S[np.ix_(mask1, mask2)]
        alpha_square = compute_alpha_tensor(S_11,S_12,u_12,v_12,bool_array,sigma_12)
        score = np.linalg.norm(S_11_AB-alpha_square*O_AB)/np.linalg.norm(S_11_AB)
    else:
        Exception("Illigal method: choose least_square, normalized_least_square, angle or tensor")
    return score