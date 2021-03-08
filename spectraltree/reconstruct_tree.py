from abc import ABC, abstractmethod
from itertools import product, combinations

import dendropy
import numpy as np
from sklearn.utils.extmath import randomized_svd

from . import utils

##########################################################
##               Reconstruction methods
##########################################################

class ReconstructionMethod(ABC):
    """
    Method for reconstructing a binary tree from observations of the leaf nodes.
    """

    @abstractmethod
    def __call__(self, sequences, taxon_metadata=None):
        """Reconstruct a binary tree from sequence data.

        Args:
            sequences: An m x n matrix, where m is the number of leaf nodes (observed variables) and n 
                is the number of samples. Entries should be integers, with each integer corresponding to
                some state of the random variable. For example, 0,1,2, and 3 could represent the characters
                A, C, G, and T of a DNA sequence.
            taxon_metadata: A TaxaMetadata object describing the observed nodes. The first element should
                correspond to the first row of `sequences`, and so on. If omitted, the default TaxonMetadata
                of size m is used. 

        Returns:
            dendropy.Tree: The reconstructed tree.
        """
        pass

class TreeSVD(ReconstructionMethod):
    def __init__(self, bifurcating=False):
        self.bifurcating = bifurcating

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

    def __call__(self, observations, taxa_metadata=None):
        self.observations = observations
        d = len(np.unique(observations))
        m = observations.shape[0]
        
        if taxa_metadata is None:
            taxa_metadata = utils.TaxaMetadata.default(m)
        else:
            if len(taxa_metadata) < m:
                raise ValueError(f"Not enough taxa provided for distance matrix (provided {len(taxa_metadata)}, should be {m}).")
        
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
        while len(available_clades) > (2 if self.bifurcating else 3): # this used to be 1
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
        return dendropy.Tree(taxon_namespace=taxa_metadata.taxon_namespace, seed_node=utils.merge_children(tuple(G[i] for i in available_clades)), is_rooted=False)
    def __repr__(self):
        return "TreeSVD"

class DistanceReconstructionMethod(ReconstructionMethod):
    """Method for reconstructing a binary tree that uses a matrix of similarities of the leaf nodes.

    The similarity matrix is calculated from the input sequence data using a function provided by the caller.

    Attributes:
        similarity_metric: A function that takes a matrix m x n matrix of sequence data and returns
            an m x m matrix of similarities scores in the range [0, 1].
    """

    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric

    def __call__(self, sequences, taxa_metadata=None):
        similarity_matrix = self.similarity_metric(sequences)
        return self.reconstruct_from_similarity(similarity_matrix, taxa_metadata)

    @abstractmethod
    def reconstruct_from_similarity(self, similarity_matrix, taxa_metadata=None):
        """Reconstruct the tree from the similarity matrix.

        Args:
            similarity_matrix: An m x m matrix of similarity scores in the range [0, 1], where m is the number of leaf nodes
            taxon_metadata: A TaxaMetadata object describing the observed nodes. The first element should
                correspond to the first row/column of `taxon_metadata`, and so on. If omitted, the default TaxonMetadata
                of size m is used.

        Returns:
            dendropy.Tree: The reconstructed tree.
        """
        pass

class NeighborJoining(DistanceReconstructionMethod):
    """Reconstructs a binary tree using the Neighbor Joining method.

    The distance matrix is calculated from the input sequence data using a similarity function provided by the caller
    and the formula `distance = - log(similarity)`

    Attributes:
        similarity_metric: A function that takes a matrix m x n matrix of sequence data and returns
            and m x m matrix of similarities scores in the range [0, 1].
    """

    def reconstruct_from_similarity(self, similarity_matrix, taxa_metadata=None):
        similarity_matrix = np.clip(similarity_matrix, a_min=1e-20, a_max=None)
        distance_matrix = -np.log(similarity_matrix)
        T = utils.array2distance_matrix(distance_matrix, taxa_metadata).nj_tree()
        return T

    def __repr__(self):
        return "NJ"
