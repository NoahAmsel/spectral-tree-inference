from itertools import combinations

import dendropy
import numpy as np

from . import utils
from .reconstruct_tree import DistanceReconstructionMethod

def sv2(M_A):
    """Second Singular Value"""
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        return s[1] # the second eigenvalue

class SpectralNeighborJoining(DistanceReconstructionMethod):
    """Reconstructs a binary tree using the Neighbor Joining method.

    Attributes:
        similarity_metric: A function that takes a matrix m x n matrix of sequence data and returns
            and m x m matrix of similarities scores in the range [0, 1].
        scorer: A function that scores a noisy matrix on its rank-1-ness, with 0 being certainly rank-1 and more
            positive scores meaning more certainly not rank-1. E.g., the second eigenvalue of the matrix.
        bifurcating: Whether to include an inner node with only two neighbors instead of three.
        alpha: A power by to which to raise each of the pairwise similarities -- equivalently, a scalar with which to multiply each of the pairwise distances.
    """ 
    def __init__(self, similarity_metric, scorer=sv2, bifurcating=False, alpha=1.0):
        super(SpectralNeighborJoining, self).__init__(similarity_metric)
        self.scorer = scorer
        self.bifurcating = bifurcating
        self.alpha = alpha
    
    def _score_merge(self, A1, A2, similarity_matrix):
        """Scores how well the data supports the existance of a clade formed by merging
        two given clades.

        Args:
            A1: A binary array of length m indicating membership in the first clade, where m is the number of leaf nodes.
            A2: A binary array of length m indicating membership in the second clade, where m is the number of leaf nodes.
            similarity_matrix: The similarity matrix

        Returns:
            A non-negative score, where 0 indicates that the union of A1 and A2 is certainly a clade, and a large positive score
            indicates that is certainly is not.
        """
        A = A1 | A2
        M_A = similarity_matrix[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
        return self.scorer(M_A)

    def reconstruct_from_similarity(self, similarity_matrix, taxa_metadata=None):
        # TODO: should all distance methods do these steps
        m, m2 = similarity_matrix.shape
        if m != m2:
            raise ValueError(f"Distance matrix should be square but has dimensions {m} x {m2}.")
        if taxa_metadata is None:
            taxa_metadata = utils.TaxaMetadata.default(m)
        else:
            if len(taxa_metadata) != m:
                raise ValueError(f"Namespace size ({len(taxa_metadata)}) should match distance matrix dimension ({m}).")

        if self.alpha != 1.0:
            similarity_matrix = similarity_matrix ** self.alpha

        # initialize leaf nodes
        G = taxa_metadata.all_leaves()

        available_clades = set(range(len(G)))   # note that len(G) == m

        # initialize Sigma
        Sigma = np.full((2*m,2*m), np.nan)  # we should only use entries that we set later; init to nan so they'll throw an error if we do
        sv1 = np.full((2*m,2*m), np.nan)
        for i,j in combinations(available_clades, 2):
            Sigma[i,j] = self._score_merge(G[i].mask, G[j].mask, similarity_matrix)
            Sigma[j,i] = Sigma[i,j]    # necessary b/c sets have unstable order, so `combinations' could return either one

        # merge
        while len(available_clades) > (2 if self.bifurcating else 3): # this used to be 1
            left, right = min(combinations(available_clades, 2), key=lambda pair: Sigma[pair])
            G.append(utils.merge_children((G[left], G[right])))
            new_ix = len(G) - 1
            available_clades.remove(left)
            available_clades.remove(right)
            for other_ix in available_clades:
                Sigma[other_ix, new_ix] = self._score_merge(G[other_ix].mask, G[new_ix].mask, similarity_matrix)
                Sigma[new_ix, other_ix] = Sigma[other_ix, new_ix]    # necessary b/c sets have unstable order, so `combinations' could return either one
            available_clades.add(new_ix)

        # for a bifurcating tree we're combining the last two available clades
        # for an unrooted one it's the last three because
        # if we're making unrooted comparisons it doesn't really matter which order we attach the last three
        return dendropy.Tree(taxon_namespace=taxa_metadata.taxon_namespace, seed_node=utils.merge_children(tuple(G[i] for i in available_clades)), is_rooted=False)

    def __repr__(self):
        return "SNJ"

##########################################################
##             Additional SNJ Scorers
##########################################################

def max_quartet(M_A):
    "maximal quartet criterion"
    m_1,m_2 = M_A.shape
    max_w = 0
    for x in combinations(range(m_1), 2):
        for y in combinations(range(m_2),2):
            w = np.abs(M_A[x[0],y[0]]*M_A[x[1],y[1]]-M_A[x[0],y[1]]*M_A[x[1],y[0]])
            if w>max_w:
                max_w = w
    return max_w

def sum_sigular_values(M_A):
    """Normalized Sum of Squared Quartets"""
    s = np.linalg.svd(M_A, compute_uv=False)
    return np.linalg.norm(s[1:])

def sum_squared_quartets(M_A):
    """Normalized Sum of Squared Quartets"""
    norm_sq = np.linalg.norm(M_A)**2
    num = norm_sq**2 - np.linalg.norm(M_A.T.dot(M_A))**2
    return num / norm_sq

def average_quartets(M_A):
    """Normalized Sum of Squared Quartets"""
    m_1,m_2 = M_A.shape
    norm_sq = m_1*(m_1-1)*m_2*(m_2-1)
    num = norm_sq**2 - np.linalg.norm(M_A.T.dot(M_A))**2
    return num / norm_sq