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

def full_sv2(A1, A2, M):
    """Second Singular Value"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        return s[1] # the second eigenvalue

SVD2_OBJ = TruncatedSVD(n_components=2, n_iter=7)
def svd2(mat):
    if (mat.shape[0] == 1) | (mat.shape[1] == 1):
        return 0
    elif (mat.shape[0] == 2) | (mat.shape[1] == 2):
        return np.linalg.svd(mat, full_matricesbool=False, compute_uvbool=False)[1]
    else:
        return SVD2_OBJ.fit(mat).singular_values_[1]

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

    def __repr__(self):
        return "RAxML"

class DistanceReconstructionMethod(ReconstructionMethod):
    def __init__(self, similarity_metric):
        self.similarity_metric = similarity_metric

    def __call__(self, sequences, taxon_namespace=None):
        similarity_matrix = self.similarity_metric(sequences)
        return self.reconstruct_from_similarity(similarity_matrix, taxon_namespace)

    @abstractmethod
    def reconstruct_from_similarity(self, similarity_matrix, taxon_namespace=None):
        pass

class NeighborJoining(DistanceReconstructionMethod):
    def reconstruct_from_similarity(self, similarity_matrix, taxon_namespace=None):
        similarity_matrix = np.clip(similarity_matrix, a_min=1e-20, a_max=None)
        distance_matrix = -np.log(similarity_matrix)
        return utils.array2distance_matrix(distance_matrix, taxon_namespace).nj_tree()

    def __repr__(self):
        return "NJ"
        
class SpectralNeighborJoining(DistanceReconstructionMethod):
    def reconstruct_from_similarity(self, similarity_matrix, taxon_namespace=None):
        return self.estimate_tree_topology(similarity_matrix, taxon_namespace)

    def estimate_tree_topology(self, similarity_matrix, taxon_namespace=None, scorer=full_sv2, scaler=1.0, bifurcating=False):
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

    def __repr__(self):
        return "SNJ"
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
