from functools import partial
from itertools import combinations

import numpy as np
import scipy.spatial.distance
import dendropy     #should this library be independent of dendropy? is that even possible?
from . import utils

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

def paralinear_distance(observations, classes=None):
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
    similarity = np.clip(similarity, a_min=1e-20, a_max=None)
    return -np.log(similarity)

def JC_distance_matrix(observations, classes=None):
    """Jukes-Cantor Corrected Distance"""
    assert classes is None
    if classes is None:
        classes = np.unique(observations)
    k = len(classes)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    inside_log = 1 - hamming_matrix*k/(k-1)
    inside_log = np.clip(inside_log, a_min=1e-16, a_max=None)
    return - (k-1) * np.log(inside_log)

def correlation_distance_matrix(observations):
    """Correlation Distance"""
    corr = np.abs(np.corrcoef(observations))
    corr = np.clip(corr, a_min=1e-16, a_max=None)
    return -np.log(corr)

def estimate_tree_topology(distance_matrix, namespace=None, scorer=sv2, scaler=1.0, bifurcating=False):
    m, m2 = distance_matrix.shape
    assert m == m2, "Distance matrix must be square"
    if namespace is None:
        namespace = utils.default_namespace(m)
    else:
        assert len(namespace) >= m, "Namespace too small for distance matrix"

    M = np.exp(-distance_matrix*scaler)

    # initialize leaf nodes
    G = [utils.leaf(i, namespace) for i in range(m)]

    available_clades = set(range(len(G)))   # len(G) == m
    # initialize Sigma
    Sigma = np.full((2*m,2*m), np.nan)  # we should only use entries that we set later; init to nan so they'll throw an error if we do
    sv1 = np.full((2*m,2*m), np.nan)
    for i,j in combinations(available_clades, 2):
        Sigma[i,j] = scorer(G[i].taxa_set, G[j].taxa_set, M)
        Sigma[j,i] = Sigma[i,j]    # necessary b/c sets have unstable order, so `combinations' could return either one

    # merge
    while len(available_clades) > (2 if bifurcating else 3): # this used to be 1
        left, right = min(combinations(available_clades, 2), key=lambda pair: Sigma[pair])
        G.append(utils.merge_children((G[left], G[right])))
        new_ix = len(G) - 1
        available_clades.remove(left)
        available_clades.remove(right)
        for other_ix in available_clades:
            Sigma[other_ix, new_ix] = scorer(G[other_ix].taxa_set, G[new_ix].taxa_set, M)
            Sigma[new_ix, other_ix] = Sigma[other_ix, new_ix]    # necessary b/c sets have unstable order, so `combinations' could return either one
        available_clades.add(new_ix)

    # HEYYYY why does ariel do something special for the last THREE groups? (rather than last two)
    # think about what would happen if at the end we have three groups: 1 leaf, 1 leaf, n-2 leaves
    # how would we know which leaf to attach, since score would be 0 for both??

    # return Phylo.BaseTree.Tree(G[-1])

    # for a bifurcating tree we're combining the last two available clades
    # for an unrooted one it's the last three because
    # if we're making unrooted comparisons it doesn't really matter which order we attach the last three
    return dendropy.Tree(taxon_namespace=namespace, seed_node=utils.merge_children((G[i] for i in available_clades)))

def estimate_edge_lengths(tree, distance_matrix):
    # check the PAUP* documentation
    pass

def neighbor_joining(distance_matrix, namespace=None):
    return utils.array2distance_matrix(distance_matrix, namespace).nj_tree()


class Reconstruction_Method:
    def __init__(self, core=estimate_tree_topology, distance=paralinear_distance, **kwargs):
        self.core = core
        self.distance = distance
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
        return self.reconstruct_from_array(observations, namespace)

    def reconstruct_from_array(self, observations, namespace=None):
        distance_matrix = self.distance(observations)
        tree = self.core(distance_matrix, namespace=namespace, **self.kwargs)
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
