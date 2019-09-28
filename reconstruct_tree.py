from itertools import combinations
import numpy as np
import scipy.spatial.distance
import utils

def score_split(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        return s[1] # the second eigenvalue

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
    similarity = M / np.sqrt(np.outer(diag,diag))
    similarity = np.clip(similarity, a_min=1e-20, a_max=None)
    return -np.log(similarity)

def JC_distance_matrix(observations, classes=None):
    assert classes is None
    if classes is None:
        classes = np.unique(observations)
    k = len(classes)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    inside_log = 1 - hamming_matrix*k/(k-1)
    inside_log = np.clip(inside_log, a_min=1e-16, a_max=None)
    return - (k-1) * np.log(inside_log)

def correlation_distance_matrix(observations):
    corr = np.corrcoef(observations)
    corr = np.clip(corr, a_min=1e-16, a_max=None)
    return np.log(corr)

def estimate_tree_topology(distance_matrix, namespace=None, scorer=score_split, bifurcating=False):
    m, m2 = distance_matrix.shape
    assert m == m2, "Distance matrix must be square"
    if namespace is None:
        namespace = utils.new_default_namespace(m)
    else:
        assert len(namespace) >= m, "Namespace too small for distance matrix"

    # initialize leaf nodes
    G = [utils.leaf(i, namespace) for i in range(m)]

    available_clades = set(range(len(G)))   # len(G) == m
    # initialize Sigma
    Sigma = np.full((2*m,2*m), np.nan)  # we should only use entries that we set later; init to nan so they'll throw an error if we do
    sv1 = np.full((2*m,2*m), np.nan)
    sv2 = np.full((2*m,2*m), np.nan)
    for i,j in combinations(available_clades, 2):
        A = G[i].taxa_set | G[j].taxa_set
        Sigma[i,j] = scorer(A, M)
        Sigma[j,i] = Sigma[i,j]    # necessary b/c sets have unstable order, so `combinations' could return either one
        #sv1[i,j] = first_sv(A, M)
        #sv1[j,i] = sv1[i,j]
        #sv2[i,j] = second_sv(A, M)
        #sv2[j,i] = sv2[i,j]

    # merge
    while len(available_clades) > (2 if bifurcating else 3): # this used to be 1
        left, right = min(combinations(available_clades, 2), key=lambda pair: Sigma[pair])
        G.append(utils.merge_children((G[left], G[right])))
        new_ix = len(G) - 1
        available_clades.remove(left)
        available_clades.remove(right)
        for other_ix in available_clades:
            A = G[other_ix].taxa_set | G[-1].taxa_set
            Sigma[other_ix, new_ix] = scorer(A, M)
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

if __name__ == "__main__":
    from utils import *
    t = balanced_binary(4)
    all_data = temp_dataset_maker(t, 1000, 0.1)
    observations, _ = charmatrix2array(all_data[0])
    dist = JC_distance_matrix(observations)
    dist[:5,:5]

    dist = paralinear_distance(observations)
    dist[:5,:5]

    dist = JC_similarity_matrix(observations)
    np.log(dist[:5,:5])

    scipy.spatial.distance.pdist(observations, metric='hamming')
