from itertools import combinations
import numpy as np
from NoahClade import NoahClade
import Bio.Phylo as Phylo
import scipy.spatial.distance

def second_sv(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        # the second eigenvalue
        return s[1]

def first_sv(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    return s[0]

def score_split(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        return s[1] # the second eigenvalue

def similarity_matrix(observations, classes=None):
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
    return M

def JC_similarity_matrix(observations, classes=None):
    # TODO: use classes argument to make this robust to missing data
    if classes is None:
        classes = np.unique(observations)
    else:
        assert False
    k = len(classes)

    normalized_hamming_distance = scipy.spatial.distance.pdist(observations, metric='hamming')
    # under J-C model, even independent sequences have a maximum hamming distance of (k-1)/k
    # in which case determinant is 0
    normalized_hamming_distance = np.clip(normalized_hamming_distance, a_min=None, a_max=(k-1)/k)
    expected_dets = (1 - normalized_hamming_distance*k/(k-1))**(k-1)
    dm  = scipy.spatial.distance.squareform(expected_dets)
    # under jukes-cantor model, base frequencies are equal at all nodes
    # so for diag elements, take determinant of (1/k)* I_k
    np.fill_diagonal(dm, (1/k)**k)
    return dm

def linear_similarity_matrix(observations):
    return np.corrcoef(observations)**2

# TODO: remove scorer argument when we've settled on one
def estimate_tree_topology(observations, labels=None, similarity=similarity_matrix, bifurcating=False, scorer=score_split):
    m, n = observations.shape

    """if labels is None:
        labels = [str(i) for i in range(m)]
    assert len(labels) == m"""

    M = similarity(observations)

    # initialize leaf nodes
    G = [NoahClade.leaf(i, labels=labels, data=observations[i,:]) for i in range(m)]

    available_clades = set(range(len(G)))   # len(G) == m
    # initialize Sigma
    Sigma = np.full((2*m,2*m), np.nan)  # we should only use entries that we set later; init to nan so they'll throw an error if we do
    sv1 = np.full((2*m,2*m), np.nan)
    sv2 = np.full((2*m,2*m), np.nan)
    for i,j in combinations(available_clades, 2):
        A = G[i].taxa_set | G[j].taxa_set
        Sigma[i,j] = scorer(A, M)
        Sigma[j,i] = Sigma[i,j]    # necessary b/c sets have unstable order, so `combinations' could return either one
        sv1[i,j] = first_sv(A, M)
        sv1[j,i] = sv1[i,j]
        sv2[i,j] = second_sv(A, M)
        sv2[j,i] = sv2[i,j]

    # merge
    while len(available_clades) > (2 if bifurcating else 3): # this used to be 1
        left, right = min(combinations(available_clades, 2), key=lambda pair: Sigma[pair])
        G.append(NoahClade(clades=[G[left], G[right]], score=Sigma[left, right]))
        # G.append(merge_taxa(G[left], G[right], score=Sigma[left, right]))
        new_ix = len(G) - 1
        available_clades.remove(left)
        available_clades.remove(right)
        for other_ix in available_clades:
            A = G[other_ix].taxa_set | G[-1].taxa_set
            Sigma[other_ix, new_ix] = scorer(A, M)
            Sigma[new_ix, other_ix] = Sigma[other_ix, new_ix]    # necessary b/c sets have unstable order, so `combinations' could return either one
            sv1[other_ix, new_ix] = first_sv(A, M)
            sv1[new_ix, other_ix] = sv1[other_ix, new_ix]
            sv2[other_ix, new_ix] = second_sv(A, M)
            sv2[new_ix, other_ix] = sv2[other_ix, new_ix]
        available_clades.add(new_ix)

    # HEYYYY why does ariel do something special for the last THREE groups? (rather than last two)
    # think about what would happen if at the end we have three groups: 1 leaf, 1 leaf, n-2 leaves
    # how would we know which leaf to attach, since score would be 0 for both??

    # return Phylo.BaseTree.Tree(G[-1])

    # for a bifurcating tree we're combining the last two available clades
    # for an unrooted one it's the last three because
    # if we're making unrooted comparisons it doesn't really matter which order we attach the last three
    return Phylo.BaseTree.Tree(NoahClade(clades=[G[i] for i in available_clades])) #, Sigma, sv1, sv2

def estimate_tree_topology_multiclass(observations, labels=None, bifurcating=False):
    return estimate_tree_topology(observations, labels=labels, similarity=similarity_matrix, bifurcating=bifurcating)

def estimate_tree_topology_continuous(observations, labels=None, bifurcating=False):
    return estimate_tree_topology(observations, labels=labels, similarity=linear_similarity_matrix, bifurcating=bifurcating)

def estimate_tree_topology_Jukes_Cantor(observations, labels=None, bifurcating=False):
    return estimate_tree_topology(observations, labels=labels, similarity=JC_similarity_matrix, bifurcating=bifurcating)


import numpy as np
import matplotlib.pylab as plt
def fun(p):
    return (1 - p*4/3)**3

def test_plot(p, sample_sizes, fun=fun):
    ests = [fun(np.random.binomial(n=1, p=p, size=(n)).mean()) for n in sample_sizes]
    true = fun(p)
    print(true)
    print(sum(ests)/len(ests))
    fig = plt.figure()
    plt.plot(ests)
    plt.plot([true]*len(sample_sizes))
    fig.show()



# def tracking_estimate_tree_topology(observations, labels=None, discrete=True, bifurcating=False, scorer=score_split, good_splits):
#     m, n = observations.shape
#
#     """if labels is None:
#         labels = [str(i) for i in range(m)]
#     assert len(labels) == m"""
#
#     if discrete:
#         M = similarity_matrix(observations)
#     else:
#         # TODO: should this be biased or not?
#         #M = np.cov(observations, bias=False)
#         M = np.corrcoef(observations)
#
#     # initialize leaf nodes
#     G = [NoahClade.leaf(i, labels=labels, data=observations[i,:]) for i in range(m)]
#
#
#     available_clades = set(range(len(G)))   # len(G) == m
#     # initialize Sigma
#     Sigma = np.full((2*m,2*m), np.nan)  # we should only use entries that we set later; init to nan so they'll throw an error if we do
#     sv1 = np.full((2*m,2*m), np.nan)
#     sv2 = np.full((2*m,2*m), np.nan)
#     for i,j in combinations(available_clades, 2):
#         A = G[i].taxa_set | G[j].taxa_set
#         Sigma[i,j] = scorer(A, M)
#         Sigma[j,i] = Sigma[i,j]    # necessary b/c sets have unstable order, so `combinations' could return either one
#         sv1[i,j] = first_sv(A, M)
#         sv1[j,i] = sv1[i,j]
#         sv2[i,j] = second_sv(A, M)
#         sv2[j,i] = sv2[i,j]
#
#     records = []
#
#     # merge
#     while len(available_clades) > (2 if bifurcating else 3): # this used to be 1
#         left, right = min(combinations(available_clades, 2), key=lambda pair: Sigma[pair])
#         G.append(NoahClade(clades=[G[left], G[right]], score=Sigma[left, right]))
#         # G.append(merge_taxa(G[left], G[right], score=Sigma[left, right]))
#         new_ix = len(G) - 1
#         available_clades.remove(left)
#         available_clades.remove(right)
#         #
#         real_possibilities = []
#         for left, right in combinations(available_clades, 2):
#             if NoahClade.taxaset2ixs(G[left].taxa_set | G[right].taxa_set) in good_splits:
#                 real_possibilities.append((left, right))
#         left_real, right_real = min(real_possibilities, key=lambda pair: Sigma[pair])
#         records.append({"newclade": G[-1], "score": G[-1].score, "nleft": G[left].taxa_set.sum(), "nright": G[right].taxa_set.sum(), "nA": G[-1].taxa_set.sum(),
#                         "bestreal": NoahClade(clades=[G[left_real], G[right_real]], score=Sigma[left_real, right_real]), "scorereal": Sigma[left_real, right_real]})
#         #
#         for other_ix in available_clades:
#             A = G[other_ix].taxa_set | G[-1].taxa_set
#             Sigma[other_ix, new_ix] = scorer(A, M)
#             Sigma[new_ix, other_ix] = Sigma[other_ix, new_ix]    # necessary b/c sets have unstable order, so `combinations' could return either one
#             sv1[other_ix, new_ix] = first_sv(A, M)
#             sv1[new_ix, other_ix] = sv1[other_ix, new_ix]
#             sv2[other_ix, new_ix] = second_sv(A, M)
#             sv2[new_ix, other_ix] = sv2[other_ix, new_ix]
#         available_clades.add(new_ix)
#
#     # HEYYYY why does ariel do something special for the last THREE groups? (rather than last two)
#     # think about what would happen if at the end we have three groups: 1 leaf, 1 leaf, n-2 leaves
#     # how would we know which leaf to attach, since score would be 0 for both??
#
#     # return Phylo.BaseTree.Tree(G[-1])
#
#     # for a bifurcating tree we're combining the last two available clades
#     # for an unrooted one it's the last three because
#     # if we're making unrooted comparisons it doesn't really matter which order we attach the last three
#     return Phylo.BaseTree.Tree(NoahClade(clades=[G[i] for i in available_clades])) #, Sigma, sv1, sv2
