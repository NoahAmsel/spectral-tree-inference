import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import Bio.Phylo as Phylo
import Bio
import Bio.Phylo.TreeConstruction
import NoahClade
from functools import partial
from itertools import combinations, product

from importlib import reload
#reload(baselines)
from reconstruct_tree import estimate_tree_topology, similarity_matrix, linear_similarity_matrix, JC_similarity_matrix
from baselines import *
from NoahClade import random_discrete_tree, random_gaussian_tree

def score_plain(A, M):
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

def score_plain12(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        # the second eigenvalue normalized by this sum
        return s[1]/(s[0]+s[1])

def score_plain_trace(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        # the second eigenvalue normalized by this sum
        return s[1]/(s.sum())

def score_sum(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    return s[1:].sum()

def score_sum_tracker(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)

    m = M.shape[0]
    leafset = tuple(np.nonzero(A)[0])
    s1 = s[0]
    s2 = s[1]
    s23 = s[1:].sum()
    size = min(A.sum(), (~A).sum())
    score_sum_tracker.records.append({"|A|": size, "|A|/|T|":size/m, "m": m, "leafset": leafset, "s1":s1, "s2":s2, "s23":s23})

    return s[1:].sum()
score_sum_tracker.records = []

def score_sumtrace(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    return s[1:].sum()/s.sum()
    #return 1 - (s[0]/s.sum()) # how is this numerically?

def score_gap(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        # if it's a good split, s[1] = 0 so the gap is big
        # if it's a bad splits, s[0] almost is s[1] so gap is small
        # add a negative to reverse this (we're favoring small splits = very negative)
        return -(s[0]**2 - s[1]**2)

def score_gap12(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        # if it's a good split, s[1] = 0 so the gap is big
        # if it's a bad splits, s[0] almost is s[1] so gap is small
        # add a negative to reverse this (we're favoring small splits = very negative)
        return -(s[0]**2 - s[1]**2)/(s[0]**2 + s[1]**2)

def score_gapnorm(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        # if it's a good split, s[1] = 0 so the gap is big
        # if it's a bad splits, s[0] almost is s[1] so gap is small
        # add a negative to reverse this (we're favoring small splits = very negative)
        return -(s[0]**2 - s[1]**2)/(np.linalg.norm(M_A)**2)

def score_weird01(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    frob = np.linalg.norm(M_A)**4
    gap = (s[0]**2 - s[1]**2)**2
    return np.sqrt(frob - gap) / (2*s[0]*s[1])

def score_weird02(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    frob = np.linalg.norm(M_A)**4
    gap = (s[0]**2 - s[1]**2)**2
    return (frob - gap)/4

def score_weird03(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    pseudo_frob = (s[0]**2 + s[1]**2)**2
    gap = (s[0]**2 - s[1]**2)**2
    return (pseudo_frob - gap)/4

def score_hack1(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    norm = np.sqrt(M_A.shape[0]*M_A.shape[1])
    return s[1:].sum() / norm

def score_hack2(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    norm = min(M_A.shape[0],M_A.shape[1])
    return s[1:].sum() / norm

def score_weird04(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    #if s.size <= 1:
    #    return 0
    return (s[0]**2)*(s[1:]**2).sum()

def score_hack3(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    return (s[1:]**2).sum()
    # better for small n but eventually worse

def det22(mat22):
    return mat22[0,0]*mat22[1,1] - mat22[0,1]*mat22[1,0]
def score_quartet_sum(A, M):
    M_A = M[np.ix_(A, ~A)]
    nA, nAc = M_A.shape
    A_pairs = list(combinations(range(nA), 2)) # the `list' is unnecessary but makes debugging easier
    Ac_pairs = list(combinations(range(nAc), 2))
    return sum(det22(M_A[np.ix_(pair1, pair2)])**2 for pair1, pair2 in product(A_pairs, Ac_pairs))

def score_sqrt_sum(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    M_A = np.sqrt(M_A)
    s = np.linalg.svd(M_A, compute_uv=False)
    return s[1:].sum()

def score_weird5(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    #if s.size <= 1:
    #    return 0
    s1_square = s[0]**2
    s_square_sum = (s**2).sum()
    return s1_square*(s_square_sum - s1_square)/s_square_sum

def score_energy(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)

    #if s.size <= 1:
    #    return 0
    # TODO: sum of abs of eigenvalue - mean of eigenvalues

def score_weird6(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    s_sq = s**2
    return (s_sq.sum()**2 - (s_sq**2).sum())/2

def score_weird7(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    s_sq = s**2
    return ((s.sum()**2 - (s**2).sum())/2) / s_sq[0]



# %%

def comp_scorers(m, Ns, k, scorers, n_trees=6, obs_per_tree=2, proba_bounds=(0.75, 0.95), baselines=[], discrete=True):
    # Ns is array of n
    if discrete:
        transition_maker = NoahClade.NoahClade.gen_symmetric_transition
        #similarity = similarity_matrix
        similarity = JC_similarity_matrix
        # THIS DOES NOT DO JUKES CANTOR CORRECTION
        #transition_maker = NoahClade.NoahClade.jukes_cantor_transition
    else:
        transition_maker = NoahClade.NoahClade.gen_linear_transition
        similarity = linear_similarity_matrix
    stats = []
    itr = 1
    total_iters = n_trees*obs_per_tree*len(Ns)*(len(scorers)+len(baselines))

    for _ in range(n_trees):
        if discrete:
            ref_tree = random_discrete_tree(m, 1, k, proba_bounds=proba_bounds)
        else:
            ref_tree = random_gaussian_tree(m, 1, std_bounds=(0.1, 0.3))
        for _ in range(obs_per_tree):
            for n in Ns:
                root_data = np.random.choice(a=k, size=n)
                ref_tree.root.gen_subtree_data(root_data, transition_maker, proba_bounds=proba_bounds)
                observations, labels = ref_tree.root.observe()
                #print("ref")
                #ref_tree.root.ascii()
                scorer_methods = []
                for scorer in scorers:
                    scorer_method = partial(estimate_tree_topology, scorer=scorer, similarity=similarity)
                    scorer_method.__name__ = scorer.__name__[6:]
                    scorer_methods.append(scorer_method)
                for method in scorer_methods+baselines:
                    inferred_tree = method(observations, labels=labels)
                    #inferred_tree.root.ascii()
                    F1, precision, recall, RF = NoahClade.tree_Fscore(inferred_tree, ref_tree)
                    stats.append({"n": n, "method": method.__name__, "F1%":100*F1, "precision%":100*precision, "recall%":100*recall, "RF":RF,})
                    print("{0} / {1}\t{2}\tRF:{3}\tF1 {4:.1f}%\tn {5}".format(itr, total_iters, method.__name__, RF, 100*F1, n))
                    itr += 1
    return stats

# %%

def get_parent(tree, child_clade):
    node_path = tree.get_path(child_clade)
    return node_path[-2]

def p(my_clade):
    _, counts = np.unique(my_clade.data, return_counts=True)
    mat = np.diag(counts/len(my_clade.data))
    return mat
    return np.linalg.det(mat)

from sklearn.metrics import confusion_matrix
def testerest(tree, lab1, lab2, n):
    #tree = ref_tree
    #lab1 = "taxon56"
    #lab2 = 'taxon34'
    x = ref_tree.find_any(lab1)
    y = ref_tree.find_any(lab2)
    a = get_parent(tree, x)
    assert a is get_parent(tree, y)
    pxy = np.linalg.det(confusion_matrix(x.data, y.data) / n)
    print(p(x)*p(y)/pxy, p(a))

p(get_parent(ref_tree, "taxon56"))

np.linalg.det(ref_tree.find_any("taxon56").transition.matrix)
p(get_parent(ref_tree, "taxon56"))
p(ref_tree.find_any("taxon56"))
testerest(ref_tree, "taxon56", 'taxon34', n)

if __name__ == "__main__":

    stats5 = pd.DataFrame(comp_scorers(m=64, Ns=[1_000, 3_000, 10_000], k=4, scorers=[score_sum], baselines=[neighbor_joining, NJ, NJ_JC]))
    sns.catplot(data=stats5, x="method", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)


    # %%
    scorers = [score_plain, score_plain12, score_plain_trace, score_sum, score_sumtrace, score_weird]

    # keep n < 10^6
    stats = pd.DataFrame(comp_scorers(m=64, Ns=[300, 1_000, 3_000, 10_000, 30_000], k=4, scorers=[score_plain, score_weird6]))

    sns.catplot(data=stats, x="method", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)
    #plt.show()

    # stats = pd.DataFrame(comp_scorers(m=20, Ns=[300, 500, 1_000, 1_500, 2_000], k=4, scorers=[score_sum, score_weird6, score_quartet_sum]))

    scorers2 = [score_plain, score_sum, score_weird02, score_weird03]
    stats2 = pd.DataFrame(comp_scorers(m=80, Ns=[300, 1_000, 3_000, 10_000, 30_000], k=4, n_trees=4, obs_per_tree=3, scorers=scorers2))
    sns.catplot(data=stats2, x="method", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)

    stats4 = pd.DataFrame(comp_scorers(m=200, Ns=[3_000, 10_000, 30_000, 100_000], k=4, n_trees=4, obs_per_tree=3, scorers=scorers2))
    sns.catplot(data=stats4, x="method", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)


    # %%
    # continuous
    NNNs=[300, 1_000, 3_000, 10_000, 30_000, 100_000]
    stats = pd.DataFrame(comp_scorers(m=128, Ns=[3_000, 10_000, 30_000,], k=4, scorers=[score_sum, score_sqrt_sum], baselines=[], discrete=False), n_trees=10) #NJ_continuous

    stats = pd.DataFrame(comp_scorers(m=30, Ns=[1_000, 3_000, 10_000,], k=4, scorers=[score_sum, score_weird6, score_quartet_sum], baselines=[NJ_continuous], discrete=False, n_trees=4))

    sns.catplot(data=stats, x="method", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)
