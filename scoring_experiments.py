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
from reconstruct_tree import estimate_tree_topology, similarity_matrix, linear_similarity_matrix, JC_similarity_matrix, adj_jc, normalized_estimate_tree_topology
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

# almost as good
def score_hack1(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    norm = np.sqrt(M_A.shape[0]*M_A.shape[1])
    return s[1:].sum() / norm

def score_hack1half(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    norm = M_A.shape[0]*M_A.shape[1]
    return s[1:].sum() / norm

def score_hack2(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    norm = min(M_A.shape[0],M_A.shape[1])
    return s[1:].sum() / norm

def score_hack2half(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    sizeA, sizeAc = M_A.shape
    norm = sizeA*(sizeA-1)*sizeAc*(sizeAc-1)
    return s[1:].sum() / norm

def score_hack2and34(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    sizeA, sizeAc = M_A.shape
    norm = sizeA*(sizeA-1)*sizeAc*(sizeAc-1)
    return s[1:].sum() / np.sqrt(norm)

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

def score_weird6(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    s_sq = s**2
    return (s_sq.sum()**2 - (s_sq**2).sum())/2

# utter shit
def score_nuke(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    return s.sum()

def score_frob_norm(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    s_sq = s**2
    return (s_sq.sum() - s_sq[0]) / s_sq[0]
    # shouldn't that be / s_sq[0]**2?

# utter shit
def score_energy(A, M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    mean_s = s.mean()
    return np.abs(s-mean_s).sum()
    #if s.size <= 1:
    #    return 0

def overestimate(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    s_sq = s**2
    return (s_sq.sum())**2 - s_sq[0]**2

# bad
def score_stable_rank(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    s_sq = s**2
    return s_sq.sum()/s_sq[0] #frobenius norm sq / operator norm sq

def score_weird7(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    s_sq = s**2
    return ((s_sq.sum()**2 - (s_sq**2).sum())/2) / s_sq[0]

def score_weird8(A, M): # take sqrt of entries first
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(np.sqrt(M_A), compute_uv=False)
    s_sq = s**2
    return (s_sq.sum()**2 - (s_sq**2).sum())/2
# if you take the cube or the cube root it's definitely worse

def score_weird9(A, M):  #performs about same as weird8 ?
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(np.sqrt(M_A), compute_uv=False)
    s_sq = s**2
    pairs = M_A.shape[0]*M_A.shape[1]
    return ((s_sq.sum()**2 - (s_sq**2).sum())/2)/pairs
# if you do pairs^2 it's definitely worse

# transforming by taking svd of 1/M_A sucks

def score_stable_sqrt(A, M):  # not great :(
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(np.sqrt(M_A), compute_uv=False)
    s_sq = s**2
    return s_sq.sum()/s_sq[0]

def score_weird11(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(np.sqrt(M_A), compute_uv=False)
    s_sq = (s/s[0])**2
    return ((s_sq.sum()**2 - (s_sq**2).sum())/2)
    # shouldn't these s's be s_sq ??

# %%

def comp_scorers(m, Ns, k, scorers, n_trees=12, obs_per_tree=1, proba_bounds=(0.75, 0.95), std_bounds=(0.1, 0.3), baselines=[], discrete=True):
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
            ref_tree = random_gaussian_tree(m, 1, std_bounds=std_bounds)
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

def clean_stats(data, Ns=None):
    if Ns is not None:
        data = data[data['n'].isin(Ns)]
    return data

def violin(data, Ns=None):
    sns.catplot(data=clean_stats(data, Ns=Ns), x="method", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)

def box(data, Ns=None):
    sns.boxplot(data=clean_stats(data, Ns=Ns), x="n", y="RF", hue='method')

def lines(data, Ns=None):
    sns.pointplot(data=clean_stats(data, Ns=Ns), x="n", y="RF", hue='method', dodge=0.2)

# %%

if __name__ == "__main__":

    normed = partial(estimate_tree_topology, scorer=score_weird6, similarity=normalized_similarity_matrix)
    normed.__name__ = 'normed'

    adjusted = partial(estimate_tree_topology, scorer=score_sum, similarity=adj_jc)
    adjusted.__name__ = 'adjusted'

    stats = pd.DataFrame(comp_scorers(m=64, Ns=[300, 1_000, 3_000, 10_000, 30_000], k=4, scorers=[score_sum, score_weird6], baselines=[normed], n_trees=20))
    violin(stats)
    box(stats, Ns=[1000, 3000, 10_000, 30_000])
    lines(stats, Ns=[1000, 3000, 10_000, 30_000])

    stats = pd.DataFrame(comp_scorers(m=100, Ns=[1_000, 3_000, 10_000], k=4, scorers=[score_weird7, score_weird8, score_weird9, score_sum], baselines=[], n_trees=50))
    stats2 = pd.DataFrame(comp_scorers(m=100, Ns=[1_000, 3_000, 10_000], k=4, scorers=[score_weird11], baselines=[], n_trees=50))
    #stats = stats.append(stats2)
    violin(stats)
    box(stats)
    lines(stats.append(stats2))
    len(stats)

    print(("="*40+"\n")*5)
    #from reconstruct_tree import normalized_estimate_tree_topology
    stats = pd.DataFrame(comp_scorers(m=100, Ns=[1_000, 3_000, 10_000], k=4, scorers=[score_plain], baselines=[normalized_estimate_tree_topology, normalized_estimate_tree_topology_OLD, normed], n_trees=30))
    violin(stats)
    box(stats)
    lines(stats)

    from reconstruct_tree import normalized_similarity_matrix
    normed = partial(estimate_tree_topology, scorer=score_plain, similarity=normalized_similarity_matrix)
    normed.__name__ = 'normed'
    stats2 = pd.DataFrame(comp_scorers(m=100, Ns=[1_000, 3_000, 10_000], k=4, scorers=[], baselines=[normed], n_trees=30))

    stats[stats['method'].isin(["plain", 'normed'])].groupby(['method', 'n']).mean()

    stats = stats.append(stats2)

    # these are really all the same
    stats = pd.DataFrame(comp_scorers(m=64, Ns=[300, 1_000, 3_000, 10_000, 30_000], k=4, scorers=[score_plain, score_sum, score_hack3], baselines=[], n_trees=50))
    violin(stats)
    box(stats)
    lines(stats)

    (1+0.1**2)/(0.1**4)
    (1/0.1)**4
    (1/0.1)**2

    # stats = pd.DataFrame(comp_scorers(m=20, Ns=[300, 500, 1_000, 1_500, 2_000], k=4, scorers=[score_sum, score_weird6, score_quartet_sum]))

    scorers2 = [score_plain, score_sum, score_weird02, score_weird03]
    stats2 = pd.DataFrame(comp_scorers(m=80, Ns=[300, 1_000, 3_000, 10_000, 30_000], k=4, n_trees=4, obs_per_tree=3, scorers=scorers2))
    sns.catplot(data=stats2, x="method", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)

    stats4 = pd.DataFrame(comp_scorers(m=200, Ns=[3_000, 10_000, 30_000, 100_000], k=4, n_trees=4, obs_per_tree=3, scorers=scorers2))
    sns.catplot(data=stats4, x="method", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)


    # %%
    # continuous
    stats = pd.DataFrame(comp_scorers(m=30, Ns=[3_000, 10_000, 30_000, 40_000, 50_000], k=4, scorers=[score_sum, score_weird6], baselines=[NJ_continuous], discrete=False, n_trees=10, std_bounds=(0.1, 0.7)))
    #stats = pd.DataFrame(comp_scorers(m=30, Ns=[1_000, 3_000, 10_000,], k=4, scorers=[score_sum, score_weird6, score_quartet_sum], baselines=[NJ_continuous], discrete=False, n_trees=4))

    violin(stats)
    lines(stats)

    stats.groupby(['n', 'method']).mean()
