import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import Bio.Phylo as Phylo
import Bio
import Bio.Phylo.TreeConstruction
import cProfile
import NoahClade

from importlib import reload
#reload(NoahClade)
import reconstruct_tree
from reconstruct_tree import estimate_tree_topology, similarity_matrix
from trials import random_discrete_tree, random_gaussian_tree

def neighbor_joining(observations, labels=None):
    if labels is None:
        m = observations.shape[0]
        labels = ["_"+str(j) for j in range(1, m+1)]
    align = Bio.Align.MultipleSeqAlignment([Bio.SeqRecord.SeqRecord(Bio.Seq.Seq("".join(str(char) for char in observations[i,:])), id=labels[i]) for i in range(observations.shape[0])])
    calculator = Bio.Phylo.TreeConstruction.DistanceCalculator('identity')
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor(calculator, 'nj')
    return constructor.build_tree(align)

def NJ_continuous(observations, labels=None):
    m = observations.shape[0]
    if labels is None:
        labels = ["_"+str(j) for j in range(1, m+1)]
    distances = -np.log(np.corrcoef(observations)**2)
    dm = Bio.Phylo.TreeConstruction._DistanceMatrix(labels, [list(distances[i,:i+1]) for i in range(0,m)])
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor()
    return constructor.nj(dm)

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

def comp_scorers(m, Ns, k, scorers, n_trees=3, obs_per_tree=3, proba_bounds=(0.75, 0.95), baselines=[], discrete=True):
    # Ns is array of n
    if discrete:
        transition_maker = NoahClade.NoahClade.gen_symmetric_transition
    else:
        transition_maker = NoahClade.NoahClade.gen_linear_transition
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
                ref_tree.root.gen_subtree_data(root_data, transition_maker)
                observations, labels = ref_tree.root.observe()
                #print("ref")
                #ref_tree.root.ascii()
                for scorer in scorers:
                    inferred_tree = estimate_tree_topology(observations, labels=labels, scorer=scorer, discrete=discrete)
                    print("{0} / {1}\t{2}".format(itr, total_iters, scorer.__name__))
                    #inferred_tree.root.ascii()
                    F1, precision, recall, RF = NoahClade.tree_Fscore(inferred_tree, ref_tree, verbose=True)
                    stats.append({"n": n, "scorer": scorer.__name__[6:], "F1%":100*F1, "precision%":100*precision, "recall%":100*recall, "RF":RF,})
                    itr += 1
                for baseline in baselines:
                    # TODO: refactor so you aren't repeating code here
                    inferred_tree = baseline(observations, labels=labels)
                    inferred_tree.root = NoahClade.NoahClade.convertClade(inferred_tree.root)
                    inferred_tree.root.reset_taxasets(labels)
                    print("{0} / {1}\t{2}".format(itr, total_iters, baseline.__name__))
                    #inferred_tree.root.ascii()
                    F1, precision, recall, RF = NoahClade.tree_Fscore(inferred_tree, ref_tree, verbose=True)
                    stats.append({"n": n, "scorer": baseline.__name__, "F1%":100*F1, "precision%":100*precision, "recall%":100*recall, "RF":RF,})
                    itr += 1
    return stats

if __name__ == "__main__":
    # %%

    m = 64
    n = 20_000
    k = 4
    ref_tree = random_discrete_tree(m, n, k, proba_bounds=(0.8, 0.9))
    ref_tree.root.ascii()
    obs, labels = ref_tree.root.observe()
    sim = similarity_matrix(obs)

    # %%
    scorers = [score_plain, score_plain12, score_plain_trace, score_sum, score_sumtrace, score_weird]

    # keep n < 10^6
    stats = pd.DataFrame(comp_scorers(m=64, Ns=[300, 1_000, 3_000, 10_000, 30_000], k=4, scorers=scorers, baselines=[neighbor_joining]))

    #%matplotlib inline
    #%matplotlib auto
    sns.catplot(data=stats, x="scorer", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)
    #plt.show()

    scorers2 = [score_sum, score_weird01]
    stats2 = pd.DataFrame(comp_scorers(m=64, Ns=[1_000, 3_000, 10_000], k=4, n_trees=3, obs_per_tree=3, scorers=scorers2))
    sns.catplot(data=stats2, x="scorer", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)

    # %%
    # continuous
    stats = pd.DataFrame(comp_scorers(m=64, Ns=[300, 1_000, 3_000, 10_000, 30_000, 100_000], k=4, scorers=[score_sum], baselines=[NJ_continuous], discrete=False))
    sns.catplot(data=stats, x="scorer", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)
