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
from trials import random_discrete_tree

def neighbor_joining(observations, labels=None):
    if labels is None:
        m = observations.shape[0]
        labels = ["_"+str(j) for j in range(1, m+1)]
    align = Bio.Align.MultipleSeqAlignment([Bio.SeqRecord.SeqRecord(Bio.Seq.Seq("".join(str(char) for char in observations[i,:])), id=labels[i]) for i in range(observations.shape[0])])
    calculator = Bio.Phylo.TreeConstruction.DistanceCalculator('identity')
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor(calculator, 'nj')
    return constructor.build_tree(align)

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

def score_sumtrace(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    return s[1:].sum()/s.sum()
    #return 1 - (s[0]/s.sum()) # how is this numerically?

def score_weird(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        return np.linalg.norm(M_A)**2 - (s[0] - s[1])**2
        """try:
            return np.linalg.norm(M_A)**2 - (s[0] - s[1])**2
        except:
            print(s)
            print(M_A)
            exit()"""

def comp_scorers(m, Ns, k, scorers, n_trees=3, obs_per_tree=3, proba_bounds=(0.75, 0.95), baselines=[neighbor_joining]):
    # Ns is array of n
    transition_maker = NoahClade.NoahClade.gen_symmetric_transition
    stats = []

    itr = 1
    total_iters = n_trees*obs_per_tree*len(Ns)*(len(scorers)+len(baselines))
    for _ in range(n_trees):
        ref_tree = random_discrete_tree(m, 1, k, proba_bounds=proba_bounds)
        for _ in range(obs_per_tree):
            for n in Ns:
                root_data = np.random.choice(a=k, size=n)
                ref_tree.root.gen_subtree_data(root_data, transition_maker)
                observations, labels = ref_tree.root.observe()
                #print("ref")
                #ref_tree.root.ascii()
                for scorer in scorers:
                    inferred_tree = estimate_tree_topology(observations, labels=labels, scorer=scorer)
                    print("{0} / {1}\t{2}".format(itr, total_iters, scorer.__name__))
                    #inferred_tree.root.ascii()
                    F1, precision, recall, RF = NoahClade.tree_Fscore(inferred_tree, ref_tree, verbose=True)
                    correct = 1 if (F1==1) else 0
                    stats.append({"n": n, "scorer": scorer.__name__[6:], "F1%":100*F1, "precision%":100*precision, "recall%":100*recall, "correct %": 100*correct, "RF":RF,})
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

# %%
#reload(reconstruct_tree)
#from reconstruct_tree import estimate_tree_topology

if __name__ == "__main__":
    # %%
    """
    m = 64
    n = 20_000
    k = 4
    print(k)
    ref_tree = random_discrete_tree(m, n, k, proba_bounds=(0.8, 0.9))
    #ref_tree.root.ascii()
    obs, labels = ref_tree.root.observe()

    inf, Sigma, sv1, sv2 = estimate_tree_topology(obs, labels=labels, scorer=score_plain12)
    temp = sv2/(sv1+sv2)
    inf.root.ascii()
    assert np.allclose(temp, Sigma, atol=1e-2, equal_nan=True)
    with np.printoptions(precision=6, suppress=True):
        print(Sigma[64,:])
        print(sv1[64,:])
        print(sv2[64,:])
        print(temp[64,:])
    """

    # %%
    scorers = [score_plain, score_plain12, score_plain_trace, score_sum, score_sumtrace, score_weird]
    #scorers = [score_plain, score_plain12]

    # keep n < 10^6
    # Ns=[300, 1_000, 3_000, 10_000, 30_000]
    stats = pd.DataFrame(comp_scorers(m=64, Ns=[300, 1_000, 3_000, 10_000, 30_000], k=4, scorers=scorers))

    #%matplotlib inline
    #%matplotlib auto
    #sns.violinplot(data=stats, x="scorer", y="F1", inner="stick")
    sns.catplot(data=stats, x="scorer", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)
    #plt.show()


    stats2 = pd.DataFrame(comp_scorers(m=128, Ns=[300, 300, 1_000, 3_000, 10_000, 30_000, 100_000], k=4, n_trees=6, obs_per_tree=6, scorers=[score_plain, score_sum]))
    sns.catplot(data=stats2, x="scorer", y="F1%", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)
