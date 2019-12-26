import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

import NoahClade
from baselines import NJ_hamming, NJ_JC
from reconstruct_tree import estimate_tree_topology_multiclass, estimate_tree_topology_Jukes_Cantor
estimate_tree_topology_multiclass.__name__ = 'spectral'
estimate_tree_topology_Jukes_Cantor.__name__ = 'spectral_JC'

def score_sum(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    return s[1:].sum()

def comp_scorers(m, Ns, k, scorers, n_trees=12, obs_per_tree=1, proba_bounds=(0.75, 0.95), std_bounds=(0.1, 0.3), baselines=[], discrete=True):
    # Ns is array of n
    if discrete:
        transition_maker = NoahClade.NoahClade.gen_symmetric_transition
        #transition_maker = NoahClade.NoahClade.jukes_cantor_transition
    else:
        transition_maker = NoahClade.NoahClade.gen_linear_transition
    stats = []
    itr = 1
    total_iters = n_trees*obs_per_tree*len(Ns)*(len(scorers)+len(baselines))

    for _ in range(n_trees):
        if discrete:
            ref_tree = NoahClade.random_discrete_tree(m, 1, k, proba_bounds=proba_bounds)
        else:
            ref_tree = NoahClade.random_gaussian_tree(m, 1, std_bounds=std_bounds)
        for _ in range(obs_per_tree):
            for n in Ns:
                root_data = np.random.choice(a=k, size=n)
                ref_tree.root.gen_subtree_data(root_data, transition_maker, proba_bounds=proba_bounds, std_bounds=std_bounds)
                observations, labels = ref_tree.root.observe()
                #print("ref")
                #ref_tree.root.ascii()
                scorer_methods = []
                for scorer in scorers:
                    scorer_method = partial(estimate_tree_topology, scorer=scorer, discrete=discrete)
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

def comp_with_params(m, Ns, k, proba_bounds, n_trees=12, folder="./figures/NJ_JC"):
    stats = pd.DataFrame(comp_scorers(m=m, Ns=Ns, k=k, scorers=[], proba_bounds=proba_bounds, baselines=[NJ_hamming, NJ_JC, estimate_tree_topology_multiclass, estimate_tree_topology_Jukes_Cantor], n_trees=n_trees))
    sns_plot = sns.catplot(data=stats, x="method", y="RF", inner="stick", col="n", bw=.6, scale="count", kind="violin",
        col_wrap=3).set_titles("n = {{col_name}}\nm = {0}, k = {1}, p ∈ {2}".format(m, k, proba_bounds))
    sns_plot.savefig(folder+"/m{0}_k{1}_p{2}-{3}.png".format(m, k, proba_bounds[0], proba_bounds[1]), transparent=False)

if __name__ == "__main__":
    # DONE:

    # comp_with_params(m=64, Ns=[500, 1000], k=4, proba_bounds=(0.85, 0.95))
    # comp_with_params(m=64, Ns=[500, 1000,], k=2, proba_bounds=(0.90, 0.95))
    # comp_with_params(m=16, Ns=[200, 400, 800], k=4, proba_bounds=(0.75, 0.95))
    # comp_with_params(m=32, Ns=[1000, 3000, 10_000], k=4, proba_bounds=(0.75, 0.95))
    # comp_with_params(m=200, Ns=[1000, 3000, 10_000, 30_000], k=4, proba_bounds=(0.75, 0.95))
    # comp_with_params(m=32, Ns=[100, 300, 1000, 3000,], k=4, proba_bounds=(0.75, 0.95))
    #comp_with_params(m=64, Ns=[100, 300, 1000, 3000,], k=20, proba_bounds=(0.75, 0.95))
    #comp_with_params(m=200, Ns=[300, 1000, 3000, 10_000], k=4, proba_bounds=(0.90, 0.95))
    #comp_with_params(m=100, Ns=[300, 1000, 3000,], k=4, proba_bounds=(0.85, 0.95))

    comp_with_params(m=80, Ns=[3_000, 5_500, 10_000, 20_000], k=4, proba_bounds=(0.75, 0.95), n_trees=20, folder="./figures/NJ_JC_grid")

    for m in [200]: #[16, 32, 64, 100, 200]:
        #for p_low in [0.70, 0.75, 0.80]: #[0.70, 0.75, 0.80, 0.85, 0.90]:
        #    comp_with_params(m=m, Ns=[3_000, 5_500, 10_000, 20_000, 40_000], k=4, proba_bounds=(p_low, 0.95), n_trees=20, folder="./figures/NJ_JC_grid")
        for p_low in [0.85, 0.90]:
            comp_with_params(m=m, Ns=[3_000, 5_500, 10_000, 20_000], k=4, proba_bounds=(p_low, 0.95), n_trees=10, folder="./figures/NJ_JC_grid")
