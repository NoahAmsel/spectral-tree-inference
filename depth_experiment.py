from itertools import zip_longest
import numpy as np
import pandas as pd
import seaborn as sns

import Bio.Phylo as Phylo

import NoahClade
from reconstruct_tree import estimate_tree_topology_Jukes_Cantor
#from scoring_experiments import violin, box, lines

def complete_binary(depth, proba_bounds, k=4, n=1):
    leaves = [NoahClade.NoahClade(name="_"+str(i)) for i in range(2**depth)]
    while len(leaves)>1:
        halfway = int(len(leaves)/2)
        leaves = [NoahClade.NoahClade(clades=[left, right]) for left, right in zip_longest(leaves[:halfway], leaves[halfway:])]
    root = leaves[0]
    root.reset_taxasets()

    root_data = np.random.choice(a=k, size=n)
    transition_maker = NoahClade.NoahClade.gen_symmetric_transition
    root.gen_subtree_data(root_data, transition_maker, num_classes=k, proba_bounds=proba_bounds)
    return Phylo.BaseTree.Tree.from_clade(root)

def comp_depths(depths, Ns, method, k, n_trees=12, proba_bounds=(0.75, 0.95), std_bounds=(0.1, 0.3), discrete=True, oldstats=None):
    # Ns is array of n
    if discrete:
        transition_maker = NoahClade.NoahClade.gen_symmetric_transition
        #transition_maker = NoahClade.NoahClade.jukes_cantor_transition
    else:
        transition_maker = NoahClade.NoahClade.gen_linear_transition
    stats = []
    itr = 1
    total_iters = len(depths)*n_trees*len(Ns)

    for depth in depths:
        for _ in range(n_trees):
            if discrete:
                ref_tree = complete_binary(depth, proba_bounds=proba_bounds, n=1, k=k)
            else:
                assert(False)
                ref_tree = NoahClade.NoahClade.random_gaussian_tree(m, 1, std_bounds=std_bounds)

            for n in Ns:
                root_data = np.random.choice(a=k, size=n)
                ref_tree.root.gen_subtree_data(root_data, transition_maker, proba_bounds=proba_bounds, std_bounds=std_bounds)
                observations, labels = ref_tree.root.observe()

                inferred_tree = method(observations, labels=labels)
                #inferred_tree.root.ascii()
                F1, precision, recall, RF = NoahClade.tree_Fscore(inferred_tree, ref_tree)
                stats.append({"n": n, "method": method.__name__, "F1%":100*F1, "precision%":100*precision, "recall%":100*recall, "RF":RF,})
                print("{0} / {1}\t{2}\tRF:{3}\tF1 {4:.1f}%\tn {5}".format(itr, total_iters, 2**depth, RF, 100*F1, n))
                itr += 1
    if oldstats is None:
        return stats
    else:
        return oldstats.append(pd.DataFrame(stats))

if __name__ == "__main__":
    stats = pd.DataFrame(comp_depths(depths=range(4, 9), Ns=[int(100*(np.sqrt(10)**j)) for j in range(6)], method=estimate_tree_topology_Jukes_Cantor, k=4, n_trees=10))

    sns.boxplot(data=clean_stats(data, Ns=Ns), x="m", y="RF", hue='n')
    sns.catplot(data=clean_stats(data, Ns=Ns), x="m", y="RF", inner="stick", col="n", bw=.8, scale="count", kind="violin", col_wrap=3)
    sns.pointplot(data=clean_stats(data, Ns=Ns), x="m", y="RF", hue='n', dodge=0.2)


if __name__ == "__main__":
