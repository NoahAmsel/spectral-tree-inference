import numpy as np
import pandas as pd
import seaborn as sns

import Bio.Phylo as Phylo

import NoahClade
from reconstruct_tree import estimate_tree_topology_Jukes_Cantor
#from scoring_experiments import violin, box, lines

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
        m = 2**depth
        for _ in range(n_trees):
            if discrete:
                ref_tree = NoahClade.NoahClade.complete_binary(depth, proba_bounds=proba_bounds, n=1, k=k)
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
                stats.append({"n": n, "m": m, "method": method.__name__, "F1%":100*F1, "precision%":100*precision, "recall%":100*recall, "RF":RF, "correct":int(RF==0)})
                print("{0} / {1}\t{2}\tRF:{3}\tF1 {4:.1f}%\tn {5}".format(itr, total_iters, 2**depth, RF, 100*F1, n))
                itr += 1
    if oldstats is None:
        return pd.DataFrame(stats)
    else:
        return oldstats.append(pd.DataFrame(stats))

def clean_stats(data, Ns=None, Ms=None):
    if Ns is not None:
        data = data[data['n'].isin(Ns)]
    if Ms is not None:
        data = data[data['m'].isin(Ms)]
    return data

#stats['m'] = [16]*60 + [32]*60 + [64]*60 + [128]*60 + [256]*60

# %%
if __name__ == "__main__":
    Ns = [int(100*(np.sqrt(10)**j)) for j in range(6)]
    stats = comp_depths(depths=range(4, 9), Ns=Ns, method=estimate_tree_topology_Jukes_Cantor, k=4, proba_bounds=(0.85, 0.95), n_trees=10, oldstats=stats)
    #stats.to_csv("depth_exp.csv")
    #stats = pd.read_csv("depth_exp.csv")

    stats2 = comp_depths(depths=[6,7], Ns=[40_000, 50_000], method=estimate_tree_topology_Jukes_Cantor, k=4, proba_bounds=(0.85, 0.95), n_trees=15)
    stats2 = comp_depths(depths=[6,7], Ns=[40_000], method=estimate_tree_topology_Jukes_Cantor, k=4, proba_bounds=(0.85, 0.95), n_trees=10)
    meee = stats2.groupby(['m','n'])['RF', 'correct'].mean().reset_index()
    

    Ms=None
    sns.boxplot(data=clean_stats(stats, Ns=Ns, Ms=Ms), x="n", y="RF", hue='m')
    sns.catplot(data=clean_stats(stats, Ns=Ns, Ms=Ms), x="n", y="RF", inner="stick", col="m", bw=.8, scale="count", kind="violin", col_wrap=3)
    sns.pointplot(data=clean_stats(stats, Ns=Ns, Ms=Ms), x="n", y="RF", hue='m', dodge=0.2)

    sns.pointplot(data=stats, x='n', y='correct', hue='m')
    sns.lineplot(data=stats, x='n', y='correct', hue='m', err_style='bars', )
    sns.lineplot(data=stats, x='n', y='correct', hue='m', err_style=None)

    meee = stats.groupby(['m','n'])['RF', 'correct'].mean().reset_index()
    stats.groupby(['m','n'])['correct'].count()
    sns.lineplot(data=stats, x='n', y='correct', hue='m')
    sns.lineplot(data=meee, x='n', y='correct', hue='m')
    sns.pointplot(data=meee, x='n', y='correct', hue='m', dodge=0.4)

    meee['est'] = np.log(meee['m'])**2 / np.log(meee['n'])**2
    meee['est'] = np.log(meee['m']) / np.log(meee['n'])
    meee['est'] = np.log(meee['m']) / np.sqrt(np.log(meee['n']))
    meee['est'] = meee['m'] / meee['n']
    sns.scatterplot(x='RF', y='est', data=meee[meee['RF']<100])
