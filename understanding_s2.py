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

# %%
import random
def profile_sig2():
    Ms = [100, 300]
    Ns = [10_000, 100_000]
    rows = []
    for m in Ms:
        ref_tree = random_gaussian_tree(m, 1, std_bounds=(0.1, 0.3))
        #ref_tree = random_discrete_tree(m, n, 2)
        #ref_tree.root.ascii()
        for n in Ns:
            ref_tree.root.gen_subtree_data(np.random.uniform(0, 1, n), NoahClade.NoahClade.gen_linear_transition, std_bounds=(0.1, 0.3))
            obs, labels = ref_tree.root.observe()
            ref_tree.root.reset_taxasets(labels)
            M = np.corrcoef(obs)
            non_terms = ref_tree.get_nonterminals()
            for node in non_terms:
                A = node.taxa_set
                M_A = M[np.ix_(A, ~A)]
                if M_A.shape[0] > 1 and M_A.shape[1] > 1:
                    s = np.linalg.svd(M_A, compute_uv=False)
                    s1 = s[0]
                    s2 = s[1]
                    s23 = s[1:].sum()
                    size = min(A.sum(), (~A).sum())
                    rows.append({"|A|": size, "|A|/|T|": size/m, "n":n, "m":m, "mn":"{0},{1}".format(m,n), "s1":s1, "s2":s2, "s23":s23, "real":True})
                else:
                    s1 = np.linalg.norm(M_A)
                    s2 = 0
            for node in non_terms:
                other = random.choice(non_terms)
                bad_A = node.taxa_set ^ other.taxa_set
                M_A = M[np.ix_(bad_A, ~bad_A)]
                if M_A.shape[0] > 1 and M_A.shape[1] > 1:
                    s = np.linalg.svd(M_A, compute_uv=False)
                    s1 = s[0]
                    s2 = s[1]
                    s23 = s[1:].sum()
                    size = min(bad_A.sum(), (~bad_A).sum())
                    rows.append({"|A|": size, "|A|/|T|": size/m, "n":n, "m":m, "mn":"{0},{1}".format(m,n), "s1":s1, "s2":s2, "s23":s23, "real":False})
                else:
                    s1 = np.linalg.norm(M_A)
                    s2 = 0
    return pd.DataFrame(rows)

data = profile_sig2()
data.sample(10)

data_sub = data[(data['m'] == 300) & (data['n'] == 100_000)]

sns.regplot(x='|A|/|T|', y='s2', data=data_sub[data_sub['real']], scatter_kws={"alpha":.3})

sns.regplot(x='|A|/|T|', y='s2', data=data_sub[~data_sub['real']], scatter_kws={"alpha":.3})

sns.relplot(x="|A|/|T|", y="s2", col="mn", hue='real', data=data, alpha=.4, col_wrap=2)


sns.regplot(x='|A|/|T|', y='s1', data=data_sub[data_sub['real']], scatter_kws={"alpha":.3})
xx = np.linspace(0,140)
plt.scatter(data_sub[data_sub['real']]['|A|'], data_sub[data_sub['real']]['s1'], alpha=.3)
#plt.plot(xx, 30*(xx**0.25))
plt.plot(xx, 10*np.sqrt(xx))
sns.regplot(x='|A|/|T|', y='s1', data=data_sub[~data_sub['real']], scatter_kws={"alpha":.3})

sns.relplot(x="|A|/|T|", y="s1", col="mn", hue='real', data=data, alpha=.3, col_wrap=2)

# %%
score_sum_tracker.records = []
ref_tree = random_gaussian_tree(100, 10_000, std_bounds=(0.1, 0.3))
obs, labs = ref_tree.root.observe()

estimate_tree_topology(obs, labels=labs, scorer=score_sum_tracker, discrete=False)
records = pd.DataFrame(score_sum_tracker.records)
ref_tree.root.reset_taxasets(labels=labs)
good_splits = ref_tree.root.find_splits(branch_lengths=False)
records['real'] = records['leafset'].isin(good_splits)
records.sample(10)

def moves_away(split, good_splits):
    return min(len(split ^ good)/len(split) for good in good_splits)
records['moves_away'] = records.apply(lambda row: moves_away(set(row['leafset']), [set(good) for good in good_splits]), axis=1)

sns.relplot(x="|A|/|T|", y="moves_away", hue='real', data=records)

sns.relplot(x="moves_away", y="s2", hue='real', data=records, alpha=.3)

sns.regplot(x='|A|/|T|', y='s2', data=records[records['real']], scatter_kws={"alpha":.3})
sns.regplot(x='|A|/|T|', y='s2', data=records[~records['real']], scatter_kws={"alpha":.3})
records.sample(10)
# %%

if __name__ == "__main__":
    # %%

    m = 64
    n = 20_000
    k = 4
    ref_tree = random_discrete_tree(m, n, k, proba_bounds=(0.8, 0.9))
    ref_tree.root.ascii()
    obs, labels = ref_tree.root.observe()
    sim = similarity_matrix(obs)
    #S = ['taxon32', 'taxon18']
    #Sc = ['taxon32', 'taxon18']
    #A = ref_tree.root.labels2taxaset(S)
    #Ac = ref_tree.root.labels2taxaset(Sc)

    # taxon12, taxon5, taxon56 taxon26
    R_S = sim[(11,4),:][:,(55,25)]

    #R_S = sim[(25,4),:][:,(56,53)]
    #R_S = sim[(25,4),:][:,(56,53)]
    s1, s2 = np.linalg.svd(R_S, compute_uv=False)
    print("="*30)
    print("\u03C31 \t", s1)
    print("\u03C32 \t", s2)
    print("||R^(S)||_F \t", np.linalg.norm(R_S))
    print("\u03C31^2 - \u03C32^2 \t", s1**2 - s2**2)
    print("det(R^(S)) \t", np.linalg.det(R_S))
    print("(\u03C31^2 - \u03C32^2)^2 \t", (s1**2 - s2**2)**2)
    print("||R^(S)||_F^4 - 4det(R^(S)) \t",np.linalg.norm(R_S)**4 - 4*(np.linalg.det(R_S)**2))
