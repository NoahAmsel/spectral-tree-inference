import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import NoahClade

from reconstruct_tree import estimate_tree_topology, similarity_matrix
from NoahClade import random_discrete_tree, random_gaussian_tree
import random

from importlib import reload
#reload(NoahClade)

def score_sum(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    return s[1:].sum()

def score_sum_tracker(A, M):
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0
    m = M.shape[0]
    leafset = tuple(np.nonzero(A)[0])
    s1 = s[0]
    s2 = s[1]
    s23 = s[1:].sum()
    size = min(A.sum(), (~A).sum())
    frob = np.linalg.norm(M_A)
    score_sum_tracker.records.append({"|A|": size, "|A|/|T|":size/m, "m": m, "leafset": leafset, "s1":s1, "s2":s2, "s23":s23, "frob":frob})

    return s[1:].sum()
score_sum_tracker.records = []

# %%
def profile_sig2(discrete=True, k=4):
    reps = 10
    Ms = [50, 100]
    Ns = [1_000, 10_000]
    rows = []
    i = 1
    total_iters = reps*len(Ms)*len(Ns)
    for rep in range(reps):
        for m in Ms:
            if discrete:
                ref_tree = random_discrete_tree(m, 1, k)
                trans = NoahClade.NoahClade.gen_symmetric_transition
            else:
                ref_tree = random_gaussian_tree(m, 1, std_bounds=(0.1, 0.3))
                trans = NoahClade.NoahClade.gen_linear_transition
            #ref_tree.root.ascii()
            for n in Ns:
                if discrete:
                    root_data = np.random.choice(a=k, size=n)
                    ref_tree.root.gen_subtree_data(root_data, trans, proba_bounds=(0.75, 0.95))
                else:
                    root_data = np.random.uniform(0, 1, n)
                    ref_tree.root.gen_subtree_data(root_data, trans, std_bounds=(0.1, 0.3))
                obs, labels = ref_tree.root.observe()
                ref_tree.root.reset_taxasets(labels)
                if discrete:
                    M = similarity_matrix(obs)
                else:
                    M = np.corrcoef(obs)
                non_terms = ref_tree.get_nonterminals()

                splits = []
                for node in non_terms:
                    # loop thru all the real splits
                    splits.append((node.taxa_set, True))
                    for _ in range(2):
                        other = random.choice(non_terms)
                        # loop thru some fake ones
                        bad_A = node.taxa_set ^ other.taxa_set
                        splits.append((bad_A, False))

                for A, real in splits:
                    M_A = M[np.ix_(A, ~A)]
                    if M_A.shape[0] > 1 and M_A.shape[1] > 1:
                        s = np.linalg.svd(M_A, compute_uv=False)
                        s1 = s[0]
                        s2 = s[1]
                        s23 = s[1:].sum()
                        frob = np.linalg.norm(M_A)
                        size = min(A.sum(), (~A).sum())#A.sum()#min(A.sum(), (~A).sum())
                        rows.append({"|A|": size, "|A|/|T|": size/m, "n":n, "m":m, "mn":"{0},{1}".format(m,n), "s1":s1, "s2":s2, "s23":s23, "frob":frob, "real":real})
                print("{0} / {1}".format(i, total_iters))
                i += 1
    return rows

# %%

def stat_comp(dataset, x, y, mm=100, nn=10_000, order=1):
    data_sub = dataset[(dataset['m'] == mm) & (dataset['n'] == nn)]
    assert len(data_sub) > 0

    print("="*40)
    print("="*40)
    sns.relplot(x=x, y=y, col="mn", hue='real', data=dataset, alpha=.3, col_wrap=2)
    plt.figure()
    sns.lmplot(x=x, y=y, col="mn", hue='real', data=dataset, col_wrap=2, scatter_kws={"alpha":.3}, order=order)
    plt.figure()
    sns.lmplot(x=x, y=y, hue='real', data=data_sub, order=order)
    plt.figure()
    sns.regplot(x=x, y=y, data=data_sub[data_sub['real']], scatter_kws={"alpha":.3}, color='b', order=order)
    plt.figure()
    sns.regplot(x=x, y=y, data=data_sub[~data_sub['real']], scatter_kws={"alpha":.3}, color='orange', order=order)
    plt.figure()

    means = data_sub.groupby(['|A|', "real"]).mean().reset_index()
    sns.regplot(x=x, y=y, data=means[means['real']], scatter_kws={"alpha":.3}, color='b', order=order)
    plt.figure()
    sns.regplot(x=x, y=y, data=means[~means['real']], scatter_kws={"alpha":.3}, color='orange', order=order)
    plt.figure()

# %%

data = pd.DataFrame(profile_sig2(False))
data.sample(10)

data['s1^2'] = data['s1']**2
data['s1x10^4'] = data['s1']*(1e4)
data['s2x10^4'] = data['s2']*(1e4)

stat_comp(data, x='|A|/|T|', y='s2x10^4')

stat_comp(data, x='|A|/|T|', y='s1x10^4')

stat_comp(data, x='|A|/|T|', y='s1x10^4', order=2)

# WHY DO THESE LOOK BAD? Oh you hae to just multiply them by a large number for them to show up

# %%
# OLD
data_sub = data[(data['m'] == 100) & (data['n'] == 10_000)]
#data_sub['s2/'] = data_sub['s2']/np.sqrt(data_sub['|A|/|T|']*(1-data_sub['|A|/|T|']))

data_sub_real = data_sub[data_sub['real']]
data_sub_fake = data_sub[~data_sub['real']]

sns.regplot(x='|A|', y='s23', data=data_sub[data_sub['real']], scatter_kws={"alpha":.3})
sns.regplot(x='|A|', y='s2', data=data_sub[~data_sub['real']], scatter_kws={"alpha":.3})

sns.regplot(x='|A|', y='s23', data=data_sub_real.groupby('|A|').mean().reset_index(inplace=False), scatter_kws={"alpha":.3})
sns.regplot(x='|A|', y='s2', data=data_sub_fake.groupby('|A|').mean().reset_index(inplace=False), scatter_kws={"alpha":.3})

sns.relplot(x="frob", y="s2", col="mn", hue='real', data=data, alpha=.4, col_wrap=2)
sns.regplot(x='|A|', y='s2', data=data.groupby('|A|').mean().reset_index(inplace=False))

sns.regplot(x='|A|/|T|', y='s1', data=data_sub[data_sub['real']], scatter_kws={"alpha":.3})
xx = np.linspace(0,140)
plt.scatter(data_sub[data_sub['real']]['|A|'], data_sub[data_sub['real']]['s1'], alpha=.3)
#plt.plot(xx, 30*(xx**0.25))
plt.plot(xx, 10*np.sqrt(xx))
sns.regplot(x='|A|/|T|', y='s1', data=data_sub[~data_sub['real']], scatter_kws={"alpha":.3})

sns.relplot(x="|A|/|T|", y="s1", col="mn", hue='real', data=data, alpha=.3, col_wrap=2)
# %%


# How does distance on the tree affect similarity?
from itertools import combinations, product

ref_tree = random_discrete_tree(64, 10_000, 4)
#trans = NoahClade.NoahClade.gen_symmetric_transition
#root_data = np.random.choice(a=4, size=n)
#ref_tree.root.gen_subtree_data(root_data, trans, proba_bounds=(0.75, 0.95))
ref_tree.root.ascii()
obs, labels = ref_tree.root.observe()
M = similarity_matrix(obs)

random_gaussian_tree(64, 10_000, std_bounds=(0.1, 0.3))
obs, labels = ref_tree.root.observe()
M = np.corrcoef(obs)

RRROW = []
for left_ix, right_ix in combinations(range(len(labels)), 2):
    dist = ref_tree.distance(labels[left_ix], labels[right_ix])
    RRROW.append({"dist":dist, "sim": M[left_ix, right_ix]})
sns.regplot(x='dist', y='sim', data=pd.DataFrame(RRROW))

# %%
def moves_away(split, good_splits):
    return min(len(split ^ good)/len(split) for good in good_splits)

def profile_me(discrete=True, k=4):
    reps = 10
    Ms = [50, 100]
    Ns = [1_000, 10_000]
    records = []
    i = 1
    total_iters = reps*len(Ms)*len(Ns)
    for rep in range(reps):
        for m in Ms:
            if discrete:
                ref_tree = random_discrete_tree(m, 1, k)
                trans = NoahClade.NoahClade.gen_symmetric_transition
            else:
                ref_tree = random_gaussian_tree(m, 1, std_bounds=(0.1, 0.3))
                trans = NoahClade.NoahClade.gen_linear_transition
            #ref_tree.root.ascii()
            for n in Ns:
                if discrete:
                    root_data = np.random.choice(a=k, size=n)
                    ref_tree.root.gen_subtree_data(root_data, trans, proba_bounds=(0.75, 0.95))
                else:
                    root_data = np.random.uniform(0, 1, n)
                    ref_tree.root.gen_subtree_data(root_data, trans, std_bounds=(0.1, 0.3))
                obs, labels = ref_tree.root.observe()
                ref_tree.root.reset_taxasets(labels)


                score_sum_tracker.records = []
                estimate_tree_topology(obs, labels=labels, scorer=score_sum_tracker, discrete=discrete)
                good_splits = ref_tree.root.find_splits(branch_lengths=False)
                good_splits_sets = [set(good) for good in good_splits]
                for record in score_sum_tracker.records:
                    record['real'] = record['leafset'] in good_splits
                    record['moves_away'] = moves_away(set(record['leafset']), good_splits_sets)
                    record['n'] = n
                records += score_sum_tracker.records
                print("{0} / {1}".format(i, total_iters))
                i += 1
    return records

# %%
data = pd.DataFrame(profile_me(False))
# LINEAR = data
# DISCRETE = data
data['mn'] = data['m'].astype('str').str.cat(data['n'].astype('str'), sep=",")

data.sample(10)

data['s1^2'] = data['s1']**2
data['s1x10^4'] = data['s1']*(1e4)
data['s2x10^4'] = data['s2']*(1e4)

stat_comp(data, x='|A|/|T|', y='s2x10^4', order=1)
stat_comp(data, x='|A|/|T|', y='s1x10^4', order=2)

stat_comp(data, x='moves_away', y='s2x10^4', order=1)

stat_comp(data, x='moves_away', y='s1x10^4', order=1)

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

records['moves_away'] = records.apply(lambda row: moves_away(set(row['leafset']), [set(good) for good in good_splits]), axis=1)

sns.relplot(x="|A|/|T|", y="moves_away", hue='real', data=records)
sns.relplot(x="moves_away", y="s2", hue='real', data=records, alpha=.3)

sns.regplot(x='|A|/|T|', y='s2', data=records[records['real']], scatter_kws={"alpha":.3})
sns.regplot(x='|A|/|T|', y='s2', data=records[~records['real']], scatter_kws={"alpha":.3})
records.sample(10)
# %%

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
