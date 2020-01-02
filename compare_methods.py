import datetime
import pickle
import os.path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import dendropy

import spectraltree

class Experiment_Datum:
    def __init__(self, sequence_model, n, method, mutation_rate, inferred_tree, reference_tree):
        self.sequence_model = sequence_model
        self.n = n
        self.method = method
        self.mutation_rate = mutation_rate
        inferred_tree.update_bipartitions()
        reference_tree.update_bipartitions()
        self.ref_tree = reference_tree
        # self.inf_tree = inferred_tree   # take this out for space/speed
        self.false_positives, self.false_negatives = dendropy.calculate.treecompare.false_positives_and_negatives(reference_tree, inferred_tree, is_bipartitions_updated=True)
        self.total_reference = len(reference_tree.bipartition_encoding)
        self.total_inferred = len(inferred_tree.bipartition_encoding)
        self.timestamp = datetime.datetime.now()

    @property
    def m(self):
        return len(self.ref_tree.leaf_edges())

    @property
    def RF(self):
        # source: https://dendropy.org/_modules/dendropy/calculate/treecompare.html#symmetric_difference
        return self.false_positives + self.false_negatives

    @property
    def precision(self):
        true_positives = self.total_inferred - self.false_positives
        return true_positives / self.total_inferred

    @property
    def recall(self):
        true_positives = self.total_inferred - self.false_positives
        return true_positives / self.total_reference

    @property
    def F1(self):
        return 2*(self.precision * self.recall)/(self.precision + self.recall)

    @property
    def correct(self):
        return self.RF == 0

    def to_row(self):
        return {
            "n": self.n,
            "m": self.m,
            "correct": self.correct,
            "F1%": 100*self.F1,
            "method": str(self.method),
        }

def save_results(results, filename=None, folder="data"):
    if filename is None:
        filename = datetime.datetime.now().strftime("results_%b%d_%-I:%-M%p.pkl")
    if folder:
        filename = os.path.join(folder, filename)
    with open(filename, "wb") as f:
        pickle.dump(results, f)

def load_results(*files, folder="data", throw_error=True):
    total_results = []
    successful = 0
    for filename in files:
        successful += 1
        fullpath = filename if folder is None else os.path.join(folder, filename)
        try:
            with open(fullpath, "rb") as f:
                res = pickle.load(f)
                total_results += res
        except Exception as error:
            successful -= 1
            if throw_error:
                raise error
    if not throw_error:
        print("Successfully read {} files.".format(successful))
    return total_results

def experiment(tree_list, sequence_model, Ns, methods, mutation_rates=[1.], reps_per_tree=1, savepath=None, folder="data"):

    print("==== Beginning Experiment =====")
    print("\t Transition: ", sequence_model)
    print("\t {} trees".format(len(tree_list)))
    print("\t {} sample sizes:".format(len(Ns)), *Ns)
    print("\t {} methods".format(len(methods)), *methods)
    print("\t {} mutation rates:".format(len(mutation_rates)), *("{0:.4f}".format(rate) for rate in mutation_rates))
    print("\t {} reps".format(reps_per_tree))

    results = []
    total_trials = len(tree_list) * reps_per_tree * len(mutation_rates) * len(Ns) * len(methods)
    i = 0
    for reference_tree in tree_list:
        for _ in range(reps_per_tree):
            for mutation_rate in mutation_rates:
                observations = spectraltree.simulate_sequences(seq_len=max(Ns), tree_model=reference_tree, seq_model=sequence_model, mutation_rate=mutation_rate)
                for n in Ns:
                    for method in methods:
                        inferred_tree = method(observations[:,:n], namespace=reference_tree.taxon_namespace)
                        results.append(Experiment_Datum(sequence_model, n, method, mutation_rate, inferred_tree, reference_tree))
                        i += 1
                        print("{0} / {1}".format(i, total_trials))

    if savepath:
        previous_results = load_results(savepath, folder=folder, throw_error=False)
        save_results(previous_results+results, filename=savepath, folder=folder)
        print("Saved to", os.path.join(folder, savepath) if folder else savepath)

    return results

# Plotting functions
def results2frame(results):
    return pd.DataFrame(result.to_row() for result in results)

def violin(result_frame, x="n", y="RF", hue="method"):
    sns.catplot(data=result_frame, x=hue, y=y, inner="stick", col=x, bw=.8, scale="count", kind="violin", col_wrap=3)

def box(result_frame, x="n", y="RF", hue="method"):
    sns.boxplot(data=result_frame, x=x, y=y, hue=hue)

# TODO: change RF to F1 something normalized like F1
def accuracy(result_frame, x="n", y="F1%", hue="method"):
    dodge = 0.1*(df['method'].nunique() - 1)
    sns.pointplot(data=result_frame, x=x, y=y, hue=hue, dodge=dodge)

def correct(result_frame, x="n", y="correct", hue="method"):
    dodge = 0.1*(df['method'].nunique() - 1)
    sns.pointplot(data=result_frame, x=x, y=y, hue=hue, dodge=dodge)

# %%
def weird1(A1, A2, M):
    A = A1 | A2
    M1 = M[np.ix_(A1, ~A)]
    M2 = M[np.ix_(A2, ~A)]
    U1, S1, Vh1 = np.linalg.svd(M1, compute_uv=True)
    vh1 = Vh1[1,:]
    U2, S2, Vh2 = np.linalg.svd(M2, compute_uv=True)
    vh2 = Vh2[1,:]
    return 1. - np.inner(vh1/np.linalg.norm(vh1), vh2/np.linalg.norm(vh2))

def weird2(A1, A2, M):
    A = A1 | A2
    M1 = M[np.ix_(A1, ~A)]
    M2 = M[np.ix_(A2, ~A)]

    M1 = M1/np.linalg.norm(M1, axis=1, keepdims=True)
    M2 = M2/np.linalg.norm(M2, axis=1, keepdims=True)
    return -np.linalg.norm(M1.dot(M2.T))/np.sqrt(len(A1)*len(A2))

# %%
if __name__ == "__main__":
    #t = utils.balanced_binary(128)
    trees = dendropy.TreeList([utils.balanced_binary(128), utils.lopsided_tree(128)])
    jc_model = dendropy.model.discrete.Jc69()
    seqgen = partial(dendropy.model.discrete.simulate_discrete_chars, seq_model=jc_model, mutation_rate=0.1)
    Ns = np.geomspace(100, 1_000, num=4).astype(int)
    methods = [Reconstruction_Method(reconstruct_tree.neighbor_joining), Reconstruction_Method(), Reconstruction_Method(scorer=reconstruct_tree.sum_squared_quartets), Reconstruction_Method(scorer=weird2)]
    results = experiment(trees, seqgen, Ns=Ns, methods=methods)
    df = results2frame(results)
    correct(df)
    accuracy(df)

    """
    import cProfile
    cProfile.run("results = experiment(trees, seqgen, Ns=Ns, methods=methods)")
    """

    """mat = dendropy.model.discrete.simulate_discrete_chars(1000, trees[0], my_model)
    observations, _ = utils.charmatrix2array(mat)
    observations.shape"""

# %%
if __name__ == "__main__":
    JC = generation.Jukes_Cantor(4)
    trees = dendropy.TreeList([utils.balanced_binary(num_taxa=128, edge_length=JC.paralinear2t(delta)) for delta in [0.55, 0.6, 0.65, 0.7]])
    jc_model = dendropy.model.discrete.Jc69()
    seqgen = partial(dendropy.model.discrete.simulate_discrete_chars, seq_model=jc_model, mutation_rate=1.)
    Ns = [500]
    methods = [Reconstruction_Method(reconstruct_tree.neighbor_joining), Reconstruction_Method()]
    results = experiment(trees, seqgen, Ns=Ns, methods=methods, reps_per_tree=100)
    df = results2frame(results)
    correct(df)
    accuracy(df)

    len(results)

    with open("data/results.pkl", "wb") as f:
        pickle.dump(results, f)

    df.to_pickle("data/df.pkl")

    f = open("data/results.pkl", "rb")
    ppp = pickle.load(f)
    ppp
    f.close()
# %%

if __name__ == "__main__":
    import pstats
    p = pstats.Stats('profile.txt')
    p.sort_stats('cumulative').print_stats(500)
