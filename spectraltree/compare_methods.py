import datetime
from itertools import product
import multiprocessing
import os.path
import pickle
import time

import dendropy
import numpy as np
import pandas as pd
import seaborn as sns

from . import utils
from . import generation
from . import reconstruct_tree

class Experiment_Datum:
    def __init__(self, sequence_model, n, method, mutation_rate, inferred_tree, reference_tree, run_time):
        inferred_tree.update_bipartitions()
        if inferred_tree.is_rooted != reference_tree.is_rooted:
            raise ValueError("Cannot compare rooted to unrooted tree.")
        self.sequence_model = sequence_model
        self.n = n
        self.method = method
        self.run_time = run_time
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
            "transition": self.sequence_model,
            "n": self.n,
            "method": str(self.method),
            "rate": self.mutation_rate,
            "m": self.m,
            "correct": self.correct,
            "F1%": 100*self.F1,
            "RF": self.RF,
            "runTime": self.run_time,
        }

    def __str__(self):
        first_part = "Method={}\tn={}\ttransition={}".format(self.method, self.n, self.sequence_model)
        rate_part = ("" if self.mutation_rate==1. else "mut_rate={:.2f}".format(self.mutation_rate))
        last_part = "m={}\tF1={:.1f}%".format(self.m, 100*self.F1)
        return "\t".join([first_part, rate_part, last_part])

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

def experiment(tree_list, sequence_model, Ns, methods, mutation_rates=[1.], reps_per_tree=1, savepath=None, folder="data", overwrite=False, alphabet="DNA", verbose=True):

    if verbose:
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
    if savepath:
        previous_results = [] if overwrite else load_results(savepath, folder=folder, throw_error=False)

    for reference_tree in tree_list:
        for mutation_rate in mutation_rates:
            # we can parallelize this loop too, but then we need to set different random seeds
            for _ in range(reps_per_tree):
                #observations = generation.simulate_sequences(seq_len=max(Ns), tree_model=reference_tree, seq_model=sequence_model, mutation_rate=mutation_rate)
                observations, taxa_meta = generation.simulate_sequences(max(Ns), tree_model=reference_tree, seq_model=sequence_model, mutation_rate=mutation_rate, alphabet=alphabet)
                for n in Ns:
                    for method in methods:
                        t1 = time.time()
                        inferred_tree = method(observations[:,:n], taxa_meta)
                        run_time = time.time() - t1
                        results.append(Experiment_Datum(sequence_model, n, method, mutation_rate, inferred_tree, reference_tree,run_time))
                        i += 1
                        if verbose:
                            print("{0} / {1}".format(i, total_trials))
                    if savepath:
                        save_results(previous_results+results, filename=savepath, folder=folder)

    if savepath:
        save_results(previous_results+results, filename=savepath, folder=folder)
        if verbose:
            print("Saved to", os.path.join(folder, savepath) if folder else savepath)

    return results

def reproduce_datum(datum):
    observations = utils.simulate_sequences(seq_len=datum.n, tree_model=datum.ref_tree, seq_model=datum.sequence_model, mutation_rate=datum.mutation_rate)
    inferred_tree = datum.method(observations[:,:datum.n], taxon_namespace=datum.ref_tree.taxon_namespace)
    return Experiment_Datum(datum.sequence_model, datum.n, datum.method, datum.mutation_rate, inferred_tree, datum.ref_tree), inferred_tree
    #return experiment(tree_list=[datum.ref_tree], sequence_model=datum.sequence_model, Ns=[datum.n], methods=[datum.method], mutation_rates=[datum.mutation_rate], verbose=False)[0]

# %%

def parallel_helper(reference_tree, mutation_rate, sequence_model, Ns, methods, reps_per_tree):
    partial_results = []
    for _ in range(reps_per_tree):
        observations = utils.simulate_sequences(seq_len=max(Ns), tree_model=reference_tree, seq_model=sequence_model, mutation_rate=mutation_rate)
        for n in Ns:
            for method in methods:
                inferred_tree = method(observations[:,:n], taxon_namespace=reference_tree.taxon_namespace)
                partial_results.append(Experiment_Datum(sequence_model, n, method, mutation_rate, inferred_tree, reference_tree))

    return partial_results

def parallel_experiment(tree_list, sequence_model, Ns, methods, mutation_rates=[1.], reps_per_tree=1, savepath=None, folder="data", overwrite=False, verbose=True):
    if verbose:
        print("==== Beginning Experiment =====")
        print("\t Transition: ", sequence_model)
        print("\t {} trees".format(len(tree_list)))
        print("\t {} sample sizes:".format(len(Ns)), *Ns)
        print("\t {} methods".format(len(methods)), *methods)
        print("\t {} mutation rates:".format(len(mutation_rates)), *("{0:.4f}".format(rate) for rate in mutation_rates))
        print("\t {} reps".format(reps_per_tree))

    total_trials = len(tree_list) * reps_per_tree * len(mutation_rates) * len(Ns) * len(methods)
    with multiprocessing.Pool() as pool:
        params = [(reference_tree, mutation_rate, sequence_model, Ns, methods, reps_per_tree) for reference_tree, mutation_rate in product(tree_list, mutation_rates)]
        result_lists = pool.starmap(parallel_helper, params)

    results = sum(result_lists, [])

    if savepath:
        previous_results = [] if overwrite else load_results(savepath, folder=folder, throw_error=False)
        save_results(previous_results+results, filename=savepath, folder=folder)
        if verbose:
            print("Saved to", os.path.join(folder, savepath) if folder else savepath)

    return results

# Plotting functions
def results2frame(results):
    return pd.DataFrame(result.to_row() for result in results)

def violin(result_frame, x="n", y="F1%", hue="method"):
    sns.catplot(data=result_frame, x=hue, y=y, inner="stick", col=x, bw=.8, scale="count", kind="violin", col_wrap=3)

def accuracy(result_frame, x="n", y="F1%", hue="method", col=None, kind="point"):
    grouping_cols = x if col is None else [col, x]
    print(result_frame.groupby(grouping_cols)[y].count()) # .describe()
    dodge = 0.1*(result_frame['method'].nunique() - 1)
    #d = {'color': ['C0', 'k'], "ls" : ["-","--"]}
    return sns.catplot(data=result_frame, x=x, y=y, kind="point", hue=hue, col=col, dodge=dodge, col_wrap=(None if col is None else 3),\
        legend=False)

def correct(result_frame, x="n", y="correct", hue="method", col=None):
    return accuracy(result_frame, x=x, y=y, hue=hue, col=col)
    #dodge = 0.1*(result_frame['method'].nunique() - 1)
    #sns.pointplot(data=result_frame, x=x, y=y, hue=hue, dodge=dodge)

def box(result_frame, x="n", y="F1%", hue="method"):
    return accuracy(result_frame, x=x, y=y, hue=hue, col=col, kind="box")

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

    binary_trees = [utils.balanced_binary(m) for m in [64, 128]]
    lopsided = [utils.lopsided_tree(m) for m in [64, 128]]
    jc = generation.Jukes_Cantor()
    Ns = [100, 400, 1600]
    Ns = [100]
    methods = [reconstruct_tree.Reconstruction_Method(reconstruct_tree.neighbor_joining), reconstruct_tree.Reconstruction_Method()] #, Reconstruction_Method(scorer=reconstruct_tree.sum_squared_quartets), Reconstruction_Method(scorer=weird2)]
    # first one: delta^2 = 1/2
    # second one: probability of transitioning to some other state is 15 %
    mutation_rates = [jc.similarity2t(np.sqrt(1/2)), jc.p2t(0.85)]

    results = experiment(tree_list=binary_trees,
                            sequence_model=jc,
                            Ns=Ns,
                            methods=methods,
                            mutation_rates=mutation_rates,
                            savepath="example_run.pkl",
                            overwrite=True)
# %%
    #results = pickle.load(open("./data/example_run.pkl", 'rb'))
    df = results2frame(results)
    df["delta^2"] = pd.Series([generation.Jukes_Cantor().similarity(t)**2 for t in df['rate']]).round(3)
    
    print(df)
#    accuracy(df)
#    accuracy(df, col="delta^2")
# %%

# if __name__ == "__main__":
#     import pstats
#     p = pstats.Stats('profile.txt')
#     p.sort_stats('cumulative').print_stats(500)
