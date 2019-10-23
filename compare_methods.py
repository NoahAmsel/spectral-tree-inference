from functools import partial
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import dendropy

import reconstruct_tree
import generation
import utils

"""
p = partial(reconstruct_tree.estimate_tree_topology, scorer=reconstruct_tree.sv2)
str(p.keywords['scorer'])
p.keywords['scorer'].__name__
"""

# TODO this should be a subclass of partial
class Reconstruction_Method:
    def __init__(self, core=reconstruct_tree.estimate_tree_topology, distance=reconstruct_tree.paralinear_distance, **kwargs):
        self.core = core
        self.distance = distance
        if self.core == reconstruct_tree.estimate_tree_topology:
            kwargs["scorer"] = kwargs.get("scorer", reconstruct_tree.sv2)
            kwargs["scaler"] = kwargs.get("scaler", 1.0)
        self.kwargs = kwargs

    @property
    def scorer(self):
        return self.kwargs.get('scorer',None)

    @property
    def scaler(self):
        return self.kwargs.get('scaler',None)

    def __call__(self, observations, namespace=None):
        distance_matrix = self.distance(observations)
        tree = self.core(distance_matrix, namespace=namespace, **self.kwargs)
        return tree

    def from_charmatrix(self, char_matrix, namespace=None):
        """
        takes dendropy.datamodel.charmatrixmodel.CharacterMatrix
        """
        observations, alphabet = utils.charmatrix2array(char_matrix)
        return self(observations)

    def __str__(self):
        if self.core == reconstruct_tree.estimate_tree_topology:
            s = self.scorer.__name__
            if self.scaler != 1.0:
                s += " x{:.2}".format(self.scaler)
        elif self.core == reconstruct_tree.neighbor_joining:
            s = "NJ"
        else:
            s = self.core.__name__
        return s

class Experiment_Datum:
    def __init__(self, sequence_model, n, method, inferred_tree, reference_tree):
        self.sequence_model = sequence_model
        self.n = n
        self.method = method
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

def experiment(tree_list, sequence_model, Ns, methods, reps_per_tree=1):
    results = []
    total_trials = len(tree_list)*len(Ns)*len(methods)*reps_per_tree
    i = 0
    for reference_tree in tree_list:
        for _ in range(reps_per_tree):
            sequences = sequence_model(seq_len=max(Ns), tree_model=reference_tree)
            observations, alphabet = utils.charmatrix2array(sequences)
            for n in Ns:
                for method in methods:
                    inferred_tree = method(observations[:,:n], namespace=reference_tree.taxon_namespace)
                    results.append(Experiment_Datum(sequence_model, n, method, inferred_tree, reference_tree))
                    i += 1
                    print("{0} / {1}".format(i, total_trials))
    return results

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
def my_weird(A, M):
    pass
# %%

if __name__ == "__main__":
    t = utils.balanced_binary(128)
    trees = dendropy.TreeList([utils.balanced_binary(128), utils.lopsided_tree(128)])
    jc_model = dendropy.model.discrete.Jc69()
    seqgen = partial(dendropy.model.discrete.simulate_discrete_chars, seq_model=jc_model, mutation_rate=0.1)
    Ns = np.geomspace(100, 1_000, num=4).astype(int)
    methods = [Reconstruction_Method(reconstruct_tree.neighbor_joining), Reconstruction_Method(), Reconstruction_Method(scorer=reconstruct_tree.sum_squared_quartets)]
    results = experiment(trees, seqgen, Ns=Ns, methods=methods)
    df = results2frame(results)
    correct(df)
    accuracy(df)

    mat = dendropy.model.discrete.simulate_discrete_chars(1000, trees[0], my_model)
    observations, _ = utils.charmatrix2array(mat)
    observations.shape
