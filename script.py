#! /usr/bin/env python

from spectraltree import *
from compare_methods import *

binary_trees = [balanced_binary(m) for m in [64, 128, 256]]
#lopsided = [lopsided_tree(m) for m in [64, 128]]
jc = Jukes_Cantor()
Ns = np.linspace(100,1200,100).astype(int)
methods = [Reconstruction_Method(neighbor_joining), Reconstruction_Method()] #, Reconstruction_Method(scorer=reconstruct_tree.sum_squared_quartets), Reconstruction_Method(scorer=weird2)]
delta_vec = np.linspace(0.65,0.95,7)
mutation_rates = [jc.similarity2t(delta) for delta in delta_vec]

results = experiment(tree_list=binary_trees,
                        sequence_model=jc,
                        Ns=Ns,
                        methods=methods,
                        mutation_rates=mutation_rates,
                        reps_per_tree=10,
                        savepath="grid1.pkl")
# %%
