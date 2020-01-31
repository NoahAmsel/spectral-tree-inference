#! /usr/bin/env python

import sys
sys.path.insert(0, '..')

from spectraltree import *
from compare_methods import *

binary_trees = [balanced_binary(m) for m in [64, 128, 256]]
lopsided = [lopsided_tree(m) for m in [64, 128]]
jc = Jukes_Cantor()
Ns = [100, 500, 800] #np.linspace(100,1200,200).astype(int)
methods = [Reconstruction_Method(), Reconstruction_Method(scorer=sum_squared_quartets)] #, Reconstruction_Method(scorer=reconstruct_tree.sum_squared_quartets), Reconstruction_Method(scorer=weird2)]
delta_vec = np.linspace(0.65,0.95,7)
mutation_rates = [jc.similarity2t(delta) for delta in delta_vec]

results = experiment(tree_list=binary_trees,
                                sequence_model=jc,
                                Ns=Ns,
                                methods=methods,
                                mutation_rates=mutation_rates,
                                reps_per_tree=2, #10
                                savepath="speedtest.pkl")
if False:
    import pstats
    p = pstats.Stats('speed_results')
    p.sort_stats('cumulative').print_stats(10)
