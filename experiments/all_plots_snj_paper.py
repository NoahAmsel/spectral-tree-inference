import datetime
import pickle
import os.path
import time
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import dendropy
import copy

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))

#import spectraltree
import utils
import generation
import reconstruct_tree
import compare_methods
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
import cProfile

jc = generation.Jukes_Cantor()
mutation_rates = [0.98,0.95]
##### Part 1: NJ vs SNJ
snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
methods = [snj, nj]

## Part 1.1: different N

num_taxa = 128
bin_tree = utils.balanced_binary(num_taxa)
cat_tree = utils.lopsided_tree(num_taxa)

sequence_model = jc

Ns = [100,150,200,250,300,350,400,450,500]

res_bin11 = compare_methods.experiment(tree_list = [bin_tree], sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
Ns = [400,500,600,700,800,900,1000]
res_cat11 = compare_methods.experiment(tree_list = [cat_tree], sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)

## Part 1.2: different Taxa
num_taxa_s = [ 64, 128, 256, 512, 1024]
bin_trees = [utils.balanced_binary(num_taxa) for num_taxa in num_taxa_s]
num_taxa_s = [100,200,300,400,500,600,700,800,900,1000]
cat_trees = [utils.lopsided_tree(num_taxa) for num_taxa in num_taxa_s]

sequence_model = jc
Ns = [400]
res_bin12 = compare_methods.experiment(tree_list = bin_trees, sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
Ns = [800]
res_cat12 = compare_methods.experiment(tree_list = cat_trees, sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)



res1 = [res_bin11,res_cat11, res_bin12,res_cat12]
save_results(res1, filename="20200803_res1_SNJPAPER", folder=".")

##### Part 2: NJ vs SNJ vs RG vs CLRG
snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
rg = reconstruct_tree.RG()
clrg = reconstruct_tree.CLRG()
methods = [snj, nj,rg,clrg]

alphabet = [1,2]

## Part 2.1: different N
num_taxa = 128
bin_tree = utils.balanced_binary(num_taxa)
cat_tree = utils.lopsided_tree(num_taxa)

sequence_model = jc
Ns = [100,150,200,250,300,350,400,450,500]
methods = []
res_bin21 = compare_methods.experiment(tree_list = [bin_tree], sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)
Ns = [400,500,600,700,800,900,1000]
res_cat21 = compare_methods.experiment(tree_list = [cat_tree], sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)

## Part 2.2: different Taxa
num_taxa_s = [32,64,128,256]
bin_trees = [utils.balanced_binary(num_taxa) for num_taxa in num_taxa_s]
num_taxa_s = [100,200,300,400,500,600,700,800,900,1000]
cat_trees = [utils.lopsided_tree(num_taxa) for num_taxa in num_taxa_s]

sequence_model = jc
Ns = [400]
res_bin22 = compare_methods.experiment(tree_list = bin_trees, sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet , verbose=True)
Ns = [800]
res_cat22 = compare_methods.experiment(tree_list = cat_trees, sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)


res2 = [res_bin21,res_cat21, res_bin22,res_cat22]
save_results(res2, filename="20200803_res2_SNJPAPER", folder=".")
##### Part 3: NJ vs SNJ vs RG vs CLRG vs Forrest vs Tree_SVD
snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
rg = reconstruct_tree.RG()
clrg = reconstruct_tree.CLRG()
forrest = reconstruct_tree.Forrest()
tree_svd = reconstruct_tree.TreeSVD()
methods = [snj, nj,rg,clrg, forrest,tree_svd]

alphabet = [1,2]

## Part 3.1: different N
num_taxa = 16
bin_tree = utils.balanced_binary(num_taxa)
cat_tree = utils.lopsided_tree(num_taxa)
num_taxa = 20
kingman_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=1)

sequence_model = jc
Ns = [30,50,70,90,110, 130,150]
res_bin31  = compare_methods.experiment(tree_list = [bin_tree], sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)
Ns = [30,50,70,90,110, 130,150]
res_cat31  = compare_methods.experiment(tree_list = [cat_tree], sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)
res_king31 = compare_methods.experiment(tree_list = [kingman_tree], sequence_model, Ns, methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)


res3 = [res_bin31,res_cat31, res_king31]
save_results(res3, filename="20200803_res3_SNJPAPER", folder=".")