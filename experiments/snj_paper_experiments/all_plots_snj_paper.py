# %%
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
sys.path.append(os.path.join(os.path.split(os.path.dirname(sys.path[0]))[0],'spectraltree'))
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
sequence_model = jc
mutation_rates = [0.98,0.95]

def part11(num_taxa, Ns_bin, Ns_cat):
    ##### Part 1: NJ vs SNJ
    print("Starting Part 11")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    methods = [snj, nj]

    ## Part 1.1: different N    
    bin_tree = utils.balanced_binary(num_taxa)
    cat_tree = utils.lopsided_tree(num_taxa)

    res_bin11 = compare_methods.experiment(tree_list = [bin_tree], sequence_model = sequence_model, Ns = Ns_bin, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    res_cat11 = compare_methods.experiment(tree_list = [cat_tree], sequence_model =sequence_model, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    return res_bin11,res_cat11



def part12(num_taxa_s_bin, num_taxa_s_cat, Ns_bin, Ns_cat):
    ##### Part 1: NJ vs SNJ
    print("Starting Part 12")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    methods = [snj, nj]

    ## Part 1.2: different Taxa
    
    #num_taxa_s = [ 64, 128]
    bin_trees = [utils.balanced_binary(num_taxa) for num_taxa in num_taxa_s_bin]
    
    #num_taxa_s = [100,200]
    cat_trees = [utils.lopsided_tree(num_taxa) for num_taxa in num_taxa_s_cat]
    res_bin12 = compare_methods.experiment(tree_list = bin_trees, sequence_model = sequence_model, Ns = Ns_bin, methods= methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    res_cat12 = compare_methods.experiment(tree_list = cat_trees, sequence_model = sequence_model, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    return res_bin12,res_cat12

def part21(num_taxa_bin, num_taxa_cat, Ns_bin, Ns_cat):
    ##### Part 2: NJ vs SNJ vs RG vs CLRG
    print("Starting Part 21")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    rg = reconstruct_tree.RG()
    clrg = reconstruct_tree.CLRG()
    methods = [snj, nj,rg,clrg]

    alphabet = "Binary"

    ## Part 2.1: different N
    bin_tree = utils.balanced_binary(num_taxa_bin)
    cat_tree = utils.lopsided_tree(num_taxa_cat)

    res_bin21 = compare_methods.experiment(tree_list = [bin_tree], sequence_model = sequence_model, Ns = Ns_bin, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)
    res_cat21 = compare_methods.experiment(tree_list = [cat_tree], sequence_model = sequence_model, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)
    return res_bin21, res_cat21


def part22(num_taxa_s_bin,num_taxa_s_cat, Ns_bin, Ns_cat):
    ## Part 2.2: different Taxa
    print("Starting Part 22")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    rg = reconstruct_tree.RG()
    clrg = reconstruct_tree.CLRG()
    methods = [snj, nj,rg,clrg]

    alphabet = "Binary"

    
    bin_trees = [utils.balanced_binary(num_taxa) for num_taxa in num_taxa_s_bin]
   
    cat_trees = [utils.lopsided_tree(num_taxa) for num_taxa in num_taxa_s_cat]

    
    res_bin22 = compare_methods.experiment(tree_list = bin_trees, sequence_model = sequence_model, Ns = Ns_bin, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet , verbose=True)
    res_cat22 = compare_methods.experiment(tree_list = cat_trees, sequence_model = sequence_model, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)

    return res_bin22, res_cat22

def part31(num_taxa_bin, num_taxa_cat, num_taxa_king, Ns_bin,Ns_cat, Ns_king):
    ##### Part 3: NJ vs SNJ vs RG vs CLRG vs Forrest vs Tree_SVD
    print("Starting Part 31")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    rg = reconstruct_tree.RG()
    clrg = reconstruct_tree.CLRG()
    forrest = reconstruct_tree.Forrest()
    tree_svd = reconstruct_tree.TreeSVD()
    methods = [snj, nj,rg,clrg, forrest,tree_svd]

    alphabet = [1,2]

    ## Part 3.1: different N
    
    bin_tree = utils.balanced_binary(num_taxa_bin)
    cat_tree = utils.lopsided_tree(num_taxa_cat)
    kingman_tree = utils.unrooted_birth_death_tree(num_taxa_king, birth_rate=1)

    res_bin31  = compare_methods.experiment(tree_list = [bin_tree], sequence_model = sequence_model, Ns = Ns_bin, methods  =methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)    
    res_cat31  = compare_methods.experiment(tree_list = [cat_tree], sequence_model = sequence_model, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)
    res_king31 = compare_methods.experiment(tree_list = [kingman_tree], sequence_model = sequence_model, Ns = Ns_king, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)
    return res_bin31,res_cat31, res_king31

#res_bin11, res_cat11 = part11(num_taxa = 128, Ns_bin = [100,150,200,250,300,350,400,450,500], Ns_cat = [400,500,600,700,800,900,1000])
res_bin11, res_cat11 = part11(num_taxa = 128, Ns_bin = [100], Ns_cat = [400])
compare_methods.save_results([res_bin11, res_cat11], filename="20200804_res11_SNJPAPER", folder="./experiments/snj_paper_experiments/")

#res_bin12, res_cat12 = part12(num_taxa_s_bin = [ 64, 128, 256, 512, 1024], num_taxa_s_cat = [100,200,300,400,500,600,700,800,900,1000], Ns_bin = [400], Ns_cat = [800])
res_bin12, res_cat12 = part12(num_taxa_s_bin = [ 64], num_taxa_s_cat = [100], Ns_bin = [400], Ns_cat = [800])
compare_methods.save_results([res_bin12, res_cat12], filename="20200804_res12_SNJPAPER", folder="./experiments/snj_paper_experiments/")

#res_bin21, res_cat21 = part21(num_taxa_bin = 128, num_taxa_cat = 128, Ns_bin = [100,150,200,250,300,350,400,450,500], Ns_cat = [400,500,600,700,800,900,1000])
res_bin21, res_cat21 = part21(num_taxa_bin = 128, num_taxa_cat = 128, Ns_bin = [100], Ns_cat = [400])
compare_methods.save_results([res_bin21, res_cat21], filename="20200804_res21_SNJPAPER", folder="./experiments/snj_paper_experiments/")

#res_bin22, res_cat22 = part22(num_taxa_s_bin = [32,64,128,256], num_taxa_s = [100,200,300,400,500,600,700,800,900,1000], Ns_bin = [400], Ns_cat = [800])
res_bin22, res_cat22 = part22(num_taxa_s_bin = [32], num_taxa_s_cat = [100], Ns_bin = [400], Ns_cat = [800])
compare_methods.save_results([res_bin22, res_cat22], filename="20200804_res22_SNJPAPER", folder="./experiments/snj_paper_experiments/")

#res_bin31,res_cat31, res_king31 = part31(num_taxa_bin = 16, num_taxa_cat = 20, num_taxa_king = 20, Ns_bin = [30,50,70,90,110, 130,150], Ns_cat = [30,50,70,90,110, 130,150], Ns_king = [30,50,70,90,110, 130,150])
res_bin31,res_cat31, res_king31 = part31(num_taxa_bin = 16, num_taxa_cat = 20, num_taxa_king = 20, Ns_bin = [30], Ns_cat = [30], Ns_king = [30])
compare_methods.save_results([res_bin31,res_cat31, res_king31], filename="20200804_res31_SNJPAPER", folder="./experiments/snj_paper_experiments/")
