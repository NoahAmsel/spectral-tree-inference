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
mutation_rates = [jc.p2t(0.98),jc.p2t(0.95)]

def part11(num_taxa, Ns_bin):
    ##### Part 1: NJ vs SNJ
    print("Starting Part 11")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    methods = [snj, nj]

    ## Part 1.1: different N    
    bin_tree = utils.balanced_binary(num_taxa)

    res_bin11 = compare_methods.experiment(tree_list = [bin_tree], sequence_model = sequence_model, Ns = Ns_bin, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    return res_bin11

def part11_cat(num_taxa, Ns_cat):
    ##### Part 1: NJ vs SNJ
    print("Starting Part 11")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    methods = [snj, nj]

    ## Part 1.1: different N    
    cat_tree = utils.lopsided_tree(num_taxa)

    res_cat11 = compare_methods.experiment(tree_list = [cat_tree], sequence_model =sequence_model, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    return res_cat11



def part12_bin(num_taxa_s_bin, Ns_bin):
    ##### Part 1: NJ vs SNJ
    print("Starting Part 12")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    methods = [snj, nj]

    ## Part 1.2: different Taxa
    bin_trees = [utils.balanced_binary(num_taxa) for num_taxa in num_taxa_s_bin]
    res_bin12 = compare_methods.experiment(tree_list = bin_trees, sequence_model = sequence_model, Ns = Ns_bin, methods= methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    
    return res_bin12

def part12_cat(num_taxa_s_cat, Ns_cat):
    ##### Part 1: NJ vs SNJ
    print("Starting Part 12")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    methods = [snj, nj]

    ## Part 1.2: different Taxa
    cat_trees = [utils.lopsided_tree(num_taxa) for num_taxa in num_taxa_s_cat]
    res_cat12 = compare_methods.experiment(tree_list = cat_trees, sequence_model = sequence_model, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    return res_cat12


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

def part31_bin(num_taxa_bin, Ns_bin):
    ##### Part 3: NJ vs SNJ vs RG vs CLRG vs Forrest vs Tree_SVD
    print("Starting Part 31")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    #rg = reconstruct_tree.RG()
    #clrg = reconstruct_tree.CLRG()
    forrest = reconstruct_tree.Forrest()
    tree_svd = reconstruct_tree.TreeSVD()
    methods = [snj, nj,rg,clrg, forrest,tree_svd]

    alphabet = "Binary"

    ## Part 3.1: different N
    
    bin_tree = utils.balanced_binary(num_taxa_bin)

    res_bin31  = compare_methods.experiment(tree_list = [bin_tree], sequence_model = sequence_model, Ns = Ns_bin, methods  =methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)    
    return res_bin31

def part31_cat(num_taxa_cat, Ns_cat):
    ##### Part 3: NJ vs SNJ vs RG vs CLRG vs Forrest vs Tree_SVD
    print("Starting Part 31")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    #rg = reconstruct_tree.RG()
    #clrg = reconstruct_tree.CLRG()
    forrest = reconstruct_tree.Forrest()
    tree_svd = reconstruct_tree.TreeSVD()
    methods = [snj, nj,rg,clrg, forrest,tree_svd]

    alphabet = "Binary"

    ## Part 3.1: different N
    
    cat_tree = utils.lopsided_tree(num_taxa_cat)

    res_cat31  = compare_methods.experiment(tree_list = [cat_tree], sequence_model = sequence_model, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)
    return res_cat31

def part31_king(num_taxa_king, Ns_king):
    ##### Part 3: NJ vs SNJ vs RG vs CLRG vs Forrest vs Tree_SVD
    print("Starting Part 31")
    snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)
    nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)
    #rg = reconstruct_tree.RG()
    #clrg = reconstruct_tree.CLRG()
    forrest = reconstruct_tree.Forrest()
    tree_svd = reconstruct_tree.TreeSVD()
    methods = [snj, nj,rg,clrg, forrest,tree_svd]

    alphabet = "Binary"

    ## Part 3.1: different N
    kingman_tree = utils.unrooted_birth_death_tree(num_taxa_king, birth_rate=1)

    res_king31 = compare_methods.experiment(tree_list = [kingman_tree], sequence_model = sequence_model, Ns = Ns_king, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)
    return res_king31


# %%
# res_bin11 = part11_bin(num_taxa = 128, Ns_bin = [100,150,200,250,300,350,400,450,500])
# compare_methods.save_results(res_bin11, filename="20200805_res11_bin_SNJPAPER", folder="./experiments/snj_paper_experiments/")
# res_cat11 = part11_cat(num_taxa = 128, Ns_cat = [400,500,600,700,800,900,1000])
# compare_methods.save_results(res_cat11, filename="20200805_res11_cat_SNJPAPER", folder="./experiments/snj_paper_experiments/")

#res_bin12, res_cat12 = part12(num_taxa_s_bin = [ 64, 128, 256, 512, 1024], num_taxa_s_cat = [100,200,300,400,500,600,700,800,900,1000], Ns_bin = [400], Ns_cat = [800])
# res_bin12 = part12_bin(num_taxa_s_bin = [ 64, 128, 256, 512], Ns_bin = [400])
# compare_methods.save_results(res_bin12, filename="20200805_res12_bin_SNJPAPER", folder="./experiments/snj_paper_experiments/")
res_cat12 = part12_cat(num_taxa_s_cat = [100,200,300,400,500], Ns_cat = [800])
compare_methods.save_results(res_cat12, filename="20200805_res12_cat_SNJPAPER", folder="./experiments/snj_paper_experiments/")

# res_bin21, res_cat21 = part21(num_taxa_bin = 128, num_taxa_cat = 128, Ns_bin = [100,150,200,250,300,350,400,450,500], Ns_cat = [400,500,600,700,800,900,1000])
# compare_methods.save_results([res_bin21, res_cat21], filename="20200804_res21_SNJPAPER", folder="./experiments/snj_paper_experiments/")

# res_bin22, res_cat22 = part22(num_taxa_s_bin = [32,64,128,256], num_taxa_s = [100,200,300,400,500,600,700,800,900,1000], Ns_bin = [400], Ns_cat = [800])
# compare_methods.save_results([res_bin22, res_cat22], filename="20200804_res22_SNJPAPER", folder="./experiments/snj_paper_experiments/")

res_bin31 = part31_bin(num_taxa_bin = 16, Ns_bin = [30,50,70,90,110, 130,150])
compare_methods.save_results(res_bin31, filename="20200805_res31_bin_SNJPAPER", folder="./experiments/snj_paper_experiments/")
res_cat31 = part31_cat( num_taxa_cat = 15, Ns_cat = [30,50,70,90,110, 130,150])
compare_methods.save_results(res_cat31, filename="20200805_res31_cat_SNJPAPER", folder="./experiments/snj_paper_experiments/")
res_king31 = part31_king(num_taxa_king = 15, Ns_king = [30,50,70,90,110, 130,150])
compare_methods.save_results(res_king31, filename="20200805_res31_king_SNJPAPER", folder="./experiments/snj_paper_experiments/")
