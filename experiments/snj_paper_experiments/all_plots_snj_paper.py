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

import spectraltree
import spectraltree.compare_methods as compare_methods
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
import cProfile

jc = spectraltree.Jukes_Cantor()
sequence_model = jc
mutation_rates = [jc.p2t(0.9),jc.p2t(0.95)]

jc2 = spectraltree.Jukes_Cantor(num_classes=2)
sequence_model2 = jc2
mutation_rates2 = [jc2.p2t(0.9),jc2.p2t(0.95)]
#mutation_rates2 = [jc2.p2t(0.85)]


def part11(num_taxa, Ns_bin):
    ##### Part 1: NJ vs SNJ
    print("Starting Part 11")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    methods = [snj, nj]

    ## Part 1.1: different N    
    bin_tree = spectraltree.balanced_binary(num_taxa)

    res_bin11 = compare_methods.experiment(tree_list = [bin_tree], sequence_model = sequence_model, Ns = Ns_bin, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    return res_bin11

def part11_cat(num_taxa, Ns_cat):
    ##### Part 1: NJ vs SNJ
    print("Starting Part 11")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    methods = [snj, nj]

    ## Part 1.1: different N    
    #alphabet = "Binary"
    cat_tree = spectraltree.lopsided_tree(num_taxa)
    res_cat11 = compare_methods.experiment(tree_list = [cat_tree], sequence_model =sequence_model, Ns = Ns_cat, 
        methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True,
        savepath = '20200821_res12_cat_SNJPAPER',folder = './experiments/snj_paper_experiments/results/')
    return res_cat11



def part12_bin(num_taxa_s_bin, Ns_bin):
    ##### Part 1: NJ vs SNJ
    print("Starting Part 12")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    methods = [snj, nj]

    ## Part 1.2: different Taxa
    bin_trees = [spectraltree.balanced_binary(num_taxa) for num_taxa in num_taxa_s_bin]
    res_bin12 = compare_methods.experiment(tree_list = bin_trees, sequence_model = sequence_model, Ns = Ns_bin, methods= methods, mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    
    return res_bin12

def part12_cat(num_taxa_s_cat, Ns_cat):
    ##### Part 1: NJ vs SNJ
    print("Starting Part 12")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    methods = [snj, nj]

    ## Part 1.2: different Taxa
    cat_trees = [spectraltree.lopsided_tree(num_taxa) for num_taxa in num_taxa_s_cat]
    res_cat12 = compare_methods.experiment(tree_list = cat_trees, sequence_model = sequence_model, Ns = Ns_cat, methods = methods, 
        mutation_rates=mutation_rates, reps_per_tree=5, verbose=True)
    return res_cat12


def part21_bin(num_taxa_bin, Ns_bin):
    ##### Part 2: NJ vs SNJ vs RG vs CLRG
    print("Starting Part 21 for binary tree")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    rg = spectraltree.RG()
    #clrg = spectraltree.CLRG()
    methods = [snj, nj,rg]
    alphabet = "DNA"

    ## Part 2.1: different N
    bin_tree = spectraltree.balanced_binary(num_taxa_bin)
    res_bin21 = compare_methods.experiment(tree_list = [bin_tree], sequence_model = sequence_model, Ns = Ns_bin, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)    
    return res_bin21

def part21_king(num_taxa_king, Ns_king):
    ##### Part 2: NJ vs SNJ vs RG vs CLRG
    print("Starting Part 21 for kingman tree")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    rg = spectraltree.RG()
    #clrg = spectraltree.CLRG()
    methods = [snj, nj,rg]
    alphabet = "DNA"

    ## Part 2.1: different N
    king_tree = spectraltree.unrooted_pure_kingman_tree(num_taxa_king)
    res_king21 = compare_methods.experiment(tree_list = [king_tree], sequence_model = sequence_model, Ns = Ns_king, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)    
    return res_king21

def part21_cat(num_taxa_cat, Ns_cat):
    ##### Part 2: NJ vs SNJ vs RG vs CLRG
    print("Starting Part 21 for caterpillar tree")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    rg = spectraltree.RG()
    #clrg = spectraltree.CLRG()
    methods = [snj, nj,rg]
    alphabet = "DNA"

    ## Part 2.1: different N
    cat_tree = spectraltree.lopsided_tree(num_taxa_cat)
    res_cat21 = compare_methods.experiment(tree_list = [cat_tree], sequence_model = sequence_model, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet, verbose=True)    
    return res_cat21


def part22_bin(num_taxa_s_bin, Ns_bin):
    ## Part 2.2: different Taxa
    print("Starting Part 22 for binary trees")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    rg = spectraltree.RG()
    clrg = spectraltree.CLRG()
    methods = [snj, nj,rg]

    alphabet = "DNA"    
    bin_trees = [spectraltree.balanced_binary(num_taxa) for num_taxa in num_taxa_s_bin]
     
    res_bin22 = compare_methods.experiment(tree_list = bin_trees, sequence_model = sequence_model, Ns = Ns_bin, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet , verbose=True)   

    return res_bin22

def part22_king(num_taxa_s_king, Ns_king):
    ## Part 2.2: different Taxa
    print("Starting Part 22 for kingman trees")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    rg = spectraltree.RG()
    clrg = spectraltree.CLRG()
    methods = [snj, nj,rg]

    alphabet = "DNA"    
    king_trees = [spectraltree.unrooted_pure_kingman_tree(num_taxa) for num_taxa in num_taxa_s_king]
     
    res_king22 = compare_methods.experiment(tree_list = king_trees, sequence_model = sequence_model, Ns = Ns_king, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet , verbose=True)   

    return res_king22



def part22_cat(num_taxa_s_cat, Ns_cat):
    ## Part 2.2: different Taxa
    print("Starting Part 22 for caterpillar trees")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    rg = spectraltree.RG()
    clrg = spectraltree.CLRG()
    methods = [snj, nj,rg]

    alphabet = "Binary"    
    cat_trees = [spectraltree.lopsided_tree(num_taxa) for num_taxa in num_taxa_s_cat]     
    res_cat22 = compare_methods.experiment(tree_list = cat_trees, sequence_model = sequence_model, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates, reps_per_tree=5, alphabet=alphabet , verbose=True)   

    return res_cat22

def part31_bin(num_taxa_bin, Ns_bin):
    ##### Part 3: NJ vs SNJ vs RG vs CLRG vs Forrest vs Tree_SVD
    print("Starting Part 31")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    #rg = spectraltree.RG()
    #clrg = spectraltree.CLRG()
    forrest = spectraltree.Forrest()
    tree_svd = spectraltree.TreeSVD()
    #methods = [snj, nj,rg,clrg, forrest,tree_svd]
    methods = [snj, nj, forrest,tree_svd]
    #methods = [tree_svd]

    alphabet = "Binary"

    ## Part 3.1: different N
    
    bin_tree = spectraltree.balanced_binary(num_taxa_bin)

    res_bin31  = compare_methods.experiment(tree_list = [bin_tree], sequence_model = sequence_model2, Ns = Ns_bin, methods  =methods, mutation_rates=mutation_rates2, reps_per_tree=5, alphabet=alphabet, verbose=True)    
    return res_bin31

def part31_cat(num_taxa_cat, Ns_cat):
    ##### Part 3: NJ vs SNJ vs RG vs CLRG vs Forrest vs Tree_SVD
    print("Starting Part 31")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    #rg = spectraltree.RG()
    #clrg = spectraltree.CLRG()
    forrest = spectraltree.Forrest()
    tree_svd = spectraltree.TreeSVD()
    #methods = [snj, nj,rg,clrg, forrest,tree_svd]
    methods = [snj, nj, forrest,tree_svd]

    alphabet = "Binary"

    ## Part 3.1: different N
    
    cat_tree = spectraltree.lopsided_tree(num_taxa_cat)

    res_cat31  = compare_methods.experiment(tree_list = [cat_tree], sequence_model = sequence_model2, Ns = Ns_cat, methods = methods, mutation_rates=mutation_rates2, reps_per_tree=5, alphabet=alphabet, verbose=True)
    return res_cat31

def part31_king(num_taxa_king, Ns_king):
    ##### Part 3: NJ vs SNJ vs RG vs CLRG vs Forrest vs Tree_SVD
    print("Starting Part 31")
    snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
    nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
    #rg = spectraltree.RG()
    #clrg = spectraltree.CLRG()
    forrest = spectraltree.Forrest()
    tree_svd = spectraltree.TreeSVD()
    #methods = [snj, nj,rg,clrg, forrest,tree_svd]
    methods = [snj, nj, forrest,tree_svd]

    alphabet = "Binary"

    ## Part 3.1: different N
    kingman_tree = spectraltree.unrooted_birth_death_tree(num_taxa_king, birth_rate=1)

    res_king31 = compare_methods.experiment(tree_list = [kingman_tree], sequence_model = sequence_model2, Ns = Ns_king, methods = methods, mutation_rates=mutation_rates2, reps_per_tree=5, alphabet=alphabet, verbose=True)
    return res_king31


# %%
res_bin11 = part11(num_taxa = 512, Ns_bin = [100,150,200,250,300,350,400,450,500])
#ompare_methods.save_results(res_bin11, filename="20200810_res11_bin_SNJPAPER", folder="./experiments/snj_paper_experiments/")
res_cat11 = part11_cat(num_taxa = 512, Ns_cat = [1000,1300,1600,1900])
compare_methods.save_results(res_cat11, filename="20200819_res11_cat_SNJPAPER", folder="./experiments/snj_paper_experiments/results/")

# %%
#res_bin12, res_cat12 = part12(num_taxa_s_bin = [ 64, 128, 256, 512, 1024], num_taxa_s_cat = [100,200,300,400,500,600,700,800,900,1000], Ns_bin = [400], Ns_cat = [800])
#res_bin12 = part12_bin(num_taxa_s_bin = [ 64, 128, 256, 512,1024], Ns_bin = [400])
#compare_methods.save_results(res_bin12, filename="20200812_res12_bin_SNJPAPER", folder="./experiments/snj_paper_experiments/results/")
#res_cat12 = part12_cat(num_taxa_s_cat = [100,200,300,400,500], Ns_cat = [800])
#compare_methods.save_results(res_cat12, filename="20200812_res12_cat_SNJPAPER", folder="./experiments/snj_paper_experiments/results/")
# %%

#res_bin21 = part21_bin(num_taxa_bin = 128, Ns_bin = [300,350,400,450,500])
#compare_methods.save_results(res_bin21, filename="20200817_res21_bin_SNJPAPER", folder="./experiments/snj_paper_experiments/")
#res_cat21 = part21_cat(num_taxa_cat = 128, Ns_cat = [400,500,600,700,800,900,1000])
#compare_methods.save_results(res_cat21, filename="20200817_res21_cat_SNJPAPER", folder="./experiments/snj_paper_experiments/")
res_king21 = part21_king(num_taxa_king = 128, Ns_king = [600,800,1000,1200,1400])
compare_methods.save_results(res_king21, filename="20200819_res21_king_SNJPAPER", folder="./experiments/snj_paper_experiments/results/")
#res_bin22 = part22_bin(num_taxa_s_bin = [32,64,128,256], Ns_bin = [400])
#compare_methods.save_results(res_bin22, filename="20200817_res22_bin_SNJPAPER", folder="./experiments/snj_paper_experiments/")
#res_cat21 = part22_cat(num_taxa_s_cat = [30,50,70,90], Ns_cat = [800])
#compare_methods.save_results(res_cat21, filename="20200817_res22_cat_SNJPAPER", folder="./experiments/snj_paper_experiments/")
res_king22 = part22_king(num_taxa_s_king = [32,64,128,256], Ns_king = [1000])
compare_methods.save_results(res_king22, filename="20200819_res22_king_SNJPAPER", folder="./experiments/snj_paper_experiments/results/")

# res_bin22, res_cat22 = part22(num_taxa_s_bin = [32,64,128,256], num_taxa_s = [100,200,300,400,500,600,700,800,900,1000], Ns_bin = [400], Ns_cat = [800])
# compare_methods.save_results([res_bin22, res_cat22], filename="20200804_res22_SNJPAPER", folder="./experiments/snj_paper_experiments/")

# res_bin31 = part31_bin(num_taxa_bin = 16, Ns_bin = [30,50,70,90,110, 130,150])
# compare_methods.save_results(res_bin31, filename="20200812_res31_bin_SNJPAPER", folder="./experiments/snj_paper_experiments/")
res_cat31 = part31_cat( num_taxa_cat = 15, Ns_cat = [30,50,70,90,110, 130,150])
compare_methods.save_results(res_cat31, filename="20200812_res31_cat_SNJPAPER", folder="./experiments/snj_paper_experiments/")
res_king31 = part31_king(num_taxa_king = 15, Ns_king = [30,50,70,90,110, 130,150])
compare_methods.save_results(res_king31, filename="20200812_res31_king_SNJPAPER", folder="./experiments/snj_paper_experiments/")
