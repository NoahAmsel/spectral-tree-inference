import sys, os, platform

import spectraltree
import spectraltree.compare_methods as compare_methods
import time
from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
from dendropy.simulate.treesim import birth_death_tree, pure_kingman_tree, mean_kingman_tree 
import igraph
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations 
from itertools import product
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD 
from sklearn.utils.extmath import randomized_svd

def cherry_count_for_tree(tree):
    cherry_count = 0
    for node in tree.leaf_nodes():
        if node.parent_node.child_nodes()[0].is_leaf() and node.parent_node.child_nodes()[1].is_leaf():
            cherry_count += 1
    cherry_count = cherry_count/2
    return cherry_count

def generate_figure(df,x='n',y='RF',hue="method", kind="point",xlabel = None,
        ylabel = None,save_plot_file = None,format = 'eps'):
    col = x
    #dodge = 0.1*(df['method'].nunique() - 1)
    dodge = 0.1*(df[hue].nunique() - 1)
    sns.set_style("whitegrid")
    #sns.set_style("white")
    plt.rcParams.update({'font.size': 20})
    sns.catplot(data=df, x=x, y=y, kind="point", hue=hue,  dodge=dodge,\
       markers=["o", "s","D","8"], linestyles=["-", "--","-.",":"],legend=True,legend_out=False)    
    #plt.xticks(np.arange(2,10,2))
    ax = plt.gca()
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #tick_spacing = 2
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    if len(ax.get_xticklabels())>5:
        if (x=='n'):
            ax.set_xticklabels(ax.get_xticklabels()[::2])
            ax.set_xticks(ax.get_xticks()[::2])
    plt.legend(loc=1, prop={'size': 12}) #1 - upper right, 2 - upper left, 3 bottom left etc.
    if xlabel:
        plt.xlabel(xlabel)
    elif x == 'n':
        plt.xlabel('Number of samples n')
    elif x == 'm':
        plt.xlabel('Number of terminal nodes m')
    if ylabel:
        plt.ylabel(ylabel)
    else:
        if y == 'RF':
            plt.ylabel('RF distance')
        if y == 'runTime':
            plt.yscale('log')
            #plt.xsclae('log')
            #ax.set_xscale('log', basex=2)
            #ax.set_yscale('log', basey=2)    
            plt.ylabel('runtime')
    if save_plot_file:
        plt.savefig('./experiments/snj_paper_experiments/figures/' + save_plot_file,format=format)
    plt.show()

#num_taxa = 50
num_reps = 10
N = 200

m_vec = np.arange(20,100,20)
#m_vec = [20,40,60,80]
#reference_tree = spectraltree.balanced_binary(num_taxa)
jc = spectraltree.Jukes_Cantor()

mutation_rate = jc.p2t(0.85)
snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix) 
nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix) 
treesvd = spectraltree.TreeSVD()
methods = [snj,nj]
#results = spectraltree.experiment([reference_tree], jc, N_vec, methods=methods,\
#     mutation_rates = [mutation_rate], reps_per_tree=num_reps)
df = pd.DataFrame(columns=['method', 'runtime', 'RF','m','cherries_ref','cherries_res'])
for i in np.arange(num_reps):
    print(i)   
    for m in m_vec:
        #reference_tree = spectraltree.unrooted_pure_kingman_tree(m)
        reference_tree = spectraltree.unrooted_birth_death_tree(m)
        ch_ref = cherry_count_for_tree(reference_tree)
        observations, taxa_meta = spectraltree.simulate_sequences(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")

        # Tree svd
        #t_s = time.time()    
        #tree_svd_rec = treesvd(observations,taxa_meta)   
        #runtime_treesvd = time.time()-t_s
        #RF_svd,F1 = spectraltree.compare_trees(tree_svd_rec, reference_tree)
        #ch_svd = cherry_count_for_tree(tree_svd_rec)
        #df = df.append({'method': 'treesvd', 'runtime': runtime_treesvd, 'RF': RF_svd,'n': n,'cherries_ref': ch_ref,'cherries_res':ch_svd}, ignore_index=True)

        # SNJ        
        t_s = time.time()    
        tree_snj = snj(observations, taxa_meta)
        runtime_snj = time.time()-t_s
        RF_snj,F1 = spectraltree.compare_trees(tree_snj, reference_tree)
        ch_snj = cherry_count_for_tree(tree_snj)
        df = df.append({'method': 'SNJ', 'runtime': runtime_snj, 'RF': RF_snj,'m': m,'cherries_ref': ch_ref,'cherries_res':ch_snj}, ignore_index=True)
        
        #NJ
        t_s = time.time()    
        tree_nj = nj(observations, taxa_meta)
        runtime_nj = time.time()-t_s
        RF_nj,F1 = spectraltree.compare_trees(tree_nj, reference_tree)
        ch_nj = cherry_count_for_tree(tree_nj)
        df = df.append({'method': 'NJ', 'runtime': runtime_nj, 'RF': RF_nj,'m': m,'cherries_ref': ch_ref,'cherries_res':ch_nj}, ignore_index=True)


folder = "./experiments/snj_paper_experiments/results/"
compare_methods.save_results(df,'20200903_cherries_bd',folder)
df['ch_diff'] = df['cherries_res']-df['cherries_ref']
generate_figure(df,x='m',y='RF',hue="method", kind="point",save_plot_file = 'bd_trees_rf_cherries.eps',
    format = 'eps')
generate_figure(df,x='m',y='ch_diff',ylabel = 'Bias in cherry count',hue="method", kind="point",save_plot_file = 'bd_trees_cherry_count.eps',
    format = 'eps')
a = 1



 
        
        #print('SNJ:  RF, ',RF,' runtime ',runtime_snj)
