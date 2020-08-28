# %%
import datetime
import pickle as pkl
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


def generate_figure(df,x='n',y='RF',hue="method", kind="point",save_plot_file = None):
    col = x
    dodge = 0.1*(df['method'].nunique() - 1)
    sns.set_style("whitegrid")
    #sns.set_style("white")
    plt.rcParams.update({'font.size': 20})
    sns.catplot(data=df, x=x, y=y, kind="point", hue=hue,  dodge=dodge,\
       legend=True,legend_out=False)    
    plt.show()
    #h.set(ylim=(0,10))
    #new_labels = ['NJ', 'SNJ']
    #ax = plt.gca()
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, new_labels)
    #plt.ylabel(r'log(1+RF)')
    #plt.xlabel(r'$\delta$')
    #plt.show()
    if save_plot_file:
        plt.savefig('./experiments/snj_paper_experiments/figures/' + save_plot_file,format='png')

filename = "20200804_res11_SNJPAPER"
folder = "./experiments/snj_paper_experiments/"
separate = 'rate'
#res = pkl.load( open( "./experiments/snj_paper_experiments/20200804_res11_SNJPAPER", "rb" ) )
res = pkl.load( open( folder + filename, "rb" ) )
df_bin = compare_methods.results2frame(res[0])
df_cat = compare_methods.results2frame(res[1])

df_bin_split = []
df_cat_split = []
for r in np.unique(df_bin['rate']):
    df_bin_split.append(df_bin.loc[df_bin[separate]==r])
    df_cat_split.append(df_cat.loc[df_cat[separate]==r])
#generate_figure(df_bin_split[0],y='RF')
#generate_figure(df_bin_split[1],y='RF')
#generate_figure(df_cat_split[0],y='RF')
generate_figure(df_cat_split[1],y='RF',save_plot_file='caterpillar.png')



    

