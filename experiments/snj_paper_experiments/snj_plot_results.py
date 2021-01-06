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
import matplotlib.ticker as ticker
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

def get_data(filename):
    res = pkl.load( open( folder + filename, "rb" ) )
    df = compare_methods.results2frame(res)

    # Split data by mutation rate
    df_split = []
    for r in np.unique(df['rate']):
        df_split.append(df.loc[df[separate]==r])
    return df_split
    
folder = "./experiments/snj_paper_experiments/results/"
separate = 'rate'


# #########################################################################################
# #Part 11
# filename = "20200823_res11_bin_SNJPAPER"
# df_split = get_data(filename)
# generate_figure(df_split[0],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# generate_figure(df_split[1],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

#filename = "20200823_res11_cat_SNJPAPER"
#df_split = get_data(filename)
#generate_figure(df_split[0],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
#generate_figure(df_split[1],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

# filename = "20200817_res11_king_SNJPAPER"
# df_split = get_data(filename)
# generate_figure(df_split[0],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# generate_figure(df_split[1],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

# #########################################################################################
# #Part 12
# filename = "20200817_res12_bin_SNJPAPER"
# df_split = get_data(filename)
# generate_figure(df_split[0],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# generate_figure(df_split[1],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

# filename = "20200817_res12_cat_SNJPAPER"
# df_split = get_data(filename)
# generate_figure(df_split[0],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# generate_figure(df_split[1],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

# filename = "20200817_res12_king_SNJPAPER"
# df_split = get_data(filename)
# generate_figure(df_split[0],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# generate_figure(df_split[1],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

#########################################################################################
## Part 21
#########################################################################################
# filename = "20200818_res21_bin_SNJPAPER"
# df_split = get_data(filename)
# # Gen Figs for mutation rate #1
# generate_figure(df_split[0],x='n',y='runTime', save_plot_file=filename+"_runtime_MR"+str(0)+'.eps')
# generate_figure(df_split[0],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# # Gen Figs for mutation rate #2
# generate_figure(df_split[1],x='n',y='runTime', save_plot_file=filename+"_runtime_MR"+str(1)+'.eps')
# generate_figure(df_split[1],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

#filename = "20200818_res21_cat_SNJPAPER"
#df_split = get_data(filename)
# # Gen Figs for mutation rate #1
# generate_figure(df_split[0],x='n',y='runTime', save_plot_file=filename+"_runtime_MR"+str(0)+'.eps')
#generate_figure(df_split[0],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# # Gen Figs for mutation rate #2
# generate_figure(df_split[1],x='n',y='runTime', save_plot_file=filename+"_runtime_MR"+str(1)+'.eps')
#generate_figure(df_split[1],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

#filename = "20200819_res21_king_SNJPAPER"
#df_split = get_data(filename)
# # Gen Figs for mutation rate #1
# generate_figure(df_split[0],x='n',y='runTime', save_plot_file=filename+"_runtime_MR"+str(0)+'.eps')
#generate_figure(df_split[0],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# # Gen Figs for mutation rate #2
# generate_figure(df_split[1],x='n',y='runTime', save_plot_file=filename+"_runtime_MR"+str(1)+'.eps')
#generate_figure(df_split[1],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

#########################################################################################
## Part 22
#########################################################################################
#filename = "20200818_res22_bin_SNJPAPER"
#df_split = get_data(filename)
# # Gen Figs for mutation rate #1
#generate_figure(df_split[0],x='m',y='runTime', save_plot_file=filename+"_runtime_MR"+str(0)+'.eps')
#generate_figure(df_split[0],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# # Gen Figs for mutation rate #2
#generate_figure(df_split[1],x='m',y='runTime', save_plot_file=filename+"_runtime_MR"+str(1)+'.eps')
#generate_figure(df_split[1],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

#filename = "20200818_res22_cat_SNJPAPER"
#df_split = get_data(filename)
# # Gen Figs for mutation rate #1
#generate_figure(df_split[0],x='m',y='runTime', save_plot_file=filename+"_runtime_MR"+str(0)+'.eps')
#generate_figure(df_split[0],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# # Gen Figs for mutation rate #2
#generate_figure(df_split[1],x='m',y='runTime', save_plot_file=filename+"_runtime_MR"+str(1)+'.eps')
#generate_figure(df_split[1],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

#filename = "20200819_res22_king_SNJPAPER"
#df_split = get_data(filename)
# # Gen Figs for mutation rate #1
# generate_figure(df_split[0],x='m',y='runTime', save_plot_file=filename+"_runtime_MR"+str(0)+'.eps')
#generate_figure(df_split[0],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')
# # Gen Figs for mutation rate #2
# generate_figure(df_split[1],x='m',y='runTime', save_plot_file=filename+"_runtime_MR"+str(1)+'.eps')
#generate_figure(df_split[1],x='m',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

#########################################################################################
#Part 31
#filename = "20200812_res31_bin_SNJPAPER"
#df_split = get_data(filename)

# Gen Figs for mutation rate #1
#generate_figure(df_split[0],x='n',y='runTime', save_plot_file=filename+"_runtime_MR"+str(0)+'.eps')
#generate_figure(df_split[0],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')

# Gen Figs for mutation rate #2
#generate_figure(df_split[1],x='n',y='runTime', save_plot_file=filename+"_runtime_MR"+str(1)+'.eps')
#generate_figure(df_split[1],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

#filename = "20200812_res31_cat_SNJPAPER"
#df_split = get_data(filename)

# Gen Figs for mutation rate #1
#generate_figure(df_split[0],x='n',y='runTime', save_plot_file=filename+"_runtime_MR"+str(0)+'.eps')
#generate_figure(df_split[0],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(0)+'.eps')

# Gen Figs for mutation rate #2
#generate_figure(df_split[1],x='n',y='runTime', save_plot_file=filename+"_runtime_MR"+str(1)+'.eps')
#generate_figure(df_split[1],x='n',y='RF', save_plot_file=filename+"_RF_MR"+str(1)+'.eps')

# heterogeniety experiment
filename = '20200901_heterogeneity_bin'
df = pkl.load( open( folder + filename, "rb" ) )
df = df[df['n']<600]
df.loc[df['method']=='SNJ-het','method']='SNJ'
df.loc[df['method']=='NJ-het','method']='NJ'
df_rep = df.iloc[np.arange(1,200,2)]
df_rep = df_rep.append(df.iloc[np.arange(0,200,2)],ignore_index=True)

df_split = []
separate = 'gamma_shape'
for r in np.unique(df_rep[separate]):
    df_split.append(df_rep.loc[df_rep[separate]==r])
idx = 0
generate_figure(df_split[idx],x='n',y='RF', save_plot_file=filename+str(idx)+'.eps')
