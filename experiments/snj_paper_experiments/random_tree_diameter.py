# %%
import pickle as pkl
import time

import dendropy
import igraph
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

import spectraltree
import spectraltree.compare_methods as compare_methods

def generate_plot(df):
    df_N = df.loc[df['N']==300]    
    df_N_delta = df_N.loc[df_N['delta']==0.9]
    df_N_delta = df_N_delta.loc[df_N_delta['diameter']<32]
    df_plot = pd.DataFrame(columns=['method', 'RF','diameter','N','delta'])
    #df_plot = df_N.iloc[::2]
    #df_plot['diff'] = np.array(df_N['RF'][1::2])-np.array(df_N['RF'][::2])
    #x = np.array(df_N['RF'][1::2])-np.array(df_N['RF'][::2])
    sns.set_style("whitegrid")    
    plt.rcParams.update({'font.size': 20})
    sns.catplot(data=df_N_delta,x='diameter',y='RF',kind="point",hue = 'method')
    plt.xlabel('Diameter')
    plt.ylabel('RF distance')
    plt.show()
    a = 1


folder = "./experiments/snj_paper_experiments/results/"
filename = '20201206_diameter_c'
df = pkl.load( open( folder + filename, "rb" ) )
generate_plot(df)



jc = spectraltree.Jukes_Cantor()
sequence_model = jc
mutation_rates = [jc.p2t(0.9),jc.p2t(0.95)]

def generate_random_tree_diameter(m):
    # generate adjacency matrix
    A = np.zeros((2*m-2,2*m-2))
    active_set = range(m)
    taxa_metadata = spectraltree.TaxaMetadata.default(m)
    G = taxa_metadata.all_leaves(edge_length=1)
    #available_clades = set(range(len(G)))   # len(G) == m
    for i in np.arange(0,m-3):
        
        # pick two nodes from active set and merge
        idx_vec = np.random.choice(active_set,2,replace=False)
        A[idx_vec[0],m+i]=1
        A[m+i,idx_vec[0]]=1
        A[idx_vec[1],m+i]=1
        A[m+i,idx_vec[1]]=1

        # merge two nodes in trees
        G.append(spectraltree.merge_children((G[idx_vec[0]], G[idx_vec[1]]), edge_length=1))
        
        # update active set
        active_set = np.delete(active_set,active_set==idx_vec[0])
        active_set = np.delete(active_set,active_set==idx_vec[1])
        active_set = np.append(active_set,m+i)

    # update adjacency
    #A[active_set[0],active_set[1]]=1    
    #A[active_set[1],active_set[0]]=1    
    A[active_set[0],2*m-3]=1    
    A[2*m-3,active_set[0]]=1    
    A[active_set[1],2*m-3]=1    
    A[2*m-3,active_set[1]]=1    
    A[active_set[2],2*m-3]=1    
    A[2*m-3,active_set[2]]=1    
    return A,dendropy.Tree(taxon_namespace=taxa_metadata.taxon_namespace, seed_node=spectraltree.merge_children(tuple(G[i] for i in active_set)), is_rooted=False)


df = pd.DataFrame(columns=['method', 'runtime', 'RF','m','diameter','N','delta'])
num_itr = 200
d = np.zeros(num_itr)
m = 250
delta_vec = [0.88,0.9,0.92,0.94]
N_vec = [100,200,300,400]
snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix) 
nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix) 

print(d.shape)
for i in range(num_itr):
    print(i)
    A,reference_tree = generate_random_tree_diameter(m)
    G = igraph.Graph.Adjacency((A > 0).tolist())
    d = G.diameter(directed=False)
    for delta in delta_vec:        
        mutation_rate = jc.p2t(delta)
        observations, taxa_meta = spectraltree.simulate_sequences(max(N_vec), tree_model=reference_tree, 
            seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")
        for N in N_vec:

            # SNJ        
            t_s = time.time()    
            tree_snj = snj(observations[:,:N], taxa_meta)
            runtime_snj = time.time()-t_s
            RF_snj,F1 = spectraltree.compare_trees(tree_snj, reference_tree)    
            df = df.append({'method': 'SNJ', 'runtime': runtime_snj, 'RF': RF_snj,
                'm': m,'diameter':d,'N':N,'delta':delta}, ignore_index=True)
        
            #NJ
            t_s = time.time()    
            tree_nj = nj(observations[:,:N], taxa_meta)
            runtime_nj = time.time()-t_s
            RF_nj,F1 = spectraltree.compare_trees(tree_nj, reference_tree)    
            df = df.append({'method': 'NJ', 'runtime': runtime_nj, 'RF': RF_nj,
                'm': m,'diameter':d,'N':N,'delta':delta}, ignore_index=True)

    compare_methods.save_results(df,'20201206_diameter_c',folder)

    #T.print_plot(width=50)
#print(d)

a=1
plt.hist(d, bins = 10)
plt.show()


        

