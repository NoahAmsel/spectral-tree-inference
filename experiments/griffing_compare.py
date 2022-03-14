import spectraltree
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import time

import dendropy
import igraph
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

import spectraltree


def generate_random_tree_w_adj(m):
    # generate adjacency matrix
    A = np.zeros((2*m-2,2*m-2))
    active_set = np.arange(m)
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
        active_set = active_set [active_set  != idx_vec[0]]
        active_set = active_set [active_set  != idx_vec[1]]
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
    return A,dendropy.Tree(taxon_namespace=taxa_metadata.taxon_namespace, seed_node=spectraltree.merge_children(tuple(G[i] for i in active_set)), is_rooted=False), taxa_metadata



def GriffingPartitioning(S):
    D = -np.log(np.clip(S, a_min=1e-20, a_max=None))
    #plt.imshow(D)
    num_taxa = S.shape[0]
    H = np.eye(num_taxa)-np.ones(num_taxa)/num_taxa
    G = -2*H@D@H
    w_g,v_g = np.linalg.eigh(G)
    w_s,v_s = np.linalg.eigh(np.diag(np.sum(S,axis=1)) - S)
    return v_g[:,-1], v_s[:,1]


# def check_is_bipartition(tree, bool_partition):
#     bipartitions = [str(x)[::-1] for x in tree.encode_bipartitions()]
#     partition_1 = "".join(list(bool_partition.astype('int').astype('str')))
#     partition_2 = "".join(list((1 - bool_partition).astype('int').astype('str')))
#     is_bipartition = (partition_1 in bipartitions) or (partition_2 in bipartitions)
#     return is_bipartition

def check_is_bipartition(tree, partition_mask, meta):
    return (meta.mask2bipartition(partition_mask).split_bitmask in tree.split_bitmask_edge_map) or \
            (meta.mask2bipartition(~partition_mask).split_bitmask in tree.split_bitmask_edge_map)

    bipartitions = [str(x)[::-1] for x in tree.encode_bipartitions()]
    partition_1 = "".join(list(partition_mask.astype('int').astype('str')))
    partition_2 = "".join(list((1 - partition_mask).astype('int').astype('str')))
    is_bipartition = (partition_1 in bipartitions) or (partition_2 in bipartitions)
    return is_bipartition


# def check_is_bipartition(tree, parent_split,bool_partition):
#     if sum(bool_partition) == 0:
#         return False
#     if sum(bool_partition) == len(bool_partition):
#         return False
#     bipartitions = [str(x)[::-1] for x in tree.encode_bipartitions()]

#     #ariel add - remove parent list
#     idx_parent = [i for i, element in enumerate(parent_split) if element]
#     bipartitions = ["".join([p[i] for i in idx_parent]) for p in bipartitions]

#     partition_1 = "".join(list(bool_partition.astype('int').astype('str')))
#     partition_2 = "".join(list((1 - bool_partition).astype('int').astype('str')))
#     is_bipartition = (partition_1 in bipartitions) or (partition_2 in bipartitions)
#     return is_bipartition


# num_taxa = 128   # Number of terminal nodes
# n = 100        # Number of independent samples (sequence length)   
# jc = spectraltree.Jukes_Cantor()   #set evolution process to the Jukes Cantor model
# mutation_rate = jc.p2t(0.9)        #set mutation rate between adjacent nodes to 1-0.9=0.1


# # create a tree according to the coalescent model
# #reference_tree = spectraltree.unrooted_pure_kingman_tree(num_taxa)

# # create a tree according to the birth death model model        
# #reference_tree = spectraltree.unrooted_birth_death_tree(num_taxa)

# # create a symmetric binary tree
# reference_tree = spectraltree.balanced_binary(num_taxa)

# # create a caterpiller tree 
# #reference_tree = spectraltree.lopsided_tree(num_taxa)

# # generate sequences: input - sequence length, specified tree, evolutionary model, mutation rate and alphabet
# observations, taxa_meta = spectraltree.simulate_sequences(n, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")


# S = spectraltree.JC_similarity_matrix(observations)
# v_g, v_s = GriffingPartitioning(S)
# print("Griffing correct: ", check_is_bipartition(reference_tree,v_g>0) )
# print("Spectral correct: ", check_is_bipartition(reference_tree,v_s>0) )
# plt.figure(figsize=(10,10))
# plt.plot(v_g)
# plt.plot(v_s)
# plt.legend(['G','L_S'])
# plt.plot([0,num_taxa],[0 ,0],'--')
# plt.show()


#STARTING LOOP


df = pd.DataFrame(columns=['Method', 'is_correct','m','diameter','N','delta'])
num_itr = 200
d = np.zeros(num_itr)
m = 128
delta_vec = [0.84,0.86, 0.88,0.9, 0.92] #[0.9]
delta_vec = [0.9]
N_vec = [100]#[75,100,150,200]
N_vec = [75,100,150,200]
jc = spectraltree.Jukes_Cantor() 

#################################
### Trying full recovery
##################################
# delta = delta_vec[-1]

# reference_tree = spectraltree.unrooted_birth_death_tree(m,birth_rate=1)
# mutation_rate = jc.p2t(delta)
# observations, taxa_meta = spectraltree.simulate_sequences(max(N_vec), tree_model=reference_tree, 
#                                                 seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")
# spectral_method = spectraltree.STDR(spectraltree.RAxML,spectraltree.JC_similarity_matrix)   
#spectral_method = spectraltree.STDR(spectraltree.NeighborJoining,spectraltree.JC_similarity_matrix)   

# tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, 
#     spectraltree.JC_similarity_matrix,
#     taxa_metadata= taxa_meta, 
#     threshold = 32,
#     min_split = 5,
#     merge_method = "least_square", 
#     verbose=False)
# RF, _ = spectraltree.compare_trees(tree_rec, reference_tree)
# print ("Trying to recover the whole tree. RF = ",RF)




for i in range(num_itr):
    print(i)
    
    #reference_tree = spectraltree.balanced_binary(m)
    #reference_tree = spectraltree.lopsided_tree(m)
    #reference_tree = spectraltree.unrooted_birth_death_tree(m,birth_rate=1)
    # for x in reference_tree.preorder_edge_iter():
    #     x.length = 1
    reference_tree = spectraltree.unrooted_pure_kingman_tree(m)
    d = 1
    for delta in delta_vec:        
        mutation_rate = jc.p2t(delta)
        observations, taxa_meta = spectraltree.simulate_sequences(max(N_vec), tree_model=reference_tree, 
            seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")
        
        #observations = observations[taxa_meta._taxa_list.argsort(),:]
        # taxa_metadata = taxa_meta # should be removed if generate_random_tree_w_adj is not used
        # taxa_perm = [taxa_meta[taxa_metadata._taxa_list[i]] for i in range(len(taxa_metadata._taxa_list))]  # should be removed if generate_random_tree_w_adj is not used
        for N in N_vec:
            S = spectraltree.JC_similarity_matrix(observations[:,:N])
            v_g, v_s = GriffingPartitioning(S)
            # v_s = v_s[taxa_perm]
            # v_g = v_g[taxa_perm]
            print("Griffing correct: ", check_is_bipartition(reference_tree,v_g>0, taxa_meta) )
            print("Spectral correct: ", check_is_bipartition(reference_tree,v_s>0, taxa_meta) )
            
            df = df.append({'Method': 'Distance based', 'is_correct': check_is_bipartition(reference_tree,v_g>0, taxa_meta),
                'm': m,'diameter':d,'N':N,'delta':delta}, ignore_index=True)
            df = df.append({'Method': 'Similarity based', 'is_correct': check_is_bipartition(reference_tree,v_s>0, taxa_meta),
                'm': m,'diameter':d,'N':N,'delta':delta}, ignore_index=True)

    #compare_methods.save_results(df,'20201206_diameter_c',folder)



print(df)
print("Griffing correct:", sum(df[(df['Method'] == 'Distance based')]['is_correct']))
print("Spectral correct:", sum(df[(df['Method'] == 'Similarity based')]['is_correct']))

#Diameter plot
# max_daim = max(df['diameter'])
# total_in_each_diam = [len(df[(df['Method'] == 'Distance based') & (df['diameter'] ==  i)]) for i in range(1,max_daim)]
# g_t_d = [sum(df[(df['Method'] == 'Distance based') & (df['diameter'] ==  i)]['is_correct']) for i in range(1,max_daim)]
# s_t_d = [sum(df[(df['Method'] == 'Similarity based') & (df['diameter'] ==  i)]['is_correct']) for i in range(1,max_daim)]
# g_t_d_normlized = [0 if total_in_each_diam[i]==0 else g_t_d[i]/total_in_each_diam[i] for i in range(len(g_t_d))]
# s_t_d_normlized = [0 if total_in_each_diam[i]==0 else s_t_d[i]/total_in_each_diam[i] for i in range(len(s_t_d))]
# plt.plot(g_t_d_normlized)
# plt.plot(s_t_d_normlized)
# plt.legend(['Distance based','Similarity based'])
# #plt.plot(total_in_each_diam,'--')
# plt.show()

#Delta plot
sns.catplot(data=df, x='delta', y='is_correct', kind="point", hue='Method', 
       markers=["o", "s"], linestyles=["-", "--"],legend=True,legend_out=False)    
plt.ylabel( "Correct ratio")
plt.xlabel('$\delta$')
plt.show()
# g_t_d = [sum(df[(df['Method'] == 'Distance based') & (df['delta'] ==  delta_vec[i])]['is_correct']) for i in range(len(delta_vec))]
# s_t_d = [sum(df[(df['Method'] == 'Similarity based') & (df['delta'] ==  delta_vec[i])]['is_correct']) for i in range(len(delta_vec))]
# plt.plot(delta_vec, g_t_d)
# plt.plot(delta_vec, s_t_d)
# plt.title('Delta plot')
# plt.legend(['Distance based','Similarity based'])
#plt.plot(total_in_each_diam,'--')
# plt.show()


#N plot

sns.catplot(data=df, x='N', y='is_correct', kind="point", hue='Method', 
       markers=["o", "s"], linestyles=["-", "--"],legend=True,legend_out=False)    
plt.ylabel( "Correct ratio")
# g_t_d = [sum(df[(df['Method'] == 'Distance based') & (df['N'] ==  N_vec[i])]['is_correct']) for i in range(len(N_vec))]
# s_t_d = [sum(df[(df['Method'] == 'Similarity based') & (df['N'] ==  N_vec[i])]['is_correct']) for i in range(len(N_vec))]
# plt.plot(N_vec, g_t_d)
# plt.plot(N_vec, s_t_d)
# plt.title('N plot')
# plt.legend(['Distance based','Similarity based'])
#plt.plot(total_in_each_diam,'--')
plt.show()




print("Done!")
