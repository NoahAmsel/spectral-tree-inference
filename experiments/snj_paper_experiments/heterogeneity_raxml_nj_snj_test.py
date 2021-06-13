import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spectraltree
import spectraltree.compare_methods as compare_methods
from spectraltree.raxml_reconstruction import raxml_gamma_corrected_distance_matrix

plot_results = 1
class params:    
    alpha = 1
    score_func = spectraltree.sv2

def generate_sequences_gamma(tree,num_taxa,jc,gamma_vec,basic_rate):
    observations = np.zeros((num_taxa,len(gamma_vec)))
    obs,meta_data = spectraltree.simulate_sequences(1, tree_model=tree, seq_model=jc, 
            mutation_rate=basic_rate*gamma_vec[0], alphabet="DNA")
    observations[:,0] = obs[0].T
    for idx,gamma in enumerate(gamma_vec[1:]):        
        obs,meta_data = spectraltree.simulate_sequences(1, tree_model=tree, seq_model=jc, 
            mutation_rate=basic_rate*gamma, alphabet="DNA")
        observations[:,idx+1]=obs[0].T
    return observations,meta_data

def gamma_func(P,a):
    return (3/4)*a*( (1-(4/3)*P)**(-1/a)-1 )


# set alpha parameters

#alpha_vec = [5,10]
#for alpha in alpha_vec:
#    gamma_vec = np.random.gamma(alpha,1/alpha,(1,N))
#    hist, bin_edges = np.histogram(gamma_vec,20)
#    bin_centers = 0.5*(bin_edges[0:-1]+bin_edges[1:])
#    plt.plot(bin_centers,hist,'s')
#plt.show()
#a=1


#plt.hist(gamma_vec, bins='auto')
num_itr = 5
num_taxa = 256
jc = spectraltree.Jukes_Cantor()
nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix) 
snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix) 
reference_tree = spectraltree.balanced_binary(num_taxa)
N_vec = np.arange(100,1000,100)
gamma_shape_vec = [5,10,15,20]
df = pd.DataFrame(columns=['method', 'RF','n','gamma_shape'])
base_rate = jc.p2t(0.9)

for i in range(num_itr):
    for gamma_shape in gamma_shape_vec:
    
        #gamma_vec = np.random.gamma(alpha,1/alpha,max(N_vec))
        observations,taxa_meta = spectraltree.simulate_sequences_gamma(max(N_vec), reference_tree, jc, base_rate, gamma_shape, 
            block_size=10, alphabet="DNA")
        # observations, taxa_meta = spectraltree.simulate_sequences(max(N_vec), tree_model=reference_tree, seq_model=jc, mutation_rate=base_rate, alphabet="DNA")
        #observations,taxa_meta = generate_sequences_gamma(reference_tree,num_taxa,jc,
        #    gamma_vec,basic_rate)
        for n in N_vec:
            print('iteration: ',i, 'gamma_shape: ',gamma_shape,' n: ',n)       
        
            # estimate distance with hetero via raxml
            dist_raxml = raxml_gamma_corrected_distance_matrix(observations[:,:n], taxa_meta)
        
            # NJ - estimate tree with heterogeneity        
            tree_nj_het = spectraltree.NeighborJoining(lambda x: x).reconstruct_from_similarity(np.exp(-dist_raxml), taxa_meta)
            RF,F1 = spectraltree.compare_trees(tree_nj_het, reference_tree)
            df = df.append({'method': 'NJ-het', 'RF': RF,'n': n,'gamma_shape':gamma_shape}, ignore_index=True)

            # NJ - estimate tree via standard distance        
            #tree_nj_standard = nj(observations[:,:n],taxa_meta)
            #RF,F1 = spectraltree.compare_trees(tree_nj_standard, reference_tree)
            #df = df.append({'method': 'NJ', 'RF': RF,'n': n,'gamma_shape':gamma_shape}, ignore_index=True)
        
            #SNJ - estimate tree with heterogeneity
            tree_snj_het = spectraltree.SpectralNeighborJoining(lambda x: x).reconstruct_from_similarity(np.exp(-dist_raxml), taxa_meta)
            RF,F1 = spectraltree.compare_trees(tree_snj_het, reference_tree)
            df = df.append({'method': 'SNJ-het', 'RF': RF,'n': n,'gamma_shape':gamma_shape}, ignore_index=True)
        
            # SNJ standrd SNJ
            #tree_snj_standard = nj(observations[:,:n],taxa_meta)
            #RF,F1 = spectraltree.compare_trees(tree_snj_standard, reference_tree)
            #df = df.append({'method': 'SNJ', 'RF': RF,'n': n,'gamma_shape':gamma_shape}, ignore_index=True)

            folder = "./experiments/snj_paper_experiments/results/"
            compare_methods.save_results(df,'20200901_heterogeneity_cat',folder)




        
a =1


# plot shape of distnace function
alpha_vec = [5,10,20,30,50]
p_vec = np.arange(0,1,0.05)
N_vec = 10000
het_dist = np.zeros((len(alpha_vec),len(p_vec)))
het_dist = np.zeros((len(alpha_vec),len(p_vec)))
hom_dist = np.zeros(len(p_vec))
for p_idx,p in enumerate(p_vec):
    for alpha_idx,alpha in enumerate(alpha_vec):            
        het_dist[alpha_idx,p_idx] = gamma_func(p,alpha)
    hom_dist[p_idx] = -1*(3/4)*np.log(1-(4/3)*p)    

plt.plot(p_vec,het_dist.T)
plt.plot(p_vec,hom_dist)
plt.legend(('5','10','20','30','50','log'))
plt.show()

N =10000
