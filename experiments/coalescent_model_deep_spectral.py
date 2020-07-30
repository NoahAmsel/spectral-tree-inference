import sys, os, platform
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))
import reconstruct_tree
import utils
import time
import generation
import compare_methods
from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
from dendropy.simulate.treesim import birth_death_tree, pure_kingman_tree, mean_kingman_tree 
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

def generate_figure(df,x='n',y='RF',hue="method", kind="point"):
    col = x
    dodge = 0.1*(df['method'].nunique() - 1)
    sns.catplot(data=df, x=x, y=y, kind="point", hue=hue,  dodge=dodge,\
       legend=True)
    plt.show()


#df = pkl.load( open( "./data/coalescent_m_400.pkl", "rb" ) )
#generate_figure(df,y='runtime')

num_taxa = 1000
reference_tree = utils.unrooted_pure_kingman_tree(num_taxa)
jc = generation.Jukes_Cantor(num_classes=4)
mutation_rate = jc.p2t(0.9)
N_vec = [800,1000,1200,1400]
num_reps = 5
# set methods
nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix) 
raxml = reconstruct_tree.RAxML()
spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.RAxML,reconstruct_tree.JC_similarity_matrix)




df = pd.DataFrame(columns=['method', 'runtime', 'RF','n'])
for i in np.arange(num_reps):    
    for n in N_vec:
        print('iteration ',i,' length ',n)
        observations, taxa_meta = generation.simulate_sequences(n, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")
    
        # run raxml
        print("RAXML")
        t_s = time.time()
        tree_raxml = raxml(observations, taxa_meta)
        runtime = time.time()-t_s
        print(runtime)
        RF,F1 = reconstruct_tree.compare_trees(tree_raxml, reference_tree)       
        df = df.append({'method': 'RaXML', 'runtime': runtime, 'RF': RF,'F1':F1,'n': n}, ignore_index=True) 
        
        
        # run deep spectral    
        print("Spectral deep")
        t_s = time.time()
        tree_spectral = spectral_method.deep_spectral_tree_reconstruction(observations, reconstruct_tree.JC_similarity_matrix,
                                         taxa_metadata= taxa_meta,
                                        threshhold = 100 ,min_split = 5, merge_method = "least_square", verbose=False)
        runtime = time.time()-t_s
        print(runtime)
        tree_spectral.write(path="temp.tre", schema="newick")
        RF,F1 = reconstruct_tree.compare_trees(tree_spectral, reference_tree)       
        df = df.append({'method': 'STR+RAXML', 'runtime': runtime, 'RF': RF,'F1':F1,'n': n}, ignore_index=True) 
        
        # run raxml with deep spectral initilization
        print("RAXML with init")
        t_s = time.time()
        tree_raxml = raxml(observations, taxa_meta, raxml_args="-T 2 --JC69 -c 1 -t temp.tre")
        runtime = time.time()-t_s
        print(runtime)
        RF,F1 = reconstruct_tree.compare_trees(tree_raxml, reference_tree)       
        df = df.append({'method': 'RAXML (init)', 'runtime': runtime, 'RF': RF,'F1':F1,'n': n}, ignore_index=True) 
        
pickle_out = open("./data/coalescent_m_1000.pkl","wb")
pkl.dump(df, pickle_out)
pickle_out.close()

a = 1
    

#results = compare_methods.experiment([reference_tree], jc, N_vec, methods=methods,\
#     mutation_rates = [mutation_rate], reps_per_tree=num_reps)
