"""
I downloaded using the instructions here and verified that it worked
http://www.sfu.ca/biology2/staff/dc/raxml/
For me, I had to add the argument -T 2 or I got an error about threads

Then I copied the raxml executable to /usr/local/bin/raxmlHPC
Dendropy will be looking for an executable with the name raxmlHPC
Make sure to include `raxml_args=["-T 2"]`

On other operating systems you may need to compile yourself using the directions here
https://cme.h-its.org/exelixis/web/software/raxml/hands_on.html
"""
# %%
import sys, os, platform
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))
import reconstruct_tree
import utils
import time
import generation
from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
import pandas as pd
import numpy as np

#num_taxa = [128,256,512,1024,2048,4096]
num_taxa = np.arange(200,2000,200)
n_itr = 10
#num_taxa = [8,16,32]
N = 1000
jc = generation.Jukes_Cantor()
mutation_rate = jc.p2t(0.95)
nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix) 
raxml = reconstruct_tree.RAxML()
#tree_list = [utils.balanced_binary(m) for m in num_taxa]


df = pd.DataFrame(columns=['method', 'runtime', 'RF','m'])
for m in num_taxa:
    for n_itr in range(n_itr):
        reference_tree = utils.unrooted_pure_kingman_tree(m)
        observations, taxa_meta = generation.simulate_sequences(N, tree_model=reference_tree, 
            seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")
        
        # NJ
        time_s = time.time()
        tree_rec = nj(observations,taxa_meta)
        runtime = time.time()-time_s
        RF,F1 = reconstruct_tree.compare_trees(reference_tree, tree_rec)
        print('NJ iteration: ',n_itr, ' num_taxa: ',m,' time: ',runtime)
    
        df = df.append({'method': 'nj', 'runtime': runtime, 'RF': RF,
            'm': m}, ignore_index=True)

        #RAXML
        time_s = time.time()
        tree_rec = raxml(observations,taxa_meta)
        runtime = time.time()-time_s
        RF,F1 = reconstruct_tree.compare_trees(reference_tree, tree_rec)
        print('RAxML iteration: ',n_itr, ' num_taxa: ',m,' time: ',runtime)
    
        df = df.append({'method': 'raxml', 'runtime': runtime, 'RF': RF,
            'm': m}, ignore_index=True)
print(df)
print('')

# print("runtime")
# print(runtime)
# print("accuracy")
# print(F1)

