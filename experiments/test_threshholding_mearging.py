import spectraltree
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import scipy
import random
from spectraltree.utils import tree2distance_matrix
import datetime

## Arguments
similarity_thresholds =  np.arange(0,0.045,0.01)
partition_in_middle = True
#similarity_threshold = 0.00
num_experiments = 200
num_taxa = 2048   # Number of terminal nodes
n = 400        # Number of independent samples (sequence length)   
n_max = 3000
jc = spectraltree.Jukes_Cantor()   #set evolution process to the Jukes Cantor model
mutation_rate = jc.p2t(0.96)        #set mutation rate between adjacent nodes to 1-0.9=0.1


## Generate tree and observations
reference_tree = spectraltree.balanced_binary(num_taxa)
observations, taxa_meta = spectraltree.simulate_sequences(n_max, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")
S_exact = spectraltree.JC_similarity_matrix(observations)

res_thresh = np.zeros((num_experiments, similarity_thresholds.size))
percent_of_zeroed = np.zeros((num_experiments, similarity_thresholds.size))
for thresh_i,similarity_threshold in enumerate(similarity_thresholds):
    ct = datetime.datetime.now()
    print(ct, " Starting testing Threshold:", similarity_threshold)
    #Loop with thresholding
    for i in range(num_experiments):
        ## Split trees
        entry_list = list(reference_tree.split_bitmask_edge_map.items())
        if partition_in_middle == True:
            mask_of_subtree = np.array([True]*int(num_taxa/2) + [False]*int(num_taxa/2), 'bool')
        else:
            while True:
                random_entry = random.choice(entry_list)
                mask_of_subtree = taxa_meta.bipartition2mask(random_entry[1].bipartition)
                if (sum(mask_of_subtree) >2) and (sum(~mask_of_subtree) >2):
                    break


        small_meta1 = spectraltree.TaxaMetadata(taxa_meta.taxon_namespace, taxa_meta.mask2taxa(mask_of_subtree), taxa_meta.alphabet)
        small_meta2 = spectraltree.TaxaMetadata(taxa_meta.taxon_namespace, taxa_meta.mask2taxa(~mask_of_subtree), taxa_meta.alphabet)
        T1 = reference_tree.extract_tree_with_taxa(list(small_meta1))
        T2 = reference_tree.extract_tree_with_taxa(list(small_meta2))

        T1.mask = taxa_meta.tree2mask(T1)
        T2.mask = taxa_meta.tree2mask(T2)
        
        # mearge
        ob = observations[:,np.random.permutation(n_max)]
        S = spectraltree.JC_similarity_matrix(ob[:,:n])
        S_thresholded = S * (S>similarity_threshold)
        num_of_not_zeroed_out = sum(sum((S>similarity_threshold)))
        percent_of_zeroed[i,thresh_i] = 1- num_of_not_zeroed_out/S.size
        print("Percent of zeroed out:", percent_of_zeroed[i,thresh_i])
        #mearge
        mearged_tree_thresh = spectraltree.join_trees_with_spectral_root_finding_ls(S, T1, T2, merge_method="least_square", taxa_metadata = taxa_meta, similarity_threshold = similarity_threshold)
        res_thresh[i,thresh_i] = spectraltree.topos_equal(reference_tree,mearged_tree_thresh)
        # mearged_tree = spectraltree.join_trees_with_spectral_root_finding_ls(S, T1, T2, merge_method="angle", taxa_metadata = taxa_meta)
        # res_reg.append(spectraltree.topos_equal(reference_tree,mearged_tree))
        #print(i,". Are trees equal (thresholded):", res_thresh[-1])
        print(i,". Are trees equal:", res_thresh[i,thresh_i])
        #RF_nj,F1 = spectraltree.compare_trees(mearged_tree, reference_tree)
        #print('Normalized RF for NJ:',np.round_(RF_nj/(2*num_taxa-6),2))
    print("Mean sucess for threshold %f is:"%similarity_threshold, np.mean(res_thresh[:,thresh_i]))

    
print()
np.savetxt('percent_of_zeroed_ver2_taxa_%d_num_exp_%d_n_%d.csv'%(num_taxa,num_experiments,n),percent_of_zeroed, delimiter=',' )
np.savetxt('temp_res_thresh_ver2_taxa_%d_num_exp_%d_n_%d.csv'%(num_taxa,num_experiments,n),res_thresh, delimiter=',' )
# percent_of_zeroed = np.genfromtxt('temp_percent_of_zeroed.csv', delimiter=',' )
# res_thresh = np.genfromtxt('temp_res_thresh.csv', delimiter=',' )
mean_zeroed = np.mean(percent_of_zeroed, axis = 0)
#plt.bar(list(map(str,similarity_thresholds)), sum(res_thresh,0)/num_experiments)

fig, ax = plt.subplots()

p1 = ax.bar(list(map(str,similarity_thresholds)), sum(res_thresh,0)/num_experiments, width=0.35, label='Correct')

ax.axhline(0, color='grey', linewidth=0.8)
ax.set_ylabel('Scores')
#ax.set_xticks(ind, labels=['G1', 'G2', 'G3', 'G4', 'G5'])
ax.legend()

# Label with label_type 'center' instead of the default 'edge'
ax.bar_label(p1,labels = mean_zeroed, fmt='%.2f')#, label_type='top')

plt.show()


print()
print("sucsess percent (thresholded):", sum(res_thresh)/len(res_thresh))
print("sucsess percent (regular    ):", sum(res_reg)/len(res_reg))