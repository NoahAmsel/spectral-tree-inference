import dendropy
import numpy as np
import scipy.spatial.distance

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))

import utils
import generation
import reconstruct_tree

def hamming_dist_missing_values(vals, missing_val =0):
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    missing_array = (vals==missing_val)
    pdist_xor = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_xor(u,v))))
    pdist_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_or(u,v))))

    # Fix Hamming matrix for missing values

    return (hamming_matrix*vals.shape[1] - pdist_xor) / (np.ones_like(hamming_matrix) * vals.shape[1] - pdist_or)



file_name = "data/steps-128_0.nex"
print("Reading tree...")
post_trees = dendropy.TreeList()
post_trees.read(
    file=open(file_name, "r"),
    schema="nexus")
tree = post_trees[0]
print("Reading sequences...")
rna1 = dendropy.RnaCharacterMatrix.get(file=open(file_name), schema="nexus")
print("translating sequeences...")
ch_list = list()
for t in rna1.taxon_namespace:
    ch_list.append([x.symbol for x in rna1[t]])

leafs_idx = [i.label[0] != " " for i in rna1.taxon_namespace]

ch_list_num = np.array(ch_list)
ch_list_num = ch_list_num[leafs_idx]
ch_list_num = np.where(ch_list_num=='A', 1, ch_list_num) 
ch_list_num = np.where(ch_list_num=='C', 2, ch_list_num) 
ch_list_num = np.where(ch_list_num=='G', 3, ch_list_num) 
ch_list_num = np.where(ch_list_num=='U', 4, ch_list_num) 
ch_list_num = np.where(ch_list_num=='-', 0, ch_list_num) 
ch_list_num = ch_list_num.astype('int')

# Computing JC_similarity
print("Computing JC Similarity")
hamming_matrix_corrected = hamming_dist_missing_values(ch_list_num)
classes = np.unique(ch_list_num)
k = len(classes)
inside_log = 1 - hamming_matrix_corrected*k/(k-1)
S =  inside_log**(k-1)


t = reconstruct_tree.neighbor_joining(S)

print("bye")

