import datetime
import pickle
import os.path

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import dendropy

import spectraltree
import utils
import generation
import reconstruct_tree

def checkSol(reference_tree, inferred_tree):
    inferred_tree.update_bipartitions()
    reference_tree.update_bipartitions()
    false_positives, false_negatives = dendropy.calculate.treecompare.false_positives_and_negatives(reference_tree, inferred_tree, is_bipartitions_updated=True)
    total_reference = len(reference_tree.bipartition_encoding)
    total_inferred = len(inferred_tree.bipartition_encoding)
    
    true_positives = total_inferred - false_positives
    precision = true_positives / total_inferred
    true_positives = total_inferred - false_positives
    recall = true_positives / total_reference


    F1 = 100* 2*(precision * recall)/(precision + recall)
    RF = false_positives + false_negatives
    return RF, F1


N = 400
jc = generation.Jukes_Cantor()
mutation_rate = [jc.p2t(0.95)]
# Construct data of type 'tree' from class dendropy
tree = utils.balanced_binary(num_taxa=128) 
tree = utils.lopsided_tree(128) 
#tree.is_rooted = True

# create sequences for each node, not sure what is th
observations = spectraltree.simulate_sequences(N, tree_model=tree, seq_model=jc, mutation_rate=mutation_rate)
# pickle.dump(observations, open("observations_temp.pkl", 'wb'))
#observations = pickle.load(open("observations_temp.pkl", 'rb'))
print(observations.shape)

#sequences = seqgen(seq_len=1000, tree_model=tree)
# extract numpy matrix
#observations, alphabet = utils.charmatrix2array(sequences)

## NJ
#dm = reconstruct_tree.JC_similarity_matrix(observations)
#tr = reconstruct_tree.neighbor_joining(similarity_matrix=dm)
#tr.print_plot()

## spectral_tree_reonstruction

S = reconstruct_tree.JC_similarity_matrix(observations)
w,v = scipy.linalg.eigh(S)
plt.plot(v[:,-2])
TT = reconstruct_tree.spectral_tree_reonstruction(S, namespace = tree.taxon_namespace)
T_sv2 = reconstruct_tree.estimate_tree_topology(S,namespace = tree.taxon_namespace)
#TT.print_plot(width=80)

RF,F1 = checkSol(tree, T_sv2)
print("SV2: ")
print("RF = ",RF)
print("F1% = ",F1)
print("")


RF,F1 = checkSol(tree, TT)
print("Spectral: ")
print("RF = ",RF)
print("F1% = ",F1)
print("")

