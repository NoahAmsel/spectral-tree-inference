import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))
import numpy as np
import utils
import generation
import reconstruct_tree
import dendropy
import scipy
import time
from itertools import product
import matplotlib.pyplot as plt
import cProfile

from dendropy.model.discrete import simulate_discrete_chars, Jc69, Hky85
from dendropy.calculate.treecompare import symmetric_difference

## compute the similarity matrix under HKY model
def HKY_similarity_matrix(observations, classes=None, verbose = False):
    m, N = observations.shape
    if classes is None:
        classes = np.unique(observations)
    k = len(classes)
    # From Tamura, K., and M. Nei. 1993
    # for each pair of sequences, 
    # 1. estimate the average base frequency for pairs of sequences
    # 2. compute purine transition proportion P1 (A <-> G)
    # 3. compute pyrimidine transition proportion P2 (T <-> C)
    # 3. compute transversion proportion Q (A <-> C, A <-> T, G <-> C, G <-> T)

    if verbose: print("Computing the average base frequency for each pair of sequences...")
    g = {}
    for x in classes:
        obs_x = observations == x
        g[x] = np.array([np.mean(np.hstack([a, b])) for a, b in product(obs_x, repeat = 2)]).reshape((m, m))
    
    g["R"] = g["A"] + g["G"]
    g["Y"] = g["T"] + g["C"]
    
    # compute transition and transversion proportion
    if verbose: print("Computing transition and transversion proportion for each pair of sequences...")
    P = {}
    for i, x in enumerate(classes):
        other_classes = np.delete(classes, i)
        for y in other_classes:
            P_x_y = np.array([np.mean(np.logical_and(a == x, b == y)) for a, b in product(observations, repeat = 2)]).reshape((m, m))
            P[x + y] = P_x_y
            
    P_1 = P['AG'] + P["GA"]
    P_2 = P['CT'] + P['TC']
    Q = P['AC'] + P['CA'] + P['AT'] + P['TA'] +\
        P['GC'] + P['CG'] + P['GT'] + P['TG']

    # compute the similarity (formula 7)
    if verbose: print("Computing similarity matrix")
    R = (1 - g["R"]/(2 * g["A"] * g["G"]) * P_1 - 1 / (2 * g["R"]) * Q)
    Y = (1 - g["Y"]/(2 * g["T"] * g["C"]) * P_2 - 1 / (2 * g["Y"]) * Q)
    T = (1 - 1/(2 * g["R"] * g["Y"]) * Q)
    S = np.sign(R) * (np.abs(R))**(2 * g["A"] * g["G"] / g["R"])
    S += np.sign(Y) * (np.abs(Y))**(2 * g["T"] * g["C"] / g["Y"])
    S += np.sign(T) * (np.abs(T))**(2 * (g["R"] * g["Y"] - g["A"] * g["G"] * g["Y"] / g["R"] - g["T"] * g["C"] * g["R"] / g["Y"]))

    return S

tree_path = os.path.join(os.path.dirname(sys.path[0]), "data/NY_H3N2.newick")
fasta_path = os.path.join(os.path.dirname(sys.path[0]), "data/NY_H3N2.fasta")
H3N2_tree = dendropy.Tree.get(path=tree_path, schema="newick")
H3N2_dna = dendropy.DnaCharacterMatrix.get(file=open(fasta_path, "r"), schema="fasta")

N = 1000 
data_HKY = simulate_discrete_chars(N, H3N2_tree, Hky85(kappa = 2), mutation_rate=0.1)
ch_list = list()
for t in data_HKY.taxon_namespace:
    ch_list.append([x.symbol for x in data_HKY[t]])
ch_arr = np.array(ch_list)
identical = np.array([np.mean(a == b) for a, b in product(ch_arr, repeat = 2)])

#start_time = time.time()
#cProfile.run('S = HKY_similarity_matrix(ch_arr)')
#compute_s_time = time.time() - start_time
#print("--- %s seconds ---" % compute_s_time)
threshold = 128
spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.RAxML,
                                                              HKY_similarity_matrix)
tree_rec = spectral_method.deep_spectral_tree_reonstruction(ch_arr, HKY_similarity_matrix, 
                                                            taxon_namespace = H3N2_tree.taxon_namespace, 
                                                            threshhold = threshold)

Deep_nj_RF, Deep_nj_RF_F1 = reconstruct_tree.compare_trees(tree_rec, H3N2_tree)
print("SNJ: ")
print("RF = ", Deep_nj_RF)
print("F1% = ", Deep_nj_RF_F1)
print("")



print(tree_path)