from time import time as _t

import numpy as np

import spectraltree
import metricNearness

N = 100
num_taxa = 60

#################################
## Tree generation
#################################
print("Creating tree")
jc = spectraltree.Jukes_Cantor()
hky = spectraltree.HKY(kappa = 2)
mutation_rate = [jc.p2t(0.95)]
# mutation_rate = [0.1]

#reference_tree = spectraltree.unrooted_birth_death_tree(num_taxa, birth_rate=0.5)
reference_tree = spectraltree.lopsided_tree(num_taxa)
# reference_tree = spectraltree.balanced_binary(num_taxa)
# for x in reference_tree.preorder_edge_iter():
#     x.length = 0.5
print("Genration observations by JC and HKY")

observationsJC, metaJC = spectraltree.simulate_sequences(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")

#################################
## SNJ - Jukes_Cantor
#################################
t0 = _t()
snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)   
sim = spectraltree.JC_similarity_matrix(observationsJC, metaJC)
inside_log = np.clip(sim, a_min=1e-16, a_max=None)
dis = - np.log(inside_log)
disC = np.array(metricNearness.metricNearness(dis))
simC = np.exp(-disC)
tree_rec = snj.reconstruct_from_similarity(sim, taxa_metadata = metaJC)
tree_recC = snj.reconstruct_from_similarity(simC, taxa_metadata = metaJC)
RFC,F1C = spectraltree.compare_trees(tree_recC, reference_tree)
RF,F1 = spectraltree.compare_trees(tree_rec, reference_tree)

print("###################")
print("SNJ - Jukes_Cantor:")
print("time:", _t() - t0)
print("RF = ",RF, "    F1% = ",F1)
print("RFC = ",RFC, "    F1C% = ",F1C)
print("")

