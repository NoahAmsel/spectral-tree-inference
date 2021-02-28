import unittest

import time

from spectraltree import utils, generation, reconstruct_tree

from dendropy.model.discrete import simulate_discrete_chars, Jc69, Hky85
from dendropy.model.discrete import simulate_discrete_chars

N = 100000
num_taxa = 16

#################################
## Tree generation
#################################
print("Creating tree")
jc = generation.Jukes_Cantor()
hky = generation.HKY(kappa = 1)
#mutation_rate = [jc.p2t(0.95)]
mutation_rate = [0.1]

#reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=0.5)
# reference_tree = utils.lopsided_tree(num_taxa)
reference_tree = utils.balanced_binary(num_taxa)
for x in reference_tree.preorder_edge_iter():
    x.length = 0.5

print("Genration observations by JC and HKY")
#observationsJC = FastCharacterMatrix(generation.simulate_sequences_ordered(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate))
# observationsHKY = FastCharacterMatrix(generation.simulate_sequences_ordered(N, tree_model=reference_tree, seq_model=hky, mutation_rate=mutation_rate))
t = time.time()
observationsJC, _ = utils.charmatrix2array(simulate_discrete_chars(N, reference_tree, Jc69(), mutation_rate=mutation_rate[0]))
print("Time to generate JC:", time.time()-t)
t = time.time()
observationsHKY, metadataHKY = utils.charmatrix2array(simulate_discrete_chars(N, reference_tree, Hky85(kappa = 1), mutation_rate=mutation_rate[0]))
print("Time to generate HKY:", time.time()-t)

t = time.time()
S_JC = reconstruct_tree.JC_similarity_matrix(observationsJC)
print()
print("Time to compute JC similarity:", time.time()-t)
print("JC similarity")
print(S_JC)

t = time.time()
S_HKY = reconstruct_tree.HKY_similarity_matrix(observationsHKY, metadataHKY)
print()
print("Time to compute HKY similarity:", time.time()-t)
print("HKY similarity")
print(S_HKY)