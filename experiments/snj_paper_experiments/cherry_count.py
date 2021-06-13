import numpy as np

import spectraltree

def cherry_count_for_tree(tree):
    cherry_count = 0
    for node in tree.leaf_nodes():
        if node.parent_node.child_nodes()[0].is_leaf() and node.parent_node.child_nodes()[1].is_leaf():
            cherry_count += 1
    cherry_count = cherry_count/2
    return cherry_count


N = 50
num_taxa = 64
jc = spectraltree.Jukes_Cantor()
mutation_rate = [jc.p2t(0.95)]
np.random.seed(0)
reference_tree = spectraltree.unrooted_birth_death_tree(num_taxa, birth_rate=1)
#reference_tree = spectraltree.lopsided_tree(num_taxa)
#reference_tree = spectraltree.balanced_binary(num_taxa)
for x in reference_tree.preorder_edge_iter():
    x.length = 1
np.random.seed(0)

cherry_count = cherry_count_for_tree(reference_tree)
print("Orig Cherry count:", cherry_count)
############################################

observations,meta = spectraltree.simulate_sequences(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate,rng=np.random, alphabet = 'DNA')
spectral_method = spectraltree.SpectralTreeReconstruction(spectraltree.NeighborJoining,spectraltree.JC_similarity_matrix)   
tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, spectraltree.JC_similarity_matrix,
                                     taxa_metadata= meta,
                                     threshhold = 8 ,min_split = 3, merge_method = "least_square", verbose=False)

str_cherry_count = cherry_count_for_tree(tree_rec)
print("STR Cherry count:", str_cherry_count)
##########################################
snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)   
tree_rec = snj(observations, meta)

snj_cherry_count = cherry_count_for_tree(tree_rec)
print("SNJ Cherry count:", snj_cherry_count)

###############################################3
nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)   
tree_rec = nj(observations, meta)

nj_cherry_count = cherry_count_for_tree(tree_rec)
print("NJ Cherry count:", nj_cherry_count)
