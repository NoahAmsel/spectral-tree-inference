##########
# This file does not run -- it looks like it calls functions that no longer exist
##########

import datetime
import pickle
import os.path
import time

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import dendropy
import copy

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))

import utils
import generation
import reconstruct_tree


N = 1000
num_taxa = 256
jc = generation.Jukes_Cantor()
mutation_rate = [jc.p2t(0.95)]
# Construct data of type 'tree' from class dendropy
namespace =  utils.default_namespace(num_taxa)
tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=1)
for x in tree.preorder_edge_iter():
    x.length = 1
#tree = utils.lopsided_tree(num_taxa=num_taxa, namespace=namespace)
#tree = utils.balanced_binary(num_taxa, namespace=namespace)
tree.is_rooted = True

for i in tree.bipartition_edge_map:
    if tree.bipartition_edge_map[i] == tree.seed_node.child_edges()[0]:
        taxa_half1 =  set(i.leafset_taxa(tree.taxon_namespace) )
    if tree.bipartition_edge_map[i] == tree.seed_node.child_edges()[1]:
        taxa_half2 =  set(i.leafset_taxa(tree.taxon_namespace) )
tree.is_rooted = False

#taxa_half1 = set([taxon for taxon in tree.taxon_namespace if int(taxon.label)<num_taxa/2])
#taxa_half2 = set([taxon for taxon in tree.taxon_namespace if int(taxon.label)>=num_taxa/2])
tree1 = tree.extract_tree_with_taxa(taxa=taxa_half1)
tree2 = tree.extract_tree_with_taxa(taxa=taxa_half2)

namespace1 = [l.taxon for l in tree1.leaf_nodes()]
namespace1.sort(key = lambda x: int(x.label))
namespace1 = dendropy.TaxonNamespace(namespace1)
namespace2 = [l.taxon for l in tree2.leaf_nodes()]
namespace2.sort(key = lambda x: int(x.label))
namespace2 = dendropy.TaxonNamespace(namespace2)

tree1.taxon_namespace = namespace1
tree2.taxon_namespace = namespace2

# namespace1 = namespace
# T1 = utils.lopsided_tree(num_taxa=int(num_taxa/2), namespace=dendropy.TaxonNamespace(namespace[:int(num_taxa/2)]))
# T2 = utils.lopsided_tree(num_taxa=int(num_taxa/2), namespace=dendropy.TaxonNamespace(namespace[int(num_taxa/2):]))
# T1.print_plot
# TT1 = T1.extract_tree_with_taxa(namespace)
# TT2 = T2.extract_tree_with_taxa(namespace)
# tree.seed_node.set_child_nodes([TT1.seed_node,TT2.seed_node])
# create sequences for each node, not sure what is th

observations = generation.simulate_sequences_ordered(N, tree_model=tree, seq_model=jc, mutation_rate=mutation_rate)

t1= time.time()
S = reconstruct_tree.JC_similarity_matrix(observations)
t2 = time.time()
TT = reconstruct_tree.join_trees_with_spectral_root_finding(S,tree1, tree2, namespace = tree.taxon_namespace)
# TT.print_plot(width = 80)
# T.print_plot(width = 80)
t3= time.time()

RF,F1 = reconstruct_tree.compare_trees(tree, TT)
print("Spectral: ")
print("RF = ",RF)
print("F1% = ",F1)
print("")
print("time:")
print("    Build Sim.: ", t2-t1)
print("    Find roots: ", t3-t2)
