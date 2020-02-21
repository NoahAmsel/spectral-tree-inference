
#
import sys, os, platform
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))
import reconstruct_tree as rt
import utils
import generation
from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
import numpy as np
import compare_methods

# Create tree with m taxa according to the coalescent model with Jukes Cantor evolution model
m = 64
#tree = unrooted_birth_death_tree(birth_rate=4., death_rate=0, num_total_tips=m)
tree = utils.unrooted_pure_kingman_tree(utils.default_namespace(m), m)
tree_list = [tree]
jc = generation.Jukes_Cantor()
delta_vec = np.linspace(0.5,0.6,3)
methods = [rt.Reconstruction_Method(rt.neighbor_joining), reconstruct_tree.Reconstruction_Method(reconstruct_tree.estimate_tree_topology)]
mutation_rates = [jc.similarity2t(delta)  for delta in delta_vec]
Ns = np.linspace(100,1000,10).astype(int)