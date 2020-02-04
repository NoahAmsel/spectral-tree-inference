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
import reconstruct_tree
import utils
import numpy as np 
import time
import generation
from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference


if platform.system() == 'Windows':
    # Windows version:
    rx = raxml.RaxmlRunner(raxml_path = os.path.join(os.path.dirname(sys.path[0]),r'spectraltree\raxmlHPC-SSE3.exe'))
elif platform.system() == 'Darwin':
    #MacOS version:
    rx = raxml.RaxmlRunner()
elif platform.system() == 'Linux':
    #Linux version
    rx = raxml.RaxmlRunner(raxml_path = os.path.join(os.path.dirname(sys.path[0]),'spectraltree/raxmlHPC-SSE3-linux'))

m_vec = np.arange(100,500,100)
#reference_trees = [utils.balanced_binary(m) for m in m_vec]
reference_trees = [utils.unrooted_pure_kingman_tree(utils.default_namespace(m), pop_size=m, rng=None) for m in m_vec]
n_vec = [200,400,600]
RF = np.zeros((len(n_vec),len(m_vec)))
runtime = np.zeros((len(n_vec),len(m_vec)))
for tree_idx,tree in enumerate(reference_trees):
    print(tree_idx)
    for n_idx,n in enumerate(n_vec):        
        data = simulate_discrete_chars(n, tree, Jc69(), mutation_rate=generation.Jukes_Cantor().p2t(0.95))
        t_start = time.time()
        tree_raxml = rx.estimate_tree(char_matrix=data, raxml_args=["-T 2"])
        t_stop = time.time()
        runtime[n_idx,tree_idx] = t_stop-t_start
        print(runtime[n_idx,tree_idx])
        RF[n_idx,tree_idx],F1 = reconstruct_tree.compare_trees(tree_raxml, tree)



