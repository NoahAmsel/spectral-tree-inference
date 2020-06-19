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
sys.path.append(os.path.join(sys.path[0],'spectraltree'))
import reconstruct_tree
import utils
import time
import generation
from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
import pandas as pd

num_taxa = [128,256,512,1024,2048,4096]
#num_taxa = [8,16,32]
N = 1000
tree_list = [utils.balanced_binary(m) for m in num_taxa]

runtime = []
RF = []
F1 = []
for reference_tree in tree_list:
    data = simulate_discrete_chars(N, reference_tree, Jc69(), mutation_rate=generation.Jukes_Cantor().p2t(0.95), )
    time_s = time.time()
    raxml = reconstruct_tree.RAxML()
    tree = raxml(data)
    runtime.append(time.time()-time_s)
    print('runtime is ',runtime)
    RF_,F1_ = reconstruct_tree.compare_trees(reference_tree, tree)
    RF.append(RF_)
    F1.append(F1_)

    #tree.print_plot()
res = pd.DataFrame({"N": num_taxa, "runtime": runtime, "Accuracy": F1})
print(res)
print('')

# print("runtime")
# print(runtime)
# print("accuracy")
# print(F1)

