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
import sys, os, platform
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
sys.path.append(os.path.join(sys.path[0],'spectraltree'))
import reconstruct_tree
import utils
import generation
from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference


num_taxa = 16
N = 1000
reference_tree = utils.balanced_binary(num_taxa)
reference_tree.print_plot()

data = simulate_discrete_chars(N, reference_tree, Jc69(), mutation_rate=generation.Jukes_Cantor().p2t(0.95), )
raxml = reconstruct_tree.RAxML()
tree = raxml(data)

tree.print_plot()

print("")
print("Data in DNAcharacterMatrix:")
print("symmetric_difference: ",symmetric_difference(reference_tree, tree))
RF,F1 = reconstruct_tree.compare_trees(reference_tree, tree)
print("raxml: ")
print("RF = ",RF)
print("F1% = ",F1)
print("")


###########################################################################
##                   TEST WITH NUMPY DATA
###########################################################################

jc = generation.Jukes_Cantor()
mutation_rate = [jc.p2t(0.95)]

observations = generation.simulate_sequences_ordered(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate)

raxml = reconstruct_tree.RAxML()
tree = raxml(data)
print("")
print("Data in numpy array:")
print("symmetric_difference: ",symmetric_difference(reference_tree, tree))
RF,F1 = reconstruct_tree.compare_trees(reference_tree, tree)
print("raxml: ")
print("RF = ",RF)
print("F1% = ",F1)
print("")


###########################################################################
##                   TEST WITHOUT CLASS
###########################################################################
from dendropy.interop import raxml

reference_tree = utils.balanced_binary(16)
reference_tree.print_plot()

data = simulate_discrete_chars(1000, reference_tree, Jc69(), mutation_rate=generation.Jukes_Cantor().p2t(0.95), )

if platform.system() == 'Windows':
    # Windows version:
    rx = raxml.RaxmlRunner(raxml_path = os.path.join(os.path.dirname(sys.path[0]),r'spectraltree\raxmlHPC-SSE3.exe'))
elif platform.system() == 'Darwin':
    #MacOS version:
    rx = raxml.RaxmlRunner()
elif platform.system() == 'Linux':
    #Linux version
    rx = raxml.RaxmlRunner(raxml_path = os.path.join(os.path.dirname(sys.path[0]),'spectraltree/raxmlHPC-SSE3-linux'))

tree = rx.estimate_tree(char_matrix=data, raxml_args=["-T 2"])

type(data)

#tree.print_plot()

print("symmetric_difference: ",symmetric_difference(reference_tree, tree))
RF,F1 = reconstruct_tree.compare_trees(reference_tree, tree)
print("raxml: ")
print("RF = ",RF)
print("F1% = ",F1)
print("")
