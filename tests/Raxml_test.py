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
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))

from utils import balanced_binary
from generation import Jukes_Cantor
from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference

reference_tree = balanced_binary(128)
reference_tree.print_plot()

data = simulate_discrete_chars(1000, reference_tree, Jc69(), mutation_rate=Jukes_Cantor().p2t(0.95), )

rx = raxml.RaxmlRunner(raxml_path = os.path.join(os.path.dirname(sys.path[0]),'spectraltree'))
tree = rx.estimate_tree(char_matrix=data, raxml_args=["-T 2"])

tree.print_plot()

print(symmetric_difference(reference_tree, tree))
