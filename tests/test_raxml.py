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

import os.path
import platform
import unittest

from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference
from numpy.random import default_rng

import spectraltree

class TestRaxml(unittest.TestCase):
    def setUp(self):
        self.reference_tree = spectraltree.balanced_binary(32)

    def test_dendropy_data(self):
        data = simulate_discrete_chars(
            seq_len=2000,
            tree_model=self.reference_tree, 
            seq_model=Jc69(), 
            mutation_rate=spectraltree.Jukes_Cantor().p2t(0.95),
            rng=default_rng(123))

        raxml = spectraltree.RAxML()
        tree = raxml(data)
        RF = spectraltree.compare_trees(self.reference_tree, tree)
        print("RF (test_without_class):", RF[0])
        self.assertEqual(RF[0], 0)

    def test_numpy_data(self):
        jc = spectraltree.Jukes_Cantor()
        observations, taxa_meta = spectraltree.simulate_sequences(
            seq_len=2000, 
            tree_model=self.reference_tree, 
            seq_model=jc, 
            mutation_rate=jc.p2t(0.95),
            rng=default_rng(123),
            alphabet="DNA")

        raxml = spectraltree.RAxML()
        tree = raxml(observations, taxa_meta)
        RF = spectraltree.compare_trees(self.reference_tree, tree)
        print("RF (test_without_class):", RF[0])
        self.assertEqual(RF[0], 0)

if __name__ == "__main__":
    unittest.main()