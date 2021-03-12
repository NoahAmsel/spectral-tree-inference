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
            seq_len=500,
            tree_model=self.reference_tree, 
            seq_model=Jc69(), 
            mutation_rate=spectraltree.Jukes_Cantor().p2t(0.95),
            rng=default_rng(123))

        raxml = spectraltree.RAxML()
        self.assertTrue(spectraltree.topos_equal(self.reference_tree, raxml(data)))  

    def test_numpy_data(self):
        jc = spectraltree.Jukes_Cantor()
        observations, taxa_meta = spectraltree.simulate_sequences(
            seq_len=500, 
            tree_model=self.reference_tree, 
            seq_model=jc, 
            mutation_rate=jc.p2t(0.95),
            rng=default_rng(123),
            alphabet="DNA")

        raxml = spectraltree.RAxML()
        self.assertTrue(spectraltree.topos_equal(self.reference_tree, raxml(observations, taxa_meta)))  

    def test_without_class(self):
        data = simulate_discrete_chars(
            seq_len=500, 
            tree_model=self.reference_tree, 
            seq_model=Jc69(), 
            mutation_rate=spectraltree.Jukes_Cantor().p2t(0.95),
            rng=default_rng(123))

        spectraltree_path = os.path.dirname(spectraltree.__file__)
        raxml_path = os.path.join(spectraltree_path, "libs", "raxmlHPC_bin")
        if platform.system() == 'Windows':
            # Windows version:
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(raxml_path,r'\raxmlHPC-SSE3.exe'))
        elif platform.system() == 'Darwin':
            #MacOS version:
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(raxml_path,'raxmlHPC-macOS'))
        elif platform.system() == 'Linux':
            #Linux version
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(raxml_path,'raxmlHPC-SSE3-linux'))

        tree = rx.estimate_tree(char_matrix=data, raxml_args=["-T 2"])
        RF = symmetric_difference(self.reference_tree, tree)
        self.assertEqual(RF, 0)

if __name__ == "__main__":
    unittest.main()