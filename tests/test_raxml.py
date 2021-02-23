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
import time
import platform
import unittest

from dendropy.interop import raxml
from dendropy.model.discrete import simulate_discrete_chars, Jc69
from dendropy.calculate.treecompare import symmetric_difference

import spectraltree
from spectraltree import utils, generation, reconstruct_tree

class TestRaxml(unittest.TestCase):
    def setUp(self):
        self.num_taxa = 32
        self.N = 1000
        self.reference_tree = utils.balanced_binary(self.num_taxa)

    def test_dendropy_data(self):
        print("test dendropy data")
        time_s = time.time()
        data = simulate_discrete_chars(self.N, self.reference_tree, Jc69(), mutation_rate=generation.Jukes_Cantor().p2t(0.95), )
        print("")
        print("Time for data generation", time.time()-time_s)
        time_s = time.time()
        raxml = reconstruct_tree.RAxML()
        tree = raxml(data)
        runtime = time.time()-time_s

        print("Data in DNAcharacterMatrix:")
        print("symmetric_difference: ",symmetric_difference(self.reference_tree, tree))
        RF,F1 = reconstruct_tree.compare_trees(self.reference_tree, tree)
        print("raxml: ")
        print("RF = ",RF)
        print("F1% = ",F1)
        print("runtime = ",runtime)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)

    def test_numpy_data(self):
        jc = generation.Jukes_Cantor()
        time_s = time.time()
        mutation_rate = [jc.p2t(0.95)]

        print("test numpy data")
        observations, taxa_meta = generation.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")
        print("")
        print("Time for data generation", time.time()-time_s)
        time_s = time.time()
        raxml = reconstruct_tree.RAxML()
        tree = raxml(observations, taxa_meta)
        runtime = time.time()-time_s
        print("Data in numpy array:")
        print("symmetric_difference: ",symmetric_difference(self.reference_tree, tree))
        RF,F1 = reconstruct_tree.compare_trees(self.reference_tree, tree)
        print("raxml: ")
        print("RF = ",RF)
        print("F1% = ",F1)
        print("runtime = ",runtime)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)

    def test_without_class(self):
        data = simulate_discrete_chars(1000, self.reference_tree, Jc69(), mutation_rate=generation.Jukes_Cantor().p2t(0.95), )
        spectraltree_path = os.path.dirname(spectraltree.__file__)
        if platform.system() == 'Windows':
            # Windows version:
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(spectraltree_path,r'\raxmlHPC-SSE3.exe'))
        elif platform.system() == 'Darwin':
            #MacOS version:
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(spectraltree_path,'raxmlHPC-macOS'))
        elif platform.system() == 'Linux':
            #Linux version
            rx = raxml.RaxmlRunner(raxml_path = os.path.join(spectraltree_path,'raxmlHPC-SSE3-linux'))

        tree = rx.estimate_tree(char_matrix=data, raxml_args=["-T 2"])

        print("symmetric_difference: ",symmetric_difference(reference_tree, tree))
        RF,F1 = reconstruct_tree.compare_trees(reference_tree, tree)
        print("raxml: ")
        print("RF = ",RF)
        print("F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)