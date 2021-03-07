from random import Random
import unittest

from numpy.random import default_rng

import spectraltree

class TestForrest(unittest.TestCase):
    def setUp(self):
        self.reference_tree = spectraltree.unrooted_birth_death_tree(8, birth_rate=0.5, rng=Random(1234))
        #reference_tree = spectraltree.lopsided_tree(num_taxa)
        #reference_tree = spectraltree.balanced_binary(num_taxa)

    def test_jukes_cantor(self):
        observationsJC, metaJC = spectraltree.simulate_sequences(
            seq_len=500, 
            tree_model=self.reference_tree, 
            seq_model=spectraltree.Jukes_Cantor(), 
            mutation_rate=0.1, 
            rng=default_rng(234),
            alphabet="DNA")

        forrest = spectraltree.Forrest()
        self.assertTrue(spectraltree.topos_equal(self.reference_tree, forrest(observationsJC, metaJC)))  

    def test_hky(self):
        observationsHKY, metaHKY = spectraltree.simulate_sequences(
            seq_len=500, 
            tree_model=self.reference_tree, 
            seq_model=spectraltree.HKY(kappa=2), 
            mutation_rate=0.1, 
            rng=default_rng(234),
            alphabet="DNA")

        forrest = spectraltree.Forrest()
        self.assertTrue(spectraltree.topos_equal(self.reference_tree, forrest(observationsHKY, metaHKY)))