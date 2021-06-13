from time import time as _t
import unittest

from numpy.random import default_rng

import spectraltree

class TestSNJ(unittest.TestCase):
    def setUp(self):
        self.reference_tree = spectraltree.lopsided_tree(32)

    def test_jukes_cantor(self):
        jc = spectraltree.Jukes_Cantor()
        observationsJC, metaJC = spectraltree.simulate_sequences(
            seq_len=5000,
            tree_model=self.reference_tree, 
            seq_model=jc, 
            mutation_rate=0.1,
            rng=default_rng(345),
            alphabet="DNA")    

        snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)
        self.assertTrue(spectraltree.topos_equal(self.reference_tree, snj(observationsJC, metaJC)))

    def test_hky(self):
        hky = spectraltree.HKY(kappa = 2)
        observationsHKY, metaHKY = spectraltree.simulate_sequences(
            seq_len=5000,
            tree_model=self.reference_tree,
            seq_model=hky,
            mutation_rate=0.1,
            rng=default_rng(345),
            alphabet="DNA")

        snj_hky = spectraltree.SpectralNeighborJoining(spectraltree.HKY_similarity_matrix(metaHKY))
        self.assertTrue(spectraltree.topos_equal(self.reference_tree, snj_hky(observationsHKY, metaHKY)))

if __name__ == "__main__":
    unittest.main()