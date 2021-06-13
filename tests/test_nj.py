import unittest

from numpy.random import default_rng

import spectraltree

class TestNJ(unittest.TestCase):
    def setUp(self):
        self.reference_tree = spectraltree.lopsided_tree(32)

    def test_jukes_cantor_similarity(self):
        observationsJC, metaJC = spectraltree.simulate_sequences(
            seq_len=400,
            tree_model=self.reference_tree,
            seq_model=spectraltree.Jukes_Cantor(),
            mutation_rate=0.1,
            rng=default_rng(345),
            alphabet="DNA")

        nj_jc = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)
        self.assertTrue(spectraltree.topos_equal(self.reference_tree, nj_jc(observationsJC, metaJC)))  

    def test_hky_similarity(self):
        observationsHKY, metaHKY = spectraltree.simulate_sequences(
            seq_len=2_000,
            tree_model=self.reference_tree, 
            seq_model=spectraltree.HKY(kappa=1.5), 
            mutation_rate=0.1,
            rng=default_rng(543),
            alphabet="DNA")

        nj_hky = spectraltree.NeighborJoining(spectraltree.HKY_similarity_matrix(metaHKY))
        self.assertTrue(spectraltree.topos_equal(self.reference_tree, nj_hky(observationsHKY, metaHKY)))

if __name__ == "__main__":
    unittest.main()