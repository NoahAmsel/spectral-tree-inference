import time
import unittest

from numpy.random import default_rng

import spectraltree

class TestRG(unittest.TestCase):
    def test_jukes_cantor(self):
        # reference_tree = spectraltree.unrooted_birth_death_tree(num_taxa, birth_rate=1, rng=)
        # reference_tree = spectraltree.lopsided_tree(num_taxa)
        reference_tree = spectraltree.balanced_binary(8)

        jc = spectraltree.Jukes_Cantor(num_classes=2)
        observations,meta = spectraltree.simulate_sequences(
            seq_len=10_000,
            tree_model=reference_tree,
            seq_model=jc,
            mutation_rate=jc.p2t(0.98),
            rng=default_rng(678),
            alphabet="Binary")

        rg = spectraltree.RG(spectraltree.JC_distance_matrix)
        recoverd_tree = rg(observations, meta)
        print("(RF distance,F1 score):",spectraltree.compare_trees(reference_tree,recoverd_tree))
        self.assertTrue(spectraltree.topos_equal(reference_tree, recoverd_tree))  # this doesn't work very well


if __name__ == "__main__":
    unittest.main()