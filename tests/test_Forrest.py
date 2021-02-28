import unittest
import numpy as np

from time import time as _t

import spectraltree

class TestForrest(unittest.TestCase):
    def setUp(self):
        self.N = 500
        self.num_taxa = 8
        self.mutation_rate = [0.1]
        self.reference_tree = spectraltree.unrooted_birth_death_tree(self.num_taxa, birth_rate=0.5)
        #reference_tree = spectraltree.lopsided_tree(num_taxa)
        #reference_tree = spectraltree.balanced_binary(num_taxa)
        self.rng = np.random.default_rng(1234)

    def test_jukes_cantor(self):
        jc = spectraltree.Jukes_Cantor()
        observationsJC, metaJC = spectraltree.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=jc, mutation_rate=self.mutation_rate, rng=self.rng, alphabet="DNA")

        t0 = _t()
        forrest = spectraltree.Forrest()
        tree_rec = forrest(observationsJC, metaJC)
        t1 = _t()
        RF,F1 = spectraltree.compare_trees(tree_rec, self.reference_tree)
        print("###################")
        print("Forrest - Jukes_Cantor:")
        print("time:", t1 - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)

    def test_hky(self):
        hky = spectraltree.HKY(kappa = 2)
        observationsHKY, metaHKY = spectraltree.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=hky, mutation_rate=self.mutation_rate, rng=self.rng, alphabet="DNA")

        t0 = _t()
        forrest = spectraltree.Forrest()
        tree_rec = forrest(observationsHKY, metaHKY)
        t1 = _t()
        RF,F1 = spectraltree.compare_trees(tree_rec, self.reference_tree)
        print("###################")
        print("Forrest - HKY:")
        print("time:", t1 - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)