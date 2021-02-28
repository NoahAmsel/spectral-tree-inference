import unittest
import numpy as np

from time import time as _t

from spectraltree import utils, generation, reconstruct_tree, forrest_reconstruction

class TestForrest(unittest.TestCase):
    def setUp(self):
        self.N = 500
        self.num_taxa = 8
        self.mutation_rate = [0.1]
        self.reference_tree = utils.unrooted_birth_death_tree(self.num_taxa, birth_rate=0.5)
        #reference_tree = utils.lopsided_tree(num_taxa)
        #reference_tree = utils.balanced_binary(num_taxa)
        self.rng = np.random.default_rng(1234)

    def test_jukes_cantor(self):
        jc = generation.Jukes_Cantor()
        observationsJC, metaJC = generation.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=jc, mutation_rate=self.mutation_rate, rng=self.rng, alphabet="DNA")

        t0 = _t()
        forrest = forrest_reconstruction.Forrest()
        tree_rec = forrest(observationsJC, metaJC)
        t1 = _t()
        RF,F1 = reconstruct_tree.compare_trees(tree_rec, self.reference_tree)
        print("###################")
        print("Forrest - Jukes_Cantor:")
        print("time:", t1 - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)

    def test_hky(self):
        hky = generation.HKY(kappa = 2)
        observationsHKY, metaHKY = generation.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=hky, mutation_rate=self.mutation_rate, rng=self.rng, alphabet="DNA")

        t0 = _t()
        forrest = forrest_reconstruction.Forrest()
        tree_rec = forrest(observationsHKY, metaHKY)
        t1 = _t()
        RF,F1 = reconstruct_tree.compare_trees(tree_rec, self.reference_tree)
        print("###################")
        print("Forrest - HKY:")
        print("time:", t1 - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)