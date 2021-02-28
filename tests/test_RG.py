import unittest
import numpy as np

import time

from spectraltree import utils, generation, reconstruct_tree, choi_reconstruction

class TestRG(unittest.TestCase):
    def test_jukes_cantor(self):
        N = 100_000
        num_taxa = 8
        jc = generation.Jukes_Cantor(num_classes=2)
        mutation_rate = [jc.p2t(0.95)]
        rng = np.random.default_rng(12345)
        # reference_tree = utils.unrooted_birth_death_tree(num_taxa, birth_rate=1)
        # reference_tree = utils.lopsided_tree(num_taxa)
        reference_tree = utils.balanced_binary(num_taxa)

        t0 = time.time()
        observations,meta = generation.simulate_sequences(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate,rng=rng, alphabet="Binary")
        print("gen time: ", time.time() - t0)
        rg = choi_reconstruction.RG()

        t0 = time.time()
        tree_rec = rg(observations, taxa_metadata= meta)
        t = time.time() - t0

        RF,F1 = reconstruct_tree.compare_trees(tree_rec, reference_tree)
        print("rg: ")
        print("time = ", t)

        print("RF = ",RF)
        print("F1% = ",F1)
        print("")

        self.assertEqual(RF, 0) # this doesn't work very well