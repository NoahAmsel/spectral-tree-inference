import numpy as np
import unittest
import time

from spectraltree import utils, generation, reconstruct_tree, str

class TestSpectralTreeReconstruction(unittest.TestCase):
    def setUp(self):
        self.N = 500
        self.num_taxa = 128
        self.threshold = 16
        self.mutation_rate = 0.05
        # mutation_rate = [jc.p2t(0.95)]

        self.rng = np.random.default_rng(12345)

        # self.reference_tree = utils.unrooted_birth_death_tree(self.num_taxa, birth_rate=1)
        # self.reference_tree = utils.lopsided_tree(self.num_taxa)
        self.reference_tree = utils.balanced_binary(self.num_taxa)
        # self.reference_tree = utils.unrooted_pure_kingman_tree(self.num_taxa)

    def test_jc(self):
        jc = generation.Jukes_Cantor()

        t0 = time.time()
        observations,meta = generation.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=jc, mutation_rate=self.mutation_rate, rng=self.rng, alphabet = 'DNA')
        print("gen time: ", time.time() - t0)
        spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.RAxML,reconstruct_tree.JC_similarity_matrix)   
        #spectral_method = reconstruct_tree.SpectralTreeReconstruction(reconstruct_tree.NeighborJoining,reconstruct_tree.JC_similarity_matrix)   

        t0 = time.time()
        tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, 
            reconstruct_tree.JC_similarity_matrix,
            taxa_metadata= meta, 
            threshhold = self.threshold,
            min_split = 5,
            merge_method = "least_square", 
            verbose=False)
        t = time.time() - t0


        RF,F1 = reconstruct_tree.compare_trees(tree_rec, self.reference_tree)
        print("Spectral: ")
        print("time = ", t)

        print("RF = ",RF)
        print("F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)

# There are currently two implementations of Spectral Tree Reconstruction -- this tests the second one:
class TestSTR(unittest.TestCase):
    def setUp(self):
        self.N = 500
        self.num_taxa = 256
        self.threshold = 32
        self.mutation_rate = 0.05
        # mutation_rate = [jc.p2t(0.95)]

        self.rng = np.random.default_rng(12345)

        # self.reference_tree = utils.unrooted_birth_death_tree(self.num_taxa, birth_rate=1)
        # self.reference_tree = utils.lopsided_tree(self.num_taxa)
        self.reference_tree = utils.balanced_binary(self.num_taxa)
        # self.reference_tree = utils.unrooted_pure_kingman_tree(self.num_taxa)

    def test_hky(self):
        hky = generation.HKY(kappa = 2)

        t0 = time.time()
        observations,meta = generation.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=hky, mutation_rate=self.mutation_rate, rng=self.rng, alphabet = 'DNA')
        print("gen time: ", time.time() - t0)
        spectral_method = str.STR(reconstruct_tree.RAxML, reconstruct_tree.HKY_similarity_matrix, threshold = self.threshold, merge_method="least_square", num_gaps = 1, min_split = 5, verbose=False)   

        t0 = time.time()
        tree_rec = spectral_method(observations,  taxa_metadata= meta)
        t = time.time() - t0


        RF,F1 = reconstruct_tree.compare_trees(tree_rec, self.reference_tree)
        print("Spectral: ")
        print("time = ", t)

        print("RF = ",RF)
        print("F1% = ",F1)
        print("")

        self.assertLessEqual(RF, 2)