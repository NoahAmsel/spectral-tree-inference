import time
import unittest

import numpy as np

import spectraltree

class TestSpectralTreeReconstruction(unittest.TestCase):
    def setUp(self):
        self.N = 500
        self.num_taxa = 128
        self.threshold = 16
        self.mutation_rate = 0.05
        # mutation_rate = [jc.p2t(0.95)]

        self.rng = np.random.default_rng(12345)

        # self.reference_tree = spectraltree.unrooted_birth_death_tree(self.num_taxa, birth_rate=1)
        # self.reference_tree = spectraltree.lopsided_tree(self.num_taxa)
        self.reference_tree = spectraltree.balanced_binary(self.num_taxa)
        # self.reference_tree = spectraltree.unrooted_pure_kingman_tree(self.num_taxa)

    def test_jc(self):
        jc = spectraltree.Jukes_Cantor()

        t0 = time.time()
        observations,meta = spectraltree.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=jc, mutation_rate=self.mutation_rate, rng=self.rng, alphabet = 'DNA')
        print("gen time: ", time.time() - t0)
        spectral_method = spectraltree.SpectralTreeReconstruction(spectraltree.RAxML,spectraltree.JC_similarity_matrix)   
        #spectral_method = spectraltree.SpectralTreeReconstruction(spectraltree.NeighborJoining,spectraltree.JC_similarity_matrix)   

        t0 = time.time()
        tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, 
            spectraltree.JC_similarity_matrix,
            taxa_metadata= meta, 
            threshhold = self.threshold,
            min_split = 5,
            merge_method = "least_square", 
            verbose=False)
        t = time.time() - t0


        RF,F1 = spectraltree.compare_trees(tree_rec, self.reference_tree)
        print("Spectral: ")
        print("time = ", t)

        print("RF = ",RF)
        print("F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)

    def test_angle_least_square(self):
        # copied from test_deep_spectral_tree_reonstruction
        N = 1000
        num_taxa = 128
        jc = spectraltree.Jukes_Cantor()
        mutation_rate = [jc.p2t(0.95)]
        num_itr = 2 #0
        # reference_tree = spectraltree.unrooted_birth_death_tree(num_taxa, birth_rate=1)
        # for x in reference_tree.preorder_edge_iter():
        #     x.length = 1
        merging_method_list = ['least_square','angle']
        RF = {'least_square': [], 'angle': []}
        F1 = {'least_square': [], 'angle': []}
        for merge_method in merging_method_list:
            for i in range(num_itr):
                #reference_tree = spectraltree.balanced_binary(num_taxa)
                reference_tree = spectraltree.lopsided_tree(num_taxa)
                observations, taxa_meta = spectraltree.simulate_sequences(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate)
                spectral_method = spectraltree.SpectralTreeReconstruction(spectraltree.NeighborJoining,spectraltree.JC_similarity_matrix)   
                tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, spectraltree.JC_similarity_matrix, 
                    taxa_metadata = taxa_meta, threshhold = 16,merge_method = merge_method)
                RF_i,F1_i = spectraltree.compare_trees(tree_rec, reference_tree)
                RF[merge_method].append(RF_i)
                F1[merge_method].append(F1_i)

        print("Angle RF: ",np.mean(RF['angle']))
        #print("Angle RF: ",np.mean(RF['angle']), "Runtime: ", runtime)

        print("LS RF: ",np.mean(RF['least_square']))

        self.assertEqual(np.mean(RF['angle']), 0)
        self.assertEqual(np.mean(RF['least_square']), 0)

    def test_threshold_partition(self):
        # copied from test_threshold_partition
        N = 1000
        num_taxa = 256
        jc = spectraltree.Jukes_Cantor()
        mutation_rate = [jc.p2t(0.95)]

        reference_tree = spectraltree.balanced_binary(num_taxa)
        observations, taxa_meta = spectraltree.simulate_sequences(N, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, rng=self.rng)

        spectral_method = spectraltree.SpectralTreeReconstruction(spectraltree.NeighborJoining,spectraltree.JC_similarity_matrix)   
        tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, spectraltree.JC_similarity_matrix, taxa_metadata = taxa_meta, num_gaps = 4,threshhold = 35)
        tree_rec_b = spectral_method.deep_spectral_tree_reconstruction(observations, spectraltree.JC_similarity_matrix, taxa_metadata = taxa_meta, num_gaps = 1,threshhold = 35)

        RF,F1 = spectraltree.compare_trees(tree_rec, reference_tree)
        RF_b,F1_b = spectraltree.compare_trees(tree_rec_b, reference_tree)
        print("Spectral deep: ")
        print("RF multi gaps= ",RF)
        print("RF standard = ",RF_b)

        self.assertEqual(RF, 0)
        self.assertEqual(RF_b, 0)

# There are currently two implementations of Spectral Tree Reconstruction -- this tests the second one:
class TestSTR(unittest.TestCase):
    def setUp(self):
        self.N = 500
        self.num_taxa = 256
        self.threshold = 32
        self.mutation_rate = 0.05
        # mutation_rate = [jc.p2t(0.95)]

        self.rng = np.random.default_rng(12345)

        # self.reference_tree = spectraltree.unrooted_birth_death_tree(self.num_taxa, birth_rate=1)
        # self.reference_tree = spectraltree.lopsided_tree(self.num_taxa)
        self.reference_tree = spectraltree.balanced_binary(self.num_taxa)
        # self.reference_tree = spectraltree.unrooted_pure_kingman_tree(self.num_taxa)

    def test_hky(self):
        hky = spectraltree.HKY(kappa = 2)

        t0 = time.time()
        observations,meta = spectraltree.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=hky, mutation_rate=self.mutation_rate, rng=self.rng, alphabet = 'DNA')
        print("gen time: ", time.time() - t0)
        spectral_method = spectraltree.STR(spectraltree.RAxML, spectraltree.HKY_similarity_matrix, threshold = self.threshold, merge_method="least_square", num_gaps = 1, min_split = 5, verbose=False)   

        t0 = time.time()
        tree_rec = spectral_method(observations,  taxa_metadata= meta)
        t = time.time() - t0


        RF,F1 = spectraltree.compare_trees(tree_rec, self.reference_tree)
        print("Spectral: ")
        print("time = ", t)

        print("RF = ",RF)
        print("F1% = ",F1)
        print("")

        self.assertLessEqual(RF, 2)