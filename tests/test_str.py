import time
import unittest

import numpy as np

import spectraltree

class TestSTDR(unittest.TestCase):
    def setUp(self):
        self.N = 500
        self.threshold = 16
        self.num_taxa = 64
        # self.reference_tree = spectraltree.unrooted_birth_death_tree(self.num_taxa, birth_rate=1)
        self.reference_tree = spectraltree.lopsided_tree(self.num_taxa)
        #self.reference_tree = spectraltree.balanced_binary(64)
        # self.reference_tree = spectraltree.unrooted_pure_kingman_tree(self.num_taxa)

    def test_jc(self):
        jc = spectraltree.Jukes_Cantor()
        observations,meta = spectraltree.simulate_sequences(
            seq_len=self.N,
            tree_model=self.reference_tree,
            seq_model=jc,
            mutation_rate=0.05,
            rng=np.random.default_rng(12345),
            alphabet='DNA')

        spectral_method = spectraltree.STDR(spectraltree.RAxML,spectraltree.JC_similarity_matrix,groudtruth_tree=self.reference_tree)   
        #spectral_method = spectraltree.STDR(spectraltree.NeighborJoining,spectraltree.JC_similarity_matrix)   

        tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, 
            spectraltree.JC_similarity_matrix,
            taxa_metadata= meta, 
            threshold = self.threshold,
            min_split = 5,
            merge_method = "least_square", 
            verbose=True)

        self.assertTrue(spectraltree.topos_equal(self.reference_tree, tree_rec))  

    def test_angle_least_square(self):
        # copied from test_deep_spectral_tree_reonstruction
        jc = spectraltree.Jukes_Cantor()
        mutation_rate = jc.p2t(0.95)
        num_itr = 2 #0
        #reference_tree = spectraltree.balanced_binary(num_taxa)
        # reference_tree = spectraltree.unrooted_birth_death_tree(num_taxa, birth_rate=1)
        # for x in reference_tree.preorder_edge_iter():
        #     x.length = 1
        merging_method_list = ['least_square','angle']
        RF = {'least_square': [], 'angle': []}
        F1 = {'least_square': [], 'angle': []}
        for merge_method in merging_method_list:
            for i in range(num_itr):

                observations, taxa_meta = spectraltree.simulate_sequences(seq_len=self.N, tree_model=self.reference_tree , seq_model=jc, mutation_rate=mutation_rate, rng=np.random.default_rng(789))
                spectral_method = spectraltree.STDR(spectraltree.NeighborJoining,spectraltree.JC_similarity_matrix,groudtruth_tree=self.reference_tree)   
                tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, spectraltree.JC_similarity_matrix, 
                    taxa_metadata = taxa_meta, threshold = self.threshold,merge_method = merge_method, verbose=True)
                RF_i,F1_i = spectraltree.compare_trees(tree_rec, self.reference_tree )
                RF[merge_method].append(RF_i)
                F1[merge_method].append(F1_i)

        self.assertEqual(np.mean(RF['angle']), 0)
        self.assertEqual(np.mean(RF['least_square']), 0)

    def test_threshold_partition(self):
        jc = spectraltree.Jukes_Cantor()
        observations, taxa_meta = spectraltree.simulate_sequences(
            seq_len=self.N,
            tree_model=self.reference_tree,
            seq_model=jc,
            mutation_rate=jc.p2t(0.95),
            rng=np.random.default_rng(321))

        spectral_method = spectraltree.STDR(spectraltree.NeighborJoining,spectraltree.JC_similarity_matrix, groudtruth_tree=self.reference_tree)   

        tree_rec = spectral_method.deep_spectral_tree_reconstruction(observations, spectraltree.JC_similarity_matrix, taxa_metadata = taxa_meta, num_gaps = 4,threshhold = 35, verbose=True)
        RF, _ = spectraltree.compare_trees(tree_rec, self.reference_tree)
        self.assertEqual(RF, 0)

        tree_rec_b = spectral_method.deep_spectral_tree_reconstruction(observations, spectraltree.JC_similarity_matrix, taxa_metadata = taxa_meta, num_gaps = 1,threshhold = 35,  verbose=True)
        RF_b, _ = spectraltree.compare_trees(tree_rec_b, self.reference_tree)
        self.assertEqual(RF_b, 0)

# There are currently two implementations of Spectral Tree Reconstruction -- this tests the second one:
class xTestSTR(unittest.TestCase):
    def setUp(self):
        # self.reference_tree = spectraltree.unrooted_birth_death_tree(self.num_taxa, birth_rate=1)
        # self.reference_tree = spectraltree.lopsided_tree(self.num_taxa)
        self.reference_tree = spectraltree.balanced_binary(128)
        # self.reference_tree = spectraltree.unrooted_pure_kingman_tree(self.num_taxa)

        self.observations_hky, self.meta_hky = spectraltree.simulate_sequences(
            seq_len=300,
            tree_model=self.reference_tree,
            seq_model=spectraltree.HKY(kappa=2),
            mutation_rate=0.05,
            rng=np.random.default_rng(234),
            alphabet='DNA')

    def test_hky(self):
        spectral_method = spectraltree.STR(spectraltree.RAxML, spectraltree.HKY_similarity_matrix(self.meta_hky), threshold=32, merge_method="least_square", num_gaps=1, min_split=5, verbose=False)   
        self.assertTrue(spectraltree.topos_equal(self.reference_tree, spectral_method(self.observations_hky, self.meta_hky)))

if __name__ == "__main__":
    unittest.main()