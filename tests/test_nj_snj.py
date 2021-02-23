import unittest

from time import time as _t

from spectraltree import utils, generation, reconstruct_tree

class TestNJ(unittest.TestCase):
    def setUp(self):
        self.num_taxa = 32
        self.N = 5000
        self.reference_tree = utils.lopsided_tree(self.num_taxa)
        self.mutation_rate = [0.1]

    def test_jukes_cantor(self):
        jc = generation.Jukes_Cantor()
        observationsJC, metaJC = generation.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=jc, mutation_rate=self.mutation_rate, alphabet="DNA")    

        t0 = _t()
        nj = reconstruct_tree.NeighborJoining(reconstruct_tree.JC_similarity_matrix)   
        tree_rec = nj(observationsJC, metaJC)
        RF,F1 = reconstruct_tree.compare_trees(tree_rec, self.reference_tree)

        print("###################")
        print("NJ - Jukes_Cantor:")
        print("time:", _t() - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)

    def test_hky(self):
        hky_N = 20_000  # occassionally fails if lower

        hky = generation.HKY(kappa = 1.5)
        observationsHKY, metaHKY = generation.simulate_sequences(hky_N, tree_model=self.reference_tree, seq_model=hky, mutation_rate=self.mutation_rate, alphabet="DNA")

        t0 = _t()
        nj = reconstruct_tree.NeighborJoining(reconstruct_tree.HKY_similarity_matrix)   
        tree_rec = nj(observationsHKY, metaHKY)
        RF,F1 = reconstruct_tree.compare_trees(tree_rec, self.reference_tree)
        print("###################")
        print("NJ - HKY:")
        print("time:", _t() - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)

class TestSNJ(unittest.TestCase):
    def setUp(self):
        self.num_taxa = 32
        self.N = 5000
        self.reference_tree = utils.lopsided_tree(self.num_taxa)
        self.mutation_rate = [0.1]

    def test_jukes_cantor(self):
        jc = generation.Jukes_Cantor()
        observationsJC, metaJC = generation.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=jc, mutation_rate=self.mutation_rate, alphabet="DNA")    

        t0 = _t()
        snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.JC_similarity_matrix)   
        tree_rec = snj(observationsJC, metaJC)
        RF,F1 = reconstruct_tree.compare_trees(tree_rec, self.reference_tree)
        print("###################")
        print("SNJ - Jukes_Cantor:")
        print("time:", _t() - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)

    def test_hky(self):
        hky = generation.HKY(kappa = 2)
        observationsHKY, metaHKY = generation.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=hky, mutation_rate=self.mutation_rate, alphabet="DNA")

        t0 = _t()
        snj = reconstruct_tree.SpectralNeighborJoining(reconstruct_tree.HKY_similarity_matrix)   
        tree_rec = snj(observationsHKY, metaHKY)
        RF,F1 = reconstruct_tree.compare_trees(tree_rec, self.reference_tree)
        print("###################")
        print("SNJ - HKY:")
        print("time:", _t() - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)