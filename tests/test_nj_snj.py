from time import time as _t
import unittest

import numpy as np

import spectraltree

class TestNJ(unittest.TestCase):
    def setUp(self):
        self.num_taxa = 32
        self.N = 5000
        self.reference_tree = spectraltree.lopsided_tree(self.num_taxa)
        self.mutation_rate = [0.1]
        self.rng = np.random.default_rng(12345)

    def test_jukes_cantor(self):
        jc = spectraltree.Jukes_Cantor()
        observationsJC, metaJC = spectraltree.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=jc, mutation_rate=self.mutation_rate, alphabet="DNA")    

        t0 = _t()
        nj = spectraltree.NeighborJoining(spectraltree.JC_similarity_matrix)   
        tree_rec = nj(observationsJC, metaJC)
        RF,F1 = spectraltree.compare_trees(tree_rec, self.reference_tree)

        print("###################")
        print("NJ - Jukes_Cantor:")
        print("time:", _t() - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)

    def test_hky(self):
        hky_N = 20_000  # occassionally fails if lower

        hky = spectraltree.HKY(kappa = 1.5)
        observationsHKY, metaHKY = spectraltree.simulate_sequences(hky_N, tree_model=self.reference_tree, seq_model=hky, mutation_rate=self.mutation_rate, alphabet="DNA", rng=self.rng)

        t0 = _t()
        nj = spectraltree.NeighborJoining(spectraltree.HKY_similarity_matrix)   
        tree_rec = nj(observationsHKY, metaHKY)
        RF,F1 = spectraltree.compare_trees(tree_rec, self.reference_tree)
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
        self.reference_tree = spectraltree.lopsided_tree(self.num_taxa)
        self.mutation_rate = [0.1]

    def test_jukes_cantor(self):
        jc = spectraltree.Jukes_Cantor()
        observationsJC, metaJC = spectraltree.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=jc, mutation_rate=self.mutation_rate, alphabet="DNA")    

        t0 = _t()
        snj = spectraltree.SpectralNeighborJoining(spectraltree.JC_similarity_matrix)   
        tree_rec = snj(observationsJC, metaJC)
        RF,F1 = spectraltree.compare_trees(tree_rec, self.reference_tree)
        print("###################")
        print("SNJ - Jukes_Cantor:")
        print("time:", _t() - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)

    def test_hky(self):
        hky = spectraltree.HKY(kappa = 2)
        observationsHKY, metaHKY = spectraltree.simulate_sequences(self.N, tree_model=self.reference_tree, seq_model=hky, mutation_rate=self.mutation_rate, alphabet="DNA")

        t0 = _t()
        snj = spectraltree.SpectralNeighborJoining(spectraltree.HKY_similarity_matrix)   
        tree_rec = snj(observationsHKY, metaHKY)
        RF,F1 = spectraltree.compare_trees(tree_rec, self.reference_tree)
        print("###################")
        print("SNJ - HKY:")
        print("time:", _t() - t0)
        print("RF = ",RF, "    F1% = ",F1)
        print("")

        self.assertEqual(RF, 0)
        self.assertEqual(F1, 100)