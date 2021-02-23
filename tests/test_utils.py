import unittest
import numpy as np
import dendropy

from spectraltree import utils


class TestTaxaMetadata(unittest.TestCase):
    def setUp(self):
        self.namespace1 = dendropy.TaxonNamespace(["dog", "cat", "snake", "fish", "tree"])
        self.taxa1 = utils.TaxaMetadata(self.namespace1, ["fish", "snake", "cat", "dog"], "DNA")
        self.dog = self.namespace1.get_taxon("dog")
        self.snake = self.namespace1.get_taxon("snake")

    # TODO: break this out
    def test_basic(self):
        self.assertListEqual(list(self.taxa1), [self.namespace1[3], self.namespace1[2], self.namespace1[1], self.namespace1[0]])
        self.assertEqual(self.taxa1['snake'], 1)
        self.assertTrue(all(self.taxa1.index2taxa([1,3]) == np.array([self.snake, self.dog])))
        self.assertEqual(self.taxa1.alphabet, dendropy.DNA_STATE_ALPHABET)

    def test_2mask(self):
        self.assertTrue(all(self.taxa1.taxon2mask('snake') == np.array([False, True, False, False])))
        self.assertTrue(all(self.taxa1.taxa2mask(['snake', 'dog']) == np.array([False,  True, False,  True])))

    def test_leaf(self):
        leaf = self.taxa1.leaf('dog', edge_length=3.0)
        self.assertEqual(leaf.taxon, self.dog)
        self.assertEqual(leaf.edge_length, 3.0)
        self.assertTrue(all(leaf.mask == np.array([False, False, False, True])))

    def test_2bipartition(self):
        self.assertEqual(str(self.taxa1.taxa2bipartition(["dog", "cat"])), "11100")
        self.assertEqual(str(self.taxa1.mask2bipartition([False, False, True, True])), "11100")

    def test_str(self):
        self.assertEqual(str(self.taxa1), str(['fish', 'snake', 'cat', 'dog']))

    def test_default(self):
        self.assertListEqual([taxon.label for taxon in utils.TaxaMetadata.default(4)], ["T1", "T2", "T3", "T4"])

class TestConversionFunctions(unittest.TestCase):
    def setUp(self):
        self.namespace1 = dendropy.TaxonNamespace(["dog", "cat", "snake", "fish", "tree"])
        self.taxa1 = utils.TaxaMetadata(self.namespace1, ["fish", "snake", "cat", "dog"], "DNA")
        self.dog = self.namespace1.get_taxon("dog")
        self.snake = self.namespace1.get_taxon("snake")

        self.array1 = np.array([
            [3, 1, 1, 0, 0],
            [3, 2, 1, 0, 0],
            [3, 2, 4, 0, 0],
            [2, 2, 0, 0, 0],
        ])

        self.tax2seq = {
            "fish": self.array1[0,:],
            "snake": self.array1[1,:],
            "cat": self.array1[2,:],
            "dog": self.array1[3,:],
        }
  
        d = {
            "fish" : "TCCAA",
            "snake" : "TGCAA",
            "cat" : "TG-AA",
            "dog": "GGAAA",
        }
        self.dna_charmatrix = dendropy.DnaCharacterMatrix.from_dict(d, taxon_namespace=self.namespace1)

        tree = utils.lopsided_tree(4, self.taxa1)
        self.dm = tree.phylogenetic_distance_matrix()
        
    def test_charmatrix2array(self):
        array2, taxa2 = utils.charmatrix2array(self.dna_charmatrix)
        self.assertTrue(self.taxa1.equals_unordered(taxa2))
        self.assertEqual(taxa2.alphabet, dendropy.DNA_STATE_ALPHABET)
        for tax in taxa2:
            self.assertTrue((array2[taxa2[tax], :] == self.tax2seq[tax.label]).all())

    def test_array2charmatrix(self):
        charmatrix2 = utils.array2charmatrix(self.array1, self.taxa1)
        self.assertEqual(charmatrix2.as_string("nexus"), self.dna_charmatrix.as_string("nexus"))
        self.assertTrue(isinstance(charmatrix2, dendropy.DnaCharacterMatrix))

    def test_array2distance_matrix(self):
        distance_mat = np.array([
                [0., 3., 4., 4.],
                [3., 0., 3., 3.],
                [4., 3., 0., 2.],
                [4., 3., 2., 0.]
        ])

        dm2 = utils.array2distance_matrix(distance_mat, self.taxa1)
        for taxon1 in self.taxa1:
            for taxon2 in self.taxa1:
                self.assertEqual(dm2.distance(taxon1, taxon2), self.dm.distance(taxon1, taxon2))

    def test_distance_matrix2array(self):    
        distance_mat, matrix_taxa = utils.distance_matrix2array(self.dm)
        # in this test, the taxa meta of a distance matrix needs no alphabet
        taxa_temp = utils.TaxaMetadata(self.namespace1, ["fish", "snake", "cat", "dog"], None)
        self.assertTrue(matrix_taxa.equals_unordered(taxa_temp))
        for taxon1 in self.taxa1:
            for taxon2 in self.taxa1:
                self.assertEqual(distance_mat[matrix_taxa[taxon1], matrix_taxa[taxon2]], self.dm.distance(taxon1, taxon2))

if __name__ == "__main__":
    unittest.main()