
import unittest
import numpy as np
import dendropy

import character_matrix


class TestTaxaIndexMapping(unittest.TestCase):
    def setUp(self):
        self.namespace1 = dendropy.TaxonNamespace(["dog", "cat", "snake", "fish", "tree"])
        self.taxa1 = character_matrix.TaxaIndexMapping(self.namespace1, ["fish", "snake", "cat", "dog"])
        self.dog = self.namespace1.get_taxon("dog")
        self.snake = self.namespace1.get_taxon("snake")

    # TODO: break this out
    def test_basic(self):
        self.assertListEqual(list(self.taxa1), [self.namespace1[3], self.namespace1[2], self.namespace1[1], self.namespace1[0]])
        self.assertEqual(self.taxa1['snake'], 1)
        self.assertTrue(all(self.taxa1.index2taxa([1,3]) == np.array([self.snake, self.dog])))

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

if __name__ == "__main__":
    unittest.main()