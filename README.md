# spectral-tree-inference

Code for constructing phylogenetic trees from DNA sequence data. The file `reconstruct_tree.py` provides an alternative to the popular Neighbor Joining method of Saitou and Nei. This new method uses a criterion based on the second singular value of a specially constructed matrix that measures similarity between each pair of leaves in the tree. Several functions in that file construct such a matrix from a list of DNA sequences from a set of species. In general, the function `paralinear_distance` should be used for DNA data unless some stronger assumptions can be made. The function `estimate_tree_topology(distance_matrix` takes such a matrix as input and outputs a tree describing the relationships between these species. 
 
 The file `experiments.py` provides functions for comparing reconstruction methods on simulation data generated according to some evolutionary model.The file `generation.py` provides fast implementations of some such models built on the Dendropy library.

This repo is in the process of being refactored! Check back soon for updates and a link to our paper.

Joint work with Ariel Jaffe and Boaz Nadler.
