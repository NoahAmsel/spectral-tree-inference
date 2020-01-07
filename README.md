# spectral-tree-inference
=======
Code for constructing phylogenetic trees from DNA sequence data. The file `reconstruct_tree.py` provides an alternative to the popular Neighbor Joining method of Saitou and Nei. This new method uses a criterion based on the second singular value of a specially constructed matrix that measures similarity between each pair of leaves in the tree. Several functions in that file construct such a matrix from a list of DNA sequences from a set of species. In general, the function `paralinear_distance` should be used for DNA data unless some stronger assumptions can be made. The function `estimate_tree_topology(distance_matrix)` takes such a matrix as input and outputs a tree describing the relationships between these species.

 The file `experiments.py` provides functions for comparing reconstruction methods on simulation data generated according to some evolutionary model. The file `generation.py` provides fast implementations of some such models built on the Dendropy library.

This repo is in the process of being refactored! Check back soon for updates and a link to our paper.

Joint work with Ariel Jaffe and Boaz Nadler.

### Building an ultrametric tree (molecular clock)

```
balanced_binary(num_taxa, namespace=None, edge_length=1.)
lopsided_tree(num_taxa, namespace=None, edge_length=1.)
```

Start with a bunch of leaves. At each step draw `t` from an exponential
distribution with mean 1./(remaining nodes choose 2), extend all the branch lengths by `t`, and merge two of them that are chosen at random. In `mean_kingman` the branch length is exactly the mean of the distribution (which gets smaller as you move up the tree).
```
pure_kingman_tree(taxon_namespace=taxa)
mean_kingman_tree(taxon_namespace=taxa)
```

Birth death.
```
t = birth_death_tree(birth_rate=1., death_rate=0., num_total_tips=len(taxa), taxon_namespace=taxa)
t.collapse_basal_bifurcation()
```

### Building a non-ultrametric tree


```
t = birth_death_tree(birth_rate=1., death_rate=2., is_retain_extinct_tips=True, num_total_tips=len(taxa), taxon_namespace=taxa)
t.collapse_basal_bifurcation()
```

tree.minmax_leaf_distance_from_root()


### Generating sequence data
```
tree = balanced_binary(2**5, edge_length=0.1)
jc = Jukes_Cantor()
observations = jc.generate_sequences(tree, seq_len=10_000)
```

### Using seqgen
This is actually slower than the simulation code I made, even when you batch them.

## reconstructing a tree
```
distance_matrix = paralinear_distance(observations)
recovered_tree = estimate_tree_topology(distance_matrix, namespace=tree.taxon_namespace)
print(recovered_tree.as_ascii_plot())
dendropy.calculate.treecompare.false_positives_and_negatives(recovered_tree, tree, is_bipartitions_updated=False)
```


## Basic Dendropy functionality
```
```

```
s1 = "(A,(B,C));"
tree1 = dendropy.Tree.get(
        data=s1,
        schema="newick")
```

Printing
```
tree1.print_plot(width=80, plot_metric="length")
```

```
tree1.as_ascii_plot()
tree1.as_string("newick")
tree1.as_python_source()
```

Checking if a clan is in a tree
```
taxon_namespace.taxa_bipartition()
```
