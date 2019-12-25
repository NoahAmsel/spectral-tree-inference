# spectral-tree-inference

### Building an ultrametric tree


```
balanced_binary(num_taxa, namespace=None, edge_length=1.)
lopsided_tree(num_taxa, namespace=None, edge_length=1.)
```

Start with a bunch of leaves. At each step draw `t` from an exponential
distribution with mean 1./(remaining nodes choose 2), extend all the branch lengths by `t`, and merge two of them that are chosen at random. In `mean_kingman` the branch length is exactly the mean of the distribution (which gets smaller as you move up the tree).
```
treesim.pure_kingman_tree(taxon_namespace=taxa)
treesim.mean_kingman_tree(taxon_namespace=taxa)
```

Birth death.
```
treesim.birth_death_tree(birth_rate=1., death_rate=0., num_total_tips=len(taxa), taxon_namespace=taxa)
```

### Building a non-ultrametric tree


```
tree = treesim.birth_death_tree(birth_rate=1., death_rate=2., is_retain_extinct_tips=True, num_total_tips=len(taxa), taxon_namespace=taxa)
```

tree.minmax_leaf_distance_from_root()
