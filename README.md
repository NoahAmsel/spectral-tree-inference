
# New repositories for STDR and SNJ
We created a new repository, with simple examples, for the code of the STDR algorithm (https://arxiv.org/abs/2102.13276). Please visit: https://github.com/aizeny/stdr

We created a new repository, with simple examples, for the code of the SNJ algorithm (https://epubs.siam.org/doi/abs/10.1137/20M1365715 or https://arxiv.org/abs/2002.12547). Please visit: https://github.com/aizeny/snj


# Update of March 8th 2021:
This update cleans up the repository significantly and implements best practices like unit testing, documentation, avoiding assertions,  and packaging and listing dependincies using setuptools. It fixes the problems we were having with imports, and it standardizes the interfaces of many functions and classes (the exception is spectral tree reconstruction -- I put it into a separate file but left its interface alone). These changes, especially the last two, break some of the existing experiment scripts. I have fixed all the experiments in `snj_paper_experiments` already, and the rest should be easy enough to fix as we need them. Use the unit tests in the `tests` folder and the scripts in `snj_paper_experiments` as guides. There are two main steps:
1. Fix the imports. We no longer need to mess with PYTHONPATH or bother importing multiple modules. Just `import spectraltree` and use the functions and class from there directly, like `spectraltree.balanced_binary(..)`. The exception to that is the module `compare_methods`, which should be imported as follows:
```
import spectraltree.compare_methods as compare_methods
```
2. Use updated interfaces. For example, all additional arguments to reconstruction methods have been moved to the constructor. When you call the method, provide only the observations matrix and (optionally) the TaxaMetadata. (Again, `spectral_tree_reconstruction` is an exception for now.) Also, instead of calling the function `spectraltree.HKY_similarity_matrix(observations, metaHKY)` we create an object and use it like a function `spectraltree.HKY_similarity_matrix(observations)(metaHKY)`.

## Installing
To make imports work, you should install the package using pip. cd inside the `spectral-tree-inference` folder and run 
```
pip install -e .
```
Now if you want to import the spectraltree package, for example for an experiment, you can simply run
```
import spectraltree
```
without modifying your path or doing anything ugly. You should only have to run `pip install -e .` one time. If you move the location of `spectraltree` folder, then run `pip uninstall spectral-tree-inference` and install again.

This also installs all the Python dependencies of spectraltree, including oct2py. However it does not install octave itself. The easiest way to do that is to run `conda install -c conda-forge octave`; for more see here <https://blink1073.github.io/oct2py/source/installation.html#gnu-octave-installation>.

Note that if you are working inside the `spectraltree` directory itself -- for example, modifying one of the reconstruction methods -- and you need to access code from another file inside `spectraltree`, use a relative import instead:
```
from . import utils
from .utils import TaxaMetadata
```

## Testing
To run the test suite, cd to `spectral-tree-inference` and run
```
python -m unittest discover
```
To run a specific module:
```
python -m unittest tests.test_snj
```
Some of the third party methods (Forrest and RG) are pretty unreliable and so those tests may not pass. Just make sure they PASS or FAIL instead of throwing an ERROR.

## Documentation
Run
```
python setup.py build_sphinx
```
(You can also go into the `docs` folder and run `make html`). Then open `docs/_build/html/index.html` in a browser.

# TODOs:
- Clean up STR
- Polish this readme and create an example Jupyter notebook
- Why do the Choi methods work so badly? Are we using them wrong?
- Fill in docstrings for utils
- For RAxML, we've been using the default argument "-T 2" which sets the number of threads to 2. But isn't this only for the multithreaded version `raxmlHPC­PTHREADS`? If it isn't, then why are we only using 2 threads?

# spectral-tree-inference
=======
Code for constructing phylogenetic trees from DNA sequence data. The file `reconstruct_tree.py` provides an alternative to the popular Neighbor Joining method of Saitou and Nei. This new method uses a criterion based on the second singular value of a specially constructed matrix that measures similarity between each pair of leaves in the tree. Several functions in that file construct such a matrix from a list of DNA sequences from a set of species. In general, the function `paralinear_distance` should be used for DNA data unless some stronger assumptions can be made. The function `estimate_tree_topology(distance_matrix)` takes such a matrix as input and outputs a tree describing the relationships between these species.

 The file `experiments.py` provides functions for comparing reconstruction methods on simulation data generated according to some evolutionary model. The file `generation.py` provides fast implementations of some such models built on the Dendropy library.

This repo is in the process of being refactored! Check back soon for updates and a link to our paper.

Joint work with Ariel Jaffe and Boaz Nadler.

### Dependencies
numpy
pandas
dendropy
seaborn


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
