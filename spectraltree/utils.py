from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
import scipy.spatial.distance
import numpy as np
import dendropy

def default_namespace(num_taxa, prefix="T"):
    return dendropy.TaxonNamespace([prefix+str(i) for i in range(1, num_taxa+1)])

class TaxaIndexMapping(Mapping):
    def __init__(self, taxon_namespace, taxa_list):
        # TODO: modify init so that it detects repeated values (even if it's specified as a taxon object one time and as a label the other) and throws an error
        self._taxon_namespace = taxon_namespace
        self._taxa_list = []
        self._taxon2index = {}
        for ix, taxon in enumerate(taxa_list):
            taxon = self._convert_label(taxon)
            if taxon in taxon_namespace:
                self._taxon2index[taxon] = ix
                self._taxon2index[taxon.label] = ix
                # append one at a time to make sure labels have been converted to taxa first
                self._taxa_list.append(taxon)
            else:
                # not in namespace, or wrong type
                # TODO: throw a real error
                assert False, "Each taxon must be included in the given taxon namespace."

        self._taxa_list = np.array(self._taxa_list)

    @classmethod
    def whole_namespace(cls, namespace):
        return cls(namespace, list(namespace))

    @classmethod
    def default(cls, length):
        return cls.whole_namespace(default_namespace(length))

    @property
    def taxon_namespace(self):
        return self._taxon_namespace

    def __getitem__(self, taxon):
        return self._taxon2index[taxon]

    def __iter__(self):
        for taxon in self._taxa_list:
            yield taxon

    def __len__(self):
        return len(self._taxa_list)

    def _convert_label(self, taxon_or_label):
        return self.taxon_namespace.get_taxon(taxon_or_label) if self.taxon_namespace.has_taxon_label(taxon_or_label) else taxon_or_label

    def _convert_labels(self, taxa_or_labels):
        return [self._convert_label(t_or_l) for t_or_l in taxa_or_labels]

    def index2taxa(self, indexer):
        return self._taxa_list[indexer]

    def taxon2mask(self, taxon):
        taxon = self._convert_label(taxon)
        mask = np.zeros(len(self), dtype=bool)
        mask[self[taxon]] = True
        return mask

    def taxa2mask(self, taxa):
        taxa = self._convert_labels(taxa)
        mask = np.zeros(len(self), dtype=bool)
        mask[[self[taxon] for taxon in taxa]] = True
        return mask

    def taxa2bipartition(self, taxa):
        taxa = self._convert_labels(taxa)
        return self._taxon_namespace.taxa_bipartition(taxa=taxa)

    def mask2bipartition(self, mask):
        return self.taxa2bipartition(self.index2taxa(mask))

    def leaf(self, taxon, **kwargs):
        taxon = self._convert_label(taxon)
        assert taxon in self, "Must supply taxon in the taxa map to produce a leaf."

        kwargs['taxon'] = taxon
        node = dendropy.Node(**kwargs)
        node.mask = self.taxon2mask(taxon)
        return node

    def all_leaves(self, **kwargs):
        return [self.leaf(taxon, **kwargs) for taxon in self]
        
    def reindex_matrix(self, matrix, old_taxa, axes=[0]):
        assert False
        assert self.equals_unordered(old_taxa), "Old and new taxa maps must have the same set of taxa."
        new_matrix = np.zeros_like(matrix)
        # how do we generalize this for different numbers of axes?

    def __str__(self):
        return str([taxon.label for taxon in self])

    def __eq__(self, other):
        return isinstance(other, TaxaIndexMapping) and (
            self.taxon_namespace == other.taxon_namespace) and (
                all(self._taxa_list == other._taxa_list))

    def equals_unordered(self, other):
        return isinstance(other, TaxaIndexMapping) and (
            self.taxon_namespace == other.taxon_namespace) and (
                set(self._taxa_list) == set(other._taxa_list))

def charmatrix2array(charmatrix):
    #charmatrix[taxon].values()
    alphabet = charmatrix.state_alphabets[0] # TODO: what if there are multiple different state_alphabets?
    taxa = []
    sequences = []
    for taxon in charmatrix:
        taxa.append(taxon)
        sequences.append([state_id.index for state_id in charmatrix[taxon].values()])
    
    return np.array(sequences), TaxaIndexMapping(charmatrix.taxon_namespace, taxa), alphabet

def array2charmatrix(matrix, alphabet=None, taxa_index_map=None):
    if taxa_index_map is None:
        taxa_index_map = TaxaIndexMapping.default(matrix.shape[0])
    else:
        assert len(taxa_index_map) == matrix.shape[0], "Taxon-Index map does not match size of matrix."
    
    # TODO: add support for DNACharacterMatrix and others
    if alphabet is None:
        # input the values in the matrix directly
        alphabet = dendropy.new_standard_state_alphabet(val for val in np.unique(matrix))
        char_matrix = dendropy.StandardCharacterMatrix(default_state_alphabet=alphabet, taxon_namespace=taxa_index_map.taxon_namespace)
        for taxon, ix in taxa_index_map.items():
            char_matrix.new_sequence(taxon, [str(x) for x in matrix[ix, :]])
    else:
        # assume values in the matrix are indices into the alphabet
        char_matrix = dendropy.StandardCharacterMatrix(default_state_alphabet=alphabet, taxon_namespace=taxa_index_map.taxon_namespace)
        for taxon, ix in taxa_index_map.items():
            char_matrix.new_sequence(taxon, [alphabet[v] for v in matrix[ix, :]])

    return char_matrix

def array2distance_matrix(matrix, taxa_index_map=None):
    m, m2 = matrix.shape
    assert m == m2, "Distance matrix must be square"
    if taxa_index_map is None:
        taxa_index_map = TaxaIndexMapping.default(m)
    else:
        assert len(taxa_index_map) == m, "Taxon-Index map does not match size of matrix."

    dict_form = defaultdict(dict)
    for row_taxon in taxa_index_map:
        for column_taxon in taxa_index_map:
            dict_form[row_taxon][column_taxon] = matrix[taxa_index_map.index2taxa(row_taxon), taxa_index_map.index2taxa(column_taxon)]
    dm = dendropy.calculate.phylogeneticdistance.PhylogeneticDistanceMatrix()
    dm.compile_from_dict(dict_form, taxa_index_map.taxon_namespace)
    return dm

def distance_matrix2array(dm):
    """
    This is patristic distance: adding the branch lengths. If we set branches to
    have different transitions, this won't be the paralinear distance
    """
    taxa = TaxaIndexMapping(dm.taxon_namespace, list(dm.taxon_iter()))
    return scipy.spatial.distance.squareform([dm.distance(taxon1, taxon2) for taxon1, taxon2 in combinations(taxa,2)]), taxa

def tree2distance_matrix(tree):
    return distance_matrix2array(tree.phylogenetic_distance_matrix())

def merge_children(children, **kwargs):
    node = dendropy.Node(**kwargs)
    for child in children:
        node.add_child(child)
    if all(hasattr(child,'taxa_set') for child in children):
        node.taxa_set = np.logical_or.reduce(tuple(child.taxa_set for child in children))
    return node

def set_edge_lengths(tree, value=None, fun=None, uniform_range=None):
    for e in tree.edges():
        if value is not None:
            assert fun is None and uniform_range is None
            e.length = value
        elif fun is not None:
            assert uniform_range is None
            e.length = fun(e.length)
        else:
            assert uniform_range is not None
            e.length = np.random.uniform(*uniform_range)

##########################################################
##               Tree Generation
##########################################################

# Including these functions here for convenience
from dendropy.simulate.treesim import birth_death_tree, pure_kingman_tree, mean_kingman_tree 

def balanced_binary(num_taxa=None, taxa=None, edge_length=1.):
    if taxa is None:
        if num_taxa:
            taxa = TaxaIndexMapping.default(num_taxa)
        else:
            assert False, "Must provide either the number of leaves or a TaxaIndexMapping"

    if num_taxa:
        assert num_taxa == len(taxa), "The number of leaves must match the number of taxa given."
    else:
        num_taxa = len(taxa)

    assert num_taxa == 2**int(np.log2(num_taxa)), "The number of leaves in a balanced binary tree must be a power of 2."

    nodes = taxa.all_leaves(edge_length=edge_length)
    while len(nodes) > 1:
        nodes = [merge_children(nodes[2*i : 2*i+2], edge_length=edge_length) for i in range(len(nodes)//2)]

    return dendropy.Tree(taxon_namespace=taxa.taxon_namespace, seed_node=nodes[0], is_rooted=False)

def lopsided_tree(num_taxa, taxa=None, edge_length=1.):
    """One node splits off at each step"""
    if taxa is None:
        if num_taxa:
            taxa = TaxaIndexMapping.default(num_taxa)
        else:
            assert False, "Must provide either the number of leaves or a TaxaIndexMapping"

    if num_taxa:
        assert num_taxa == len(taxa), "The number of leaves must match the number of taxa given."
    else:
        num_taxa = len(taxa)

    nodes = taxa.all_leaves(edge_length=edge_length)
    while len(nodes) > 1:
        # maintain this order, because we want the order of the taxa when we
        # iterate thru the leaves to be the same as specified in taxa argument
        b = nodes.pop()
        a = nodes.pop()
        nodes.append(merge_children((a,b), edge_length=edge_length))

    return dendropy.Tree(taxon_namespace=taxa.taxon_namespace, seed_node=nodes[0], is_rooted=False)

def unrooted_birth_death_tree(num_taxa, namespace=None, birth_rate=0.5, death_rate = 0, **kwargs):
    if namespace == None:
        namespace = default_namespace(num_taxa)
    tree = dendropy.model.birthdeath.birth_death_tree(birth_rate, death_rate, birth_rate_sd=0.0, death_rate_sd=0.0, taxon_namespace = namespace,  num_total_tips=num_taxa, **kwargs)
    tree.is_rooted = False
    return tree

def unrooted_pure_kingman_tree(taxon_namespace, pop_size=1, rng=None):
    tree = pure_kingman_tree(taxon_namespace, pop_size=pop_size, rng=rng)
    tree.is_rooted = False
    return tree

def unrooted_mean_kingman_tree(taxon_namespace, pop_size=1, rng=None):
    tree = mean_kingman_tree(taxon_namespace, pop_size=pop_size, rng=rng)
    tree.is_rooted = False
    return tree

# %%
if __name__ == "__main__":

    t = balanced_binary(4)
    t.print_plot()

    name = t.taxon_namespace
    len(name)
    mmm = np.array([[0,2,3,4],[2,0,7,8],[3,7,0,9],[4,8,9,0]])
    distmat = array2distance_matrix(mmm, name)
    dtable = distmat.as_data_table()
    dtable.write_csv("stdout")

    trees = dendropy.TreeList([t])
    trees
    trees.extend([t, t])
    trees.taxon_namespace is trees[0].taxon_namespace
    trees.taxon_namespace is trees[-1].taxon_namespace

    trees.taxon_namespace[2]

    all_data = temp_dataset_maker(trees, 1000, 1)
    t0_data = all_data[0]
    observations, alpha = charmatrix2array(t0_data)

    observations
    list(alpha)

    """model = dendropy.model.discrete.Jc69() #
    model = dendropy.model.discrete.DiscreteCharacterEvolver()
    mat = dendropy.model.discrete.simulate_discrete_chars(100, t, model)
    hh = mat[0]
    hh.symbols_as_string()"""

    pdm = dendropy.PhylogeneticDistanceMatrix(t)
    print(pdm)
    tt = pdm.as_data_table()

if __name__ == "__main__":
    tree = treesim.birth_death_tree(birth_rate=1., death_rate=0., num_total_tips=len(taxa), taxon_namespace=taxa)


    tree = treesim.birth_death_tree(birth_rate=1., death_rate=2., is_retain_extinct_tips=True, num_total_tips=len(taxa), taxon_namespace=taxa)

    tree.minmax_leaf_distance_from_root()

    print(tree.as_python_source())

    print(tree.as_ascii_plot())


    for e in tree.edges():
        print(e.length)

    for n in tree.nodes():
        print(n.edge_length)

    e = tree.edges()[5]
    e.length


    n0 = tree.nodes()[0]
    n0.branch
