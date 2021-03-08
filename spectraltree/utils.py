from collections import defaultdict
from collections.abc import Mapping
from functools import reduce
from itertools import combinations

import dendropy
import igraph
import numpy as np
import scipy.spatial.distance

def default_namespace(num_taxa, prefix="T"):
    return dendropy.TaxonNamespace([prefix+str(i) for i in range(1, num_taxa+1)])

class TaxaMetadata(Mapping):
    _alphabet_label2obj = {
        "DNA": dendropy.DNA_STATE_ALPHABET,
        "RNA": dendropy.RNA_STATE_ALPHABET,
        "Nucleotide": dendropy.NUCLEOTIDE_STATE_ALPHABET,
        "Protein": dendropy.PROTEIN_STATE_ALPHABET,
        "Binary": dendropy.BINARY_STATE_ALPHABET,
    }

    def __init__(self, taxon_namespace, taxa_list, alphabet=None):
        if isinstance(alphabet, dendropy.StateAlphabet) or (alphabet is None):
            self._alphabet = alphabet
        else:
            self._alphabet = self._alphabet_label2obj[alphabet]

        # TODO: modify init so that it detects repeated values (even if it's specified as a taxon object one time and as a label the other) and throws an error
        self._taxon_namespace = taxon_namespace

        # for fast label lookups
        self._label2taxon = {taxon.label: taxon for taxon in self._taxon_namespace}

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
                raise ValueError(f"Each taxon must be included in the given taxon namespace, but {taxon} isn't.")

        self._taxa_list = np.array(self._taxa_list)

    @classmethod
    def whole_namespace(cls, namespace):
        return cls(namespace, list(namespace))

    @classmethod
    def default(cls, length):
        return cls.whole_namespace(default_namespace(length))

    @property
    def alphabet(self):
        return self._alphabet

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
        return self._label2taxon.get(taxon_or_label, taxon_or_label)

    def _convert_labels(self, taxa_or_labels):
        return [self._convert_label(t_or_l) for t_or_l in taxa_or_labels]

    def index2taxa(self, indexer):
        return self._taxa_list[indexer]

    def taxon2mask(self, taxon):
        mask = np.zeros(len(self), dtype=bool)
        mask[self[taxon]] = True
        return mask

    def taxa2mask(self, taxa):
        mask = np.zeros(len(self), dtype=bool)
        mask[[self[taxon] for taxon in taxa]] = True
        return mask

    def mask2taxa(self, mask):
        # need comma because output is a tuple of one element
        taxa_ix, = mask.nonzero()
        return [self.index2taxa(ix) for ix in taxa_ix]

    def taxa2bipartition(self, taxa):
        """
        taxa is list of taxon objects or labels
        """
        taxa = self._convert_labels(taxa)
        return self._taxon_namespace.taxa_bipartition(taxa=taxa)

    def bipartition2taxa(self, bipartition):
        """
        Given a dendropy bipartition, output a list of taxa (objects)
        that it includes
        """
        return bipartition.leafset_taxa(self._taxon_namespace)

    def mask2bipartition(self, mask):
        return self.taxa2bipartition(self.index2taxa(mask))

    def bipartition2mask(self, bipartition):
        return self.taxa2mask(self.bipartition2taxa(bipartition))

    def taxa2sub_taxa_metadata(self, taxa):
        # TODO: replace TaxaMetadata with typeof(self)
        return TaxaMetadata(self.taxon_namespace, taxa, self.alphabet)

    def mask2sub_taxa_metadata(self, mask):
        taxa = self.mask2taxa(mask)
        return self.taxa2sub_taxa_metadata(taxa)

    def tree2mask(self, tree):
        """
        Warning! This is slow. It's faster to store the mask as a 
        member of the tree rather than calling this function again and again.
        """
        return self.taxa2mask([leaf.taxon for leaf in tree.leaf_nodes()])

    def invert_mask_in_tree(self, tree, mask):
        tree_mask = self.tree2mask(tree)
        # assure mask is a subset of the nodes in the tree
        if np.logical_and(np.logical_not(tree_mask), mask).any():
            raise ValueError("Provided mask includes nodes that are not in the tree.")
        return tree_mask ^ mask

    def invert_bipartition_in_tree(self, tree, bipartition):
        return self.invert_mask_in_tree(self.bipartition2mask(bipartition))

    def leaf(self, taxon, **kwargs):
        taxon = self._convert_label(taxon)
        if taxon not in self:
            raise ValueError("Given taxon must be in the TaxaMetadata object")

        kwargs['taxon'] = taxon
        node = dendropy.Node(**kwargs)
        node.mask = self.taxon2mask(taxon)
        return node

    def all_leaves(self, **kwargs):
        return [self.leaf(taxon, **kwargs) for taxon in self]
        
    def reindex_matrix(self, matrix, old_taxa, axes=[0]):
        assert False
        if not self.equals_unordered(old_taxa):
            raise ValueError("Old and new taxa maps must contain the same set of taxa.")
        new_matrix = np.zeros_like(matrix)
        # how do we generalize this for different numbers of axes?

    def __str__(self):
        return str([taxon.label for taxon in self])

    def __eq__(self, other):
        return isinstance(other, TaxaMetadata) and (
            self.taxon_namespace == other.taxon_namespace) and (
                self.alphabet == other.alphabet) and (
                    all(self._taxa_list == other._taxa_list))

    def equals_unordered(self, other):
        return isinstance(other, TaxaMetadata) and (
            self.taxon_namespace == other.taxon_namespace) and (
                self.alphabet == other.alphabet) and (
                    set(self._taxa_list) == set(other._taxa_list))

##########################################################
##               Converters
##########################################################

def charmatrix2array(charmatrix):
    #charmatrix[taxon].values()
    alphabet = charmatrix.state_alphabets[0] # TODO: what if there are multiple different state_alphabets?
    taxa = []
    sequences = []
    for taxon in charmatrix:
        taxa.append(taxon)
        sequences.append([state_id.index for state_id in charmatrix[taxon].values()])
    
    return np.array(sequences), TaxaMetadata(charmatrix.taxon_namespace, taxa, alphabet=alphabet)

def array2charmatrix(matrix, taxa_metadata=None):
    if taxa_metadata is None:
        taxa_metadata = TaxaMetadata.default(matrix.shape[0])
    else:
        if len(taxa_metadata) != matrix.shape[0]:
            raise ValueError(f"Size of TaxaMetadata ({len(taxa_metadata)}) does not match size of matrix ({matrix.shape[0]}).")
    
    alphabet = taxa_metadata.alphabet
    # TODO: add support for DNACharacterMatrix and others
    if alphabet is None:
        # input the values in the matrix directly
        alphabet = dendropy.new_standard_state_alphabet(val for val in np.unique(matrix))
        char_matrix = dendropy.StandardCharacterMatrix(default_state_alphabet=alphabet, taxon_namespace=taxa_metadata.taxon_namespace)
        for taxon, ix in taxa_metadata.items():
            char_matrix.new_sequence(taxon, [str(x) for x in matrix[ix, :]])
    else:
        # assume values in the matrix are indices into the alphabet
        alpha2matrix_class = {
            dendropy.DNA_STATE_ALPHABET: dendropy.DnaCharacterMatrix,
            dendropy.RNA_STATE_ALPHABET: dendropy.RnaCharacterMatrix,
            dendropy.NUCLEOTIDE_STATE_ALPHABET: dendropy.NucleotideCharacterMatrix,
            dendropy.PROTEIN_STATE_ALPHABET: dendropy.ProteinCharacterMatrix,
            dendropy.BINARY_STATE_ALPHABET: dendropy.RestrictionSitesCharacterMatrix,
        }
        if alphabet in alpha2matrix_class:
            char_matrix = alpha2matrix_class[alphabet](taxon_namespace=taxa_metadata.taxon_namespace)
        else:
            char_matrix = dendropy.StandardCharacterMatrix(default_state_alphabet=alphabet, taxon_namespace=taxa_metadata.taxon_namespace)

        for taxon, ix in taxa_metadata.items():
            # you need .item to convert from numpy.int64 to int. dendropy expects only int
            char_matrix.new_sequence(taxon, [alphabet[v.item()] for v in matrix[ix, :]])

    return char_matrix

def array2distance_matrix(matrix, taxa_metadata=None):
    m, m2 = matrix.shape
    if m != m2:
        raise ValueError(f"Distance matrix should be square but has dimensions {m} x {m2}.")
    if taxa_metadata is None:
        taxa_metadata = TaxaMetadata.default(m)
    else:
        if len(taxa_metadata) != m:
            raise ValueError(f"Namespace size ({len(taxa_metadata)}) should match distance matrix dimension ({m}).")

    dict_form = defaultdict(dict)
    for row_taxon in taxa_metadata:
        for column_taxon in taxa_metadata:
            dict_form[row_taxon][column_taxon] = matrix[taxa_metadata[row_taxon], taxa_metadata[column_taxon]]
    dm = dendropy.calculate.phylogeneticdistance.PhylogeneticDistanceMatrix()
    dm.compile_from_dict(dict_form, taxa_metadata.taxon_namespace)
    return dm

def distance_matrix2array(dm):
    """
    This is patristic distance: adding the branch lengths. If we set branches to
    have different transitions, this won't be the paralinear distance
    """
    taxa = TaxaMetadata(dm.taxon_namespace, list(dm.taxon_iter()))
    return scipy.spatial.distance.squareform([dm.distance(taxon1, taxon2) for taxon1, taxon2 in combinations(taxa,2)]), taxa

def tree2distance_matrix(tree):
    return distance_matrix2array(tree.phylogenetic_distance_matrix())

def adjacency_matrix_to_tree_igraph(A,num_taxa,taxa_metadata):
    G = igraph.Graph.Adjacency((A > 0).tolist())
    merges = [x.tuple for x in G.es]
    T = igraph.Dendrogram(merges)

def adjacency_matrix_to_tree(A,num_taxa,taxa_metadata):
    m = A.shape[0]
    edge_length = 1
    nodes = taxa_metadata.all_leaves(edge_length=edge_length)
    active_nodes = np.arange(num_taxa)
    
    for p_idx in np.arange(num_taxa,m):        
        #child_idx = np.where(A[p_idx,active_nodes]>0)[0]  
        a = A[p_idx,active_nodes]
        child_idx = [i for i in range(len(a)) if a[i] > 0]            
        child_nodes = [nodes[i] for i in active_nodes[child_idx]]
        p_node = merge_children(child_nodes, edge_length=edge_length)    
        nodes.append(p_node)
        active_nodes = np.delete(active_nodes,child_idx)
        active_nodes = np.append(active_nodes,p_idx)
        
    return dendropy.Tree(taxon_namespace=taxa_metadata.taxon_namespace, seed_node=nodes[-1], is_rooted=False)    

##########################################################
##               Tree Generation
##########################################################

# Including these functions here for convenience
from dendropy.simulate.treesim import birth_death_tree, pure_kingman_tree, mean_kingman_tree 

def merge_children(children, **kwargs):
    if len(children) == 0:
        raise ValueError
    node = dendropy.Node(**kwargs)
    for child in children:
        node.add_child(child)
    if all(hasattr(child,'mask') for child in children):
        node.mask = reduce(np.logical_or, (child.mask for child in children))
    return node

def balanced_binary(num_taxa=None, taxa=None, edge_length=1.):
    if taxa is None:
        if num_taxa:
            taxa = TaxaMetadata.default(num_taxa)
        else:
            raise TypeError("Must provide either the number of leaves or a TaxaMetadata")

    if num_taxa:
        if num_taxa != len(taxa):
            raise ValueError(f"The desired number of leaves {num_taxa} must match the number of taxa given {len(taxa)}.")
    else:
        num_taxa = len(taxa)

    if num_taxa != 2**int(np.log2(num_taxa)):
        raise ValueError(f"The number of leaves in a balanced binary tree must be a power of 2, not {num_taxa}.")

    nodes = taxa.all_leaves(edge_length=edge_length)
    while len(nodes) > 1:
        nodes = [merge_children(nodes[2*i : 2*i+2], edge_length=edge_length) for i in range(len(nodes)//2)]

    return dendropy.Tree(taxon_namespace=taxa.taxon_namespace, seed_node=nodes[0], is_rooted=False)

def lopsided_tree(num_taxa, taxa=None, edge_length=1.):
    """One node splits off at each step"""
    if taxa is None:
        if num_taxa:
            taxa = TaxaMetadata.default(num_taxa)
        else:
            raise TypeError("Must provide either the number of leaves or a TaxaMetadata")

    if num_taxa:
        if num_taxa != len(taxa):
            raise ValueError(f"The desired number of leaves {num_taxa} must match the number of taxa given {len(taxa)}.")
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

def unrooted_pure_kingman_tree(num_taxa,taxon_namespace=None, pop_size=1, rng=None):
    if taxon_namespace == None:
        taxon_namespace = default_namespace(num_taxa)
    tree = pure_kingman_tree(taxon_namespace, pop_size=pop_size, rng=rng)
    tree.is_rooted = False
    return tree

def unrooted_mean_kingman_tree(taxon_namespace, pop_size=1, rng=None):
    tree = mean_kingman_tree(taxon_namespace, pop_size=pop_size, rng=rng)
    tree.is_rooted = False
    return tree

##########################################################
##               Miscellaneous
##########################################################

def compare_trees(reference_tree, inferred_tree):
    inferred_tree.update_bipartitions()
    reference_tree.update_bipartitions()
    false_positives, false_negatives = dendropy.calculate.treecompare.false_positives_and_negatives(reference_tree, inferred_tree, is_bipartitions_updated=True)
    total_reference = len(reference_tree.bipartition_encoding)
    total_inferred = len(inferred_tree.bipartition_encoding)
    
    true_positives = total_inferred - false_positives
    precision = true_positives / total_inferred
    recall = true_positives / total_reference

    F1 = 100* 2*(precision * recall)/(precision + recall)
    RF = false_positives + false_negatives
    return RF, F1

def topos_equal(reference_tree, inferred_tree):
    return dendropy.calculate.treecompare.symmetric_difference(reference_tree, inferred_tree) == 0

def set_edge_lengths(tree, value=None, fun=None, uniform_range=None):
    if value is not None:
        assert fun is None and uniform_range is None
        for e in tree.edges():
            e.length = value
    elif fun is not None:
        assert uniform_range is None
        for e in tree.edges():
            e.length = fun(e.length)
    else:
        assert uniform_range is not None
        for e in tree.edges():
            e.length = np.random.uniform(*uniform_range)