from collections.abc import Mapping
import numpy as np
import dendropy
#import utils

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
        
    def __str__(self):
        return str([taxon.label for taxon in self])



"""
class FastCharacterMatrix(Mapping):

    def __getitem__(self, taxon):
        return self.matrix[self.taxon2index[taxon], :]
    
    
    def __iter__(self):
        for taxon in self.taxon2index:
            yield taxon

    def __len__(self):
        return len(self.matrix)

    #
    # def __setitem__(self, taxon, sequence):
    #    assert len(sequence) == self.matrix.shape[1]
    #    if taxon in self.taxon2index:
    #        self.matrix[self.taxon2index[taxon], :] = sequence
    #    else:
    #        self.taxon2index[taxon] = len(self.matrix)
    #        self.matrix = np.append(self.matrix, sequence.reshape(1,-1), axis=0)

    #def __delitem__(self, taxon):
    #    pass # TODO
    # 

    def __init__(self, matrix, taxon_namespace=None, taxon2index=None, alphabet=None):
        self.matrix = matrix
        if not taxon_namespace:
            assert taxon2index is None
            taxon_namespace = utils.default_namespace(len(self.matrix))
        self.taxon_namespace = taxon_namespace
        if taxon2index:
            for taxon in taxon2index:
                assert taxon in self.taxon_namespace
        else:
            # default is just to use the iterated order of the taxon_namespace object
            taxon2index = {taxon: ix for ix, taxon in enumerate(self.taxon_namespace)}
        self.taxon2index = taxon2index
        self.alphabet = alphabet
        #if alphabet:
        #    self.alphabet = alphabet
        #else:
        #    self.alphabet = ['A','C','T','G']


    @classmethod
    def from_dictionary(cls, taxon2array, taxon_namespace, alphabet=None):
        taxon2index = {}
        sequence_array = []
        for taxon, array in taxon2array.items():
            assert taxon in taxon_namespace
            taxon2index[taxon] = len(sequence_array)
            sequence_array.append(array)
        matrix = np.array(sequence_array)
        return cls(matrix, taxon_namespace=taxon_namespace, taxon2index=taxon2index, alphabet=alphabet)

    @classmethod
    def from_dendropy(cls, charmatrix):
        # convert the Dendropy DnaCharacterDataSequence objects into arrays of ints
        taxon2array = {taxon: [state_id.index for state_id in sequence.values()] for taxon, sequence in charmatrix.items()}
        alphabet = charmatrix.state_alphabets[0]    #TODO: something more intelligent here
        return cls.from_dictionary(taxon2array, charmatrix.taxon_namespace, alphabet=alphabet)

    @classmethod
    def from_leaf_sequences(cls, tree, seq_attr):
        taxon2array = {leaf.taxon: getattr(leaf, seq_attr)[-1] for leaf in tree.leaf_node_iter()}
        return cls.from_dictionary(taxon2array, tree.taxon_namespace)


    def to_dendropy(self, alphabet=None):
        if alphabet is None:
            if self.alphabet:
                alphabet = self.alphabet
        
        char_matrix = dendropy.StandardCharacterMatrix()
        char_matrix.taxon_namespace = self.taxon_namespace
        for taxon, sequence in self.items():
            char_matrix.new_sequence(taxon, [alphabet[v] for v in sequence] if alphabet else [str(x) for x in sequence])

        return char_matrix


import numpy as np
from generation import FixedDiscreteTransition, simulate_sequences, Jukes_Cantor
from utils import balanced_binary
import dendropy
if False:
    my_tree = balanced_binary(4)
    #TT = np.array([[0.9, 0.1], [0.4, 0.6]])
    #my_trans = FixedDiscreteTransition(TT, np.array([0,1]))
    my_trans = Jukes_Cantor()
    #my_tree.edges()[-1].seq_model = FixedDiscreteTransition(np.array([[1,0],[1,0]]), np.array([0,1]))
    #seq_evolver = dendropy.model.discrete.DiscreteCharacterEvolver(seq_model=my_trans, mutation_rate=1.)
    #seq_evolver.extend_char_matrix_with_characters_on_tree()
    #char_matrix = numpy_matrix_with_characters_on_tree_ordered(seq_evolver.seq_attr, tree_model)
    npmat = simulate_sequences(1000, my_tree, my_trans)


    data = dendropy.model.discrete.simulate_discrete_chars(1000, my_tree, dendropy.model.discrete.Jc69())

    print(data.taxon_namespace)

    for n, seq in data.items():
        print(n, seq)

    zz = np.zeros((5,3))
    zz
    np.append(zz, np.ones(3).reshape(1,-1), axis=0)
    len(zz)
"""