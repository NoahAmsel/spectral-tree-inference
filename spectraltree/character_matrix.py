from collections.abc import Mapping
import dendropy

class FastCharacterMatrix(Mapping):

    def __getitem__(self, taxon):
        return self.matrix[self.taxon2index[taxon], :]

    def __iter__(self):
        for taxon in taxon2index:
            yield taxon

    def __len__(self):
        return len(self.matrix)

    """
    def __setitem__(self, taxon, sequence):
        assert len(sequence) == self.matrix.shape[1]
        if taxon in self.taxon2index:
            self.matrix[self.taxon2index[taxon], :] = sequence
        else:
            self.taxon2index[taxon] = len(self.matrix)
            self.matrix = np.append(self.matrix, sequence.reshape(1,-1), axis=0)

    def __delitem__(self, taxon):
        pass # TODO
    """

    def __init__(self, matrix, taxon_namespace=None, taxon2index=None, alphabet=None):
        self.matrix = matrix
        if not taxon_namespace:
            assert taxon2index is None
            taxon_namespace = default_namespace(len(self.matrix))
        self.taxon_namespace = namespace
        if taxon2index:
            for taxon in taxon2index:
                assert taxon in self.taxon_namespace
        else:
            # default is just to use the iterated order of the taxon_namespace object
            taxon2index = {taxon: ix for ix, taxon in enumerate(self.taxon_namespace)}
        self.taxon2index = taxon2index
        self.alphabet = alphabet

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


    def to_dendropy(alphabet=None):
        if alphabet is None:
            if self.alphabet:
                alphabet = self.alphabet

        char_matrix = dendropy.StandardCharacterMatrix()
        for taxon, sequence in self.items():
            char_matrix.new_sequence(taxon, [alphabet[v] for v in values] if alphabet else values)

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
