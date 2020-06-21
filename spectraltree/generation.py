from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg
import scipy.spatial.distance
import dendropy

from character_matrix import TaxaMetadata

def nchoose2(n):
    return int(n*(n-1)/2)

class Transition(ABC):
    """
    Transition for use with dendropy's `DiscreteCharacterEvolver` class.
    A transition describes how a sequence evolves along the edge from a parent node to a child node.
    """

    @abstractmethod
    def simulate_descendant_states(self, ancestral_states, edge_length, mutation_rate=1.):
        pass

    @abstractmethod
    def stationary_sample(self, seq_len, rng=None):
        pass

class GaussianTransition(Transition):
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def simulate_descendant_states(self, ancestral_states, edge_length, mutation_rate=1.):
        return np.random.normal(loc=(ancestral_states*self.w + self.b), scale=edge_length*mutation_rate)

    def stationary_sample(self, seq_len, rng=None):
        return np.zeros(seq_len)

class DiscreteTransition(Transition):
    """
    Transition over discrete set of states. Replaces dendropy's
    `DiscreteCharacterEvolutionModel` class, using numpy to improve speed.
    To use, subclass and implement the `pmatrix` method, which returns the
    transition matrix.
    """

    def __init__(self, stationary_freqs):
        super().__init__()
        self._k = len(stationary_freqs)
        self._stationary_freqs = stationary_freqs / stationary_freqs.sum()

    @property
    def k(self):
        return self._k

    @property
    def stationary_freqs(self):
        return self._stationary_freqs

    @abstractmethod
    def pmatrix(self, tlen, mutation_rate=1.):
        pass

    def simulate_descendant_states(self, ancestral_states, edge_length, mutation_rate=1.):
        T = self.pmatrix(edge_length, mutation_rate)
        return self.transition(T, ancestral_states)

    def stationary_sample(self, seq_len, rng=None):
        return np.random.choice(a=list(range(self.k)), size=seq_len, p=self.stationary_freqs)

    @staticmethod
    def transition(T, ancestral_states):
        num_classes = T.shape[0] # TODO should we add support for different numbers of states in different levels?
        num_samples = len(ancestral_states)
        # this is optimized for num_classes << num_samples
        # since each call to np.random.choice is extremely slow, we limit
        # ourselves to only num_classes calls rather than doing each of num_samples
        # coordinates in a separate call
        redundant_samples = [np.random.choice(a=num_classes, size=num_samples, p=T[i,:]) for i in range(num_classes)]
        mask = np.array([np.array(ancestral_states) == i for i in range(num_classes)])
        return (redundant_samples*mask).sum(axis=0)

class FixedDiscreteTransition(DiscreteTransition):
    def __init__(self, stationary_freqs, pmatrix):
        assert pmatrix.shape[0] == pmatrix.shape[1] == len(stationary_freqs)
        assert np.all(pmatrix >= 0)
        super().__init__(stationary_freqs)
        self._pmatrix = pmatrix

    def pmatrix(self, t=None, mutation_rate=1.):
        return self._pmatrix

    def __eq__(self, other):
        return np.all(self._pmatrix == other._pmatrix)

class ContinuousTimeDiscreteTransition(DiscreteTransition):
    def __init__(self, stationary_freqs, Q):
        assert Q.shape[0] == Q.shape[1] == len(stationary_freqs)
        super().__init__(stationary_freqs)
        assert np.allclose(Q.sum(axis=1), np.zeros(self.k))
        self._Q = self.scale_rate_matrix(Q, self.stationary_freqs)

    @property
    def Q(self):
        return self._Q

    @staticmethod
    def scale_rate_matrix(unscaled_Q, stationary_freqs):
        expected_transitions = - unscaled_Q.diagonal().dot(stationary_freqs)
        return unscaled_Q / expected_transitions

    def pmatrix(self, t, mutation_rate=1.):
        return scipy.linalg.expm(self.Q * t * mutation_rate)

    def paralinear_distance(self, t, mutation_rate=1.):
        """
        Paralinear distance is - log det P = - log det e^(Q*t) = -Tr(Q*t)
        """
        return -np.trace(self.Q) * t * mutation_rate

    def paralinear2t(self, dist, mutation_rate=1.):
        return dist / (-np.trace(self.Q) * mutation_rate)

    def similarity(self, t, mutation_rate=1.):
        """
        delta = det(P) = exp(- paralinear distance)
        """
        return np.exp(-self.paralinear_distance(t, mutation_rate=mutation_rate))

    def similarity2t(self, similarity, mutation_rate=1.):
        """
        paralinear distance = - log delta
        """
        return self.paralinear2t(-np.log(similarity), mutation_rate=1.)

    def __eq__(self, other):
        return np.all(self.Q == other.Q)

    def __str__(self):
        return "Continuous-Time transition with rate matrix\n" + str(self.Q)

class GTR(ContinuousTimeDiscreteTransition):
    def __init__(self, stationary_freqs, transition_rates):
        assert len(transition_rates) == nchoose2(len(stationary_freqs))
        # save this in case we can use seqgen
        Q = scipy.spatial.distance.squareform(transition_rates)
        Q *= stationary_freqs
        # set diagonal so that rows sum to 0
        np.fill_diagonal(Q, Q.diagonal() - Q.sum(axis=1))
        super().__init__(stationary_freqs, Q)
        self._transition_rates = transition_rates

class TN93(GTR):
    def __init__(self, stationary_freqs, kappa1, kappa2):
        #  we could allow for more than for classes
        # assume the first half are of one type ("pyramidines") and the second half is of the other type ("purines")
        # but right now we don't
        assert len(stationary_freqs) == 4
        transition_rates = np.array([kappa1, 1, 1, 1, 1, kappa2]).astype(float)
        super().__init__(stationary_freqs, transition_rates)

class T92(TN93):
    def __init__(self, theta, kappa1, kappa2):
        #  we could allow for more than for classes
        # assume the first half are of one type ("pyramidines") and the second half is of the other type ("purines")
        # but right now we don't
        assert 0. <= theta <= 1.
        GC = theta / 2.
        AT = (1. - theta) / 2.
        super().__init__(np.array([AT, GC, GC, AT]), kappa1, kappa2)

class HKY(TN93):
    def __init__(self, stationary_freqs, kappa):
        super().__init__(stationary_freqs, kappa, kappa)

class Jukes_Cantor(GTR):
    def __init__(self, num_classes=4):
        base_frequencies = np.ones(num_classes)
        transition_rates = np.ones(nchoose2(num_classes))
        super().__init__(stationary_freqs=base_frequencies, transition_rates=transition_rates)

    def k_ratio(self):
        return (self.k-1) / self.k

    def t2p(self, t, mutation_rate=1.):
        """
        Returns the probability of not transitioning given a branch of length t
        """
        #return self.pmatrix(t, mutation_rate)[0,0]
        #return 0.25 + 0.75 * np.exp(-4.*t/3.)
        return 1./self.k + self.k_ratio() * np.exp(- t*mutation_rate / self.k_ratio())

    def p2t(self, p, mutation_rate=1.):
        """
        Returns the branch length t necessary for the probability of not
        transitioning to be p.
        """
        #return -(3./4) * np.log((4./3)*p - (1./3))
        return - self.k_ratio() * np.log( p / self.k_ratio() - 1./(self.k-1)) / mutation_rate

    def p2pmatrix(self, p, mutation_rate=1.):
        return self.pmatrix(self.p2t(p, mutation_rate))

    def __str__(self):
        suff = "" if self.k==4 else " (k={})".format(self.k)
        return "Jukes Cantor" + suff

def numpy_matrix_with_characters_on_tree_ordered(seq_attr, tree):
    """
    Extracts sequences from all leaves and packs them into a numpy matrix.
    Repalces `extend_char_matrix_with_characters_on_tree` method of `DiscreteCharacterEvolver`, which doesn't use numpy.
    """
    sequences = [0 for i in tree.taxon_namespace]
    for leaf_ix, leaf in enumerate(tree.leaf_node_iter()):
        
        sequences[tree.taxon_namespace[:].index(leaf.taxon)] = getattr(leaf, seq_attr)[-1]
        #sequences.append(np.concatenate(getattr(leaf, seq_attr)))
    return np.array(sequences)


def OLD_numpy_matrix_with_characters_on_tree(seq_attr, tree):
    """
    Extracts sequences from all leaves and packs them into a numpy matrix.
    Repalces `extend_char_matrix_with_characters_on_tree` method of `DiscreteCharacterEvolver`, which doesn't use numpy.
    """
    index_map = {taxon: ix for ix, taxon in enumerate(tree.taxon_namespace)}
    sequences = [None]*len(tree.leaf_nodes())
    assert len(sequences) == len(index_map)  # exactly one entry in taxon_namespace for each leaf
    for leaf in tree.leaf_node_iter():
        sequences[index_map[leaf.taxon]] = getattr(leaf, seq_attr)[-1]
        #sequences[index_map(leaf.taxon)] = np.concatenate(getattr(leaf, seq_attr))
    return np.array(sequences)

def numpy_matrix_with_characters_on_tree(seq_attr, tree):
    """
    Extracts sequences from all leaves and packs them into a numpy matrix.
    Repalces `extend_char_matrix_with_characters_on_tree` method of `DiscreteCharacterEvolver`, which doesn't use numpy.
    """
    taxa = []
    sequences = []
    for leaf in tree.leaf_node_iter():
        taxa.append(leaf.taxon)
        sequences.append(getattr(leaf, seq_attr)[-1])
        # sequences.append(np.concatenate(getattr(leaf, seq_attr)))  # TODO
    
    return np.array(sequences), TaxaMetadata(tree.taxon_namespace, taxa)

def simulate_sequences_ordered(seq_len, tree_model, seq_model, mutation_rate=1.0, root_states=None, retain_sequences_on_tree=False, rng=None):
    """
    Convenience function that generates a matrix of sequence observations from a given sequence model and tree

    Parameters
    ----------

    seq_len       : int
        Length of sequence (number of characters).
    tree_model    : |Tree|
        Tree on which to simulate.
    seq_model     : |Transition|
        The character substitution model under which to to evolve the
        characters.
    mutation_rate : float
        Mutation *modifier* rate (should be 1.0 if branch lengths on tree
        reflect true expected number of changes).
    root_states   : list
        Vector of root states (length must equal ``seq_len``).
    retain_sequences_on_tree : bool
        If |False|, sequence annotations will be cleared from tree after
        simulation. Set to |True| if you want to, e.g., evolve and accumulate
        different sequences on tree, or retain information for other purposes.
    rng           : random number generator
        If not given, 'GLOBAL_RNG' will be used.

    Returns
    -------

    char_matrix :  |numpy.array|
        Matrix where each row is the sequence generated for a given leaf.

    """
    seq_evolver = dendropy.model.discrete.DiscreteCharacterEvolver(seq_model=seq_model, mutation_rate=mutation_rate)
    tree = seq_evolver.evolve_states(
        tree=tree_model,
        seq_len=seq_len,
        root_states=root_states,
        rng=rng)
    char_matrix = numpy_matrix_with_characters_on_tree_ordered(seq_evolver.seq_attr, tree_model)
    if not retain_sequences_on_tree:
        seq_evolver.clean_tree(tree)
    return char_matrix


def simulate_sequences(seq_len, tree_model, seq_model, mutation_rate=1.0, root_states=None, retain_sequences_on_tree=False, rng=None):
    """
    Convenience function that generates a matrix of sequence observations from a given sequence model and tree

    Parameters
    ----------

    seq_len       : int
        Length of sequence (number of characters).
    tree_model    : |Tree|
        Tree on which to simulate.
    seq_model     : |Transition|
        The character substitution model under which to to evolve the
        characters.
    mutation_rate : float
        Mutation *modifier* rate (should be 1.0 if branch lengths on tree
        reflect true expected number of changes).
    root_states   : list
        Vector of root states (length must equal ``seq_len``).
    retain_sequences_on_tree : bool
        If |False|, sequence annotations will be cleared from tree after
        simulation. Set to |True| if you want to, e.g., evolve and accumulate
        different sequences on tree, or retain information for other purposes.
    rng           : random number generator
        If not given, 'GLOBAL_RNG' will be used.

    Returns
    -------

    char_matrix :  |numpy.array|
        Matrix where each row is the sequence generated for a given leaf.

    """
    seq_evolver = dendropy.model.discrete.DiscreteCharacterEvolver(seq_model=seq_model, mutation_rate=mutation_rate)
    tree = seq_evolver.evolve_states(
        tree=tree_model,
        seq_len=seq_len,
        root_states=root_states,
        rng=rng)
    char_matrix = numpy_matrix_with_characters_on_tree(seq_evolver.seq_attr, tree_model)
    if not retain_sequences_on_tree:
        seq_evolver.clean_tree(tree)
    return char_matrix

if __name__ == "__main__":
    from utils import balanced_binary

    TT = np.array([[0.9, 0.1], [0.4, 0.6]])
    my_trans = FixedDiscreteTransition(np.array([0,1]), TT)
    my_tree = balanced_binary(4)
    my_tree.edges()[-1].seq_model = FixedDiscreteTransition(np.array([0,1]), np.array([[1,0],[1,0]]))
    matrix, taxa = simulate_sequences(1000, my_tree, my_trans, retain_sequences_on_tree=True)
    print(matrix.mean(axis=1))


if __name__ == "__main__":
    jc = Jukes_Cantor()
    ttt = 4.
    print(0.25 + 0.75 * np.exp(-4.*ttt/3.))
    print(jc.pmatrix(2,2))
    jc.paralinear_distance(2,2)
    print(jc.p2t(jc.t2p(0.03, 2.), 2.))
