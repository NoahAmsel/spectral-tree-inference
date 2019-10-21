from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg
import scipy.spatial.distance
import dendropy

import utils

def nchoose2(n):
    return int(n*(n-1)/2)

# TODO: make t and optional argument in all these functions
# and write a wrapper to add the statement
# if t is None: t = self.t
class Transition(ABC):
    """
    Abstract superclass for all kinds of transitions
    """
    @abstractmethod
    def transition_function(self, t):
        pass

    @abstractmethod
    def random_sequence(self, n):
        pass

    def __call__(self, input_sequence, t):
        return self.transition_function(t)(input_sequence)

    def transition(self, input_sequence, t):
        """
        Alias for self.__call__
        """
        return self(input_sequence, t=t)

    def generate_descendents_data(self, node, seed_data, scalar):
        node.seqence = seed_data
        for child in node.child_nodes():
            temp = self.transition(seed_data, child.edge_length)
            self.generate_descendents_data(child, temp, scalar)

    def generate_sequences(self, tree, seq_len=None, seed_data=None, scalar=1.0):
        if seed_data is None:
            assert seq_len is not None
            assert seq_len > 0
            seed_data = self.random_sequence(seq_len)
        else:
            assert seq_len == len(seed_data)

        self.generate_descendents_data(tree.seed_node, seed_data, scalar)
        #return dendropy.charmatrix()

    def generate_sequences_list(self, tree_list, seq_len=None, seed_data=None, scalar=1.0):
        for tree in tree_list:
            self.generate_sequences(tree, seq_len=seq_len, seed_data=seed_data, scalar=scalar)

    @staticmethod
    def transition_function_from_matrix(T):
        def transition(input_sequence):
            num_classes = T.shape[0] # TODO should this be [1]
            num_samples = len(input_sequence)
            # this is optimized for num_classes << num_samples
            # since each call to np.random.choice is extremely slow, we limit
            # ourselves to only num_classes calls rather than doing each of num_samples
            # coordinates in a separate call
            redundant_samples = [np.random.choice(a=num_classes, size=num_samples, p=T[i,:]) for i in range(num_classes)]
            mask = np.array([np.array(input_sequence) == i for i in range(num_classes)])
            return (redundant_samples*mask).sum(axis=0)
        transition.matrix = T
        return transition

class GTR(Transition):
    """
    General time-reversible model.
    See https://en.wikipedia.org/wiki/Models_of_DNA_evolution
    """
    def __init__(self, base_frequencies, transition_rates):
        self.k = len(base_frequencies)
        assert len(transition_rates) == nchoose2(self.k)
        # save these in case we can use seqgen
        self.base_frequencies = base_frequencies / base_frequencies.sum()
        self.transition_rates = transition_rates
        Q = scipy.spatial.distance.squareform(self.transition_rates)
        Q *= self.base_frequencies
        # set diagonal so that rows sum to 0
        np.fill_diagonal(Q, Q.diagonal()-Q.sum(axis=1))
        self.Q = GTR.scale_rate_matrix(Q, self.base_frequencies)

    @property
    def k_ratio(self):
        return (self.k-1) / self.k

    @staticmethod
    def scale_rate_matrix(unscaled_Q, baseline_frequences):
        expected_transitions = - unscaled_Q.diagonal().dot(baseline_frequences)
        return unscaled_Q / expected_transitions

    def transition_function(self, t):
        return GTR.transition_function_from_matrix(scipy.linalg.expm(self.Q*t))

    def random_sequence(self, length):
        length = int(length)
        return np.random.choice(a=list(range(self.k)), size=length, p=self.base_frequencies)

    def paralinear_distance(self, t):
        """
        Paralinear distance is - log det e^(Q*t) = -Tr(Q*t)
        """
        return -np.trace(self.Q) * t

    def paralinear2t(self, dist):
        return - dist / np.trace(self.Q)

    def seqgen(self, tree, seq_len=None, seed_data=None, scaler=None):
        tree_list = dendropy.TreeList([tree])
        self.seqgen_list(tree_list, seq_len=seq_len, seed_data=seed_data, scaler=scaler)

    def seqgen_list(self, tree_list, seq_len=None, seed_data=None, scaler=None):
        # seqgen only works for Amino Acid or Nucleotide data
        assert self.k == 4 or self.k == 20
        if seed_data is not None:
            if seq_len is None:
                seq_len = len(seed_data)
            else:
                assert seq_len == len(seed_data)

        s = dendropy.interop.seqgen.SeqGen()
        if self.k == 4:
            s.char_model = "GTR"
        elif self.k == 20:
            s.char_model = "GENERAL"
        s.seq_len = seq_len
        s.state_freqs = ",".join([str(s) for s in self.base_frequencies])
        s.general_rates = ",".join([str(r) for r in self.transition_rates])
        if scaler is not None:
            s.scale_branch_lens = scaler
        if seed_data is not None:
            states = np.array(list("ACGT" if k == 4 else "ARNDCQEGHILKMFPSTWYV"))
            s.ancestral_seq = states[seed_data]
        print(s._compose_arguments())
        d0 = s.generate(tree_list)
        print(type(d0))
        return d0.char_matrices

class Jukes_Cantor(GTR):
    def __init__(self, num_classes):
        base_frequencies = np.ones(num_classes)
        transition_rates = np.ones(nchoose2(num_classes))
        super().__init__(base_frequencies=base_frequencies, transition_rates=transition_rates)

    def t2p(self, t):
        """
        Returns the probability of not transitioning given a branch of length t
        """
        #return self.transition_function(t).matrix[0,0]
        #return 0.25 + 0.75 * np.exp(-4.*t/3.)
        return 1./k + self.k_ratio * np.exp(- t / self.k_ratio)

    def p2t(self, p):
        """
        Returns the branch length t necessary for the probability of not
        transitioning to be p.
        """
        #return -(3./4) * np.log((4./3)*p - (1./3))
        return - self.k_ratio * np.log( p / self.k_ratio - 1./(self.k-1))

    def p2transition_function(self, p):
        return self.transition_function(self.p2t(p))

# %%

if __name__ == "__main__":
    jc = Jukes_Cantor(4)
    t = utils.lopsided_tree(4)
    trees = dendropy.TreeList([t])
    seqlists = jc.seqgen(trees, 100, scaler=0.3)
    type(seqlists)
    seqs = seqlists[0]

    mat, _ = utils.charmatrix2array(seqs)
    mat.shape

    jc.p2transition_function(0.90).matrix
    jc.paralinear2t(0.5)


    model = dendropy.model.discrete.Jc69() #
    my_model = dendropy.model.discrete.DiscreteCharacterEvolver()
    mat = dendropy.model.discrete.simulate_discrete_chars(100, t, my_model)
    hh = mat[0]
    hh.symbols_as_string()
