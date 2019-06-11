import numpy as np
import Bio.Phylo as Phylo
from Bio.Phylo.Consensus import _bitstring_topology, _BitString
from string import ascii_lowercase

# TODO: right now this uses Biopython, but also check out dendropy

def random_string(stringLength=5):
    """Generate a random string of fixed length """
    letters = list(ascii_lowercase)
    return ''.join(np.random.choice(letters) for _ in range(stringLength))

def tree_Fscore(inferred, reference, rooted=False, verbose=True):
    term_names_inferred = set(term.name for term in inferred.find_clades(terminal=True))
    term_names_reference = set(term.name for term in reference.find_clades(terminal=True))
    assert term_names_inferred == term_names_reference
    # these are set objects
    splits_inferred = inferred.root.find_splits(symmetric=(not rooted), branch_lengths=False)
    splits_reference = reference.root.find_splits(symmetric=(not rooted), branch_lengths=False)

    RF = len(splits_inferred ^ splits_reference) # Robinson-Foulds is size of symmetric difference
    both = len(splits_inferred & splits_reference)
    precision = float(both)/len(splits_inferred)
    recall = float(both)/len(splits_reference)
    F1 = 2*(precision*recall)/(precision+recall)
    if verbose:
        print("RF: {:.2f}\t\tF1: {:.2f}%".format(RF, 100*precision, 100*recall, 100*F1))
    return F1, precision, recall, RF

def quartet_test():
    # TODO
    pass

def equal_topology(tree1, tree2, rooted=False, branch_lengths=False):
    # https://biopython.org/DIST/docs/api/Bio.Phylo.Consensus-pysrc.html#_equal_topology
    term_names1 = set(term.name for term in tree1.find_clades(terminal=True))
    term_names2 = set(term.name for term in tree2.find_clades(terminal=True))
    splits1 = tree1.root.find_splits(symmetric=(not rooted), branch_lengths=branch_lengths)
    splits2 = tree2.root.find_splits(symmetric=(not rooted), branch_lengths=branch_lengths)
    return (term_names1 == term_names2) and (splits1 == splits2)

class NoahClade(Phylo.BaseTree.Clade):
    def __init__(self, branch_length=None, name=None, clades=None, confidence=None,
                                                                    color=None,
                                                                    width=None,
                                                                    data=None,
                                                                    score=None,
                                                                    transition=None,
                                                                    taxa_set=None):
        # TODO: are these necessary for printing or something? why did I specify this
        # when printing, branch length of None defaults to 0
        # no, looks like ALL the branch lengths are 0/None, it prints
        # every edge with an equal length
        if branch_length is None:
            branch_length = 1.0
        if name is None:
            name = random_string()

        super(NoahClade, self).__init__(branch_length=branch_length,
                                        name=name,
                                        clades=clades,
                                        confidence=confidence,
                                        color=color,
                                        width=width)
        self.data = data
        self.score = score
        # this is a function data_vector => data_vector that governs the transition from the parent of self to self
        self.transition = transition

        # it will try to merge the children's taxa_sets if they are all present
        if (not self.is_terminal()) and taxa_set is None and all(child.taxa_set is not None for child in self.clades):
            taxa_set = np.logical_or.reduce([child.taxa_set for child in self.clades])
        self.taxa_set = taxa_set

    # TODO:
    #def __repr__(self):

    @classmethod
    def convertClade(cls, clade):
        children = [cls.convertClade(child) for child in clade.clades]
        return cls(branch_length=clade.branch_length,
                    name=clade.name,
                    clades=children,
                    confidence=clade.confidence, color=clade.color, width=clade.width)

    @classmethod
    def leaf(cls, i, labels=None, m=None, **kwargs):
        if labels is None:
            # labels and m can't both be None
            labels = ["_"+str(j) for j in range(1, m+1)]
        taxa_set = np.full(len(labels), False)
        taxa_set[i] = True
        return cls(name=labels[i], taxa_set=taxa_set, **kwargs)

    @classmethod
    def randomized(cls, taxa, branch_length=1.0, branch_stdev=None):
        tree = Phylo.BaseTree.Tree.randomized(taxa, branch_length=branch_length, branch_stdev=branch_stdev)
        tree.root = cls.convertClade(tree.root)
        tree.root.reset_taxasets()
        return tree

    def observe(self, labels=None):
        if labels is None:
            # keep this separate because the below breaks when leaves have the same name
            labels = [leaf.name for leaf in self.get_terminals()]
            return np.array([leaf.data for leaf in self.get_terminals()]), labels
        else:
            # this breaks when two leaves have the same name
            leaflabel2data = {leaf.name: leaf.data for leaf in self.get_terminals()}
            return np.array([leaflabel2data[label] for label in labels]), labels

    def gen_subtree_data(self, root_data, transition_generator, **transition_generator_args):
        self.data = root_data
        if not self.is_terminal():
            for child in self:
                if child.transition is None:
                    child.transition = transition_generator(**transition_generator_args)
                new_states = child.transition(self.data)
                child.gen_subtree_data(new_states, transition_generator=transition_generator, **transition_generator_args)

    def reset_taxasets(self, labels=None, m=None):
        if labels is None:
            if m is None:
                labels = [leaf.name for leaf in self.get_terminals()]
            else:
                labels = ["_"+str(i) for i in range(1, m+1)]
        if self.is_terminal():
            self.taxa_set = np.full(len(labels), False)
            self.taxa_set[labels.index(self.name)] = True
        else:
            for child in self:
                child.reset_taxasets(labels=labels)
            self.taxa_set = np.logical_or.reduce([child.taxa_set for child in self.clades])
        return labels

    def find_splits(self, symmetric=True, branch_lengths=False):
        # TODO: reset_taxasets automatically?? otherwise there could be errors
        if branch_lengths:
            splits = dict()
        else:
            splits = set()
        all_term = self.get_terminals()
        for node in self.find_clades():
            # if this assert fails, try calling reset_taxasets to fix it
            assert node.taxa_set is not None
            split = tuple(np.nonzero(node.taxa_set)[0])
            if branch_lengths:
                splits[split] = node.branch_length
            else:
                splits.add(split)
            if symmetric:
                split_other = tuple(np.nonzero(~node.taxa_set)[0])
                if branch_lengths:
                    splits[split_other] = node.branch_length
                else:
                    splits.add(split_other)
        return splits

    def taxaset2ixs(self, taxa_set):
        return tuple(np.nonzero(node.taxa_set)[0])

    def labels2taxaset(self, subset, all_labels=None):
        if all_labels is None:
            all_labels = [leaf.name for leaf in self.get_terminals()]
        subset = set(subset)
        return np.array([(label in subset) for label in all_labels])

    def draw(self):
        Phylo.draw(self)

    def ascii(self):
        Phylo.draw_ascii(self)

    @staticmethod
    def transition_from_transition_matrix(T):
        def transition(data):
            num_classes = T.shape[0]
            num_samples = len(data)
            # this is optimized for num_classes << num_samples
            # since each call to np.random.choice is extremely slow, we limit
            # ourselves to only num_classes calls rather than doing each of num_samples
            # coordinates in a separate call
            redundant_samples = [np.random.choice(a=num_classes, size=num_samples, p=T[i,:]) for i in range(num_classes)]
            mask = np.array([data == i for i in range(num_classes)])
            return (redundant_samples*mask).sum(axis=0)
        transition.matrix = T
        return transition

    @staticmethod
    def gen_discrete_transition(num_classes, proba_bounds=(0.50, 0.95)):
        # TODO: refactor this for jesus sake
        # this is too slow
        #assert isinstance(num_classes, int) and num_classes > 0
        T = np.zeros((num_classes,num_classes))
        for i in range(num_classes):
            diag = np.random.uniform(proba_bounds[0], proba_bounds[1])
            off_diags = np.random.uniform(size=num_classes-1)
            off_diags *= (1. - diag)/off_diags.sum()
            T[i,:] = np.concatenate((off_diags[:i], [diag], off_diags[i:]))
        return NoahClade.transition_from_transition_matrix(T)

    @staticmethod
    def gen_symmetric_transition(num_classes, proba_bounds=(0.50, 0.95)):
        #assert isinstance(num_classes, int) and num_classes > 0
        diag = np.random.uniform(proba_bounds[0], proba_bounds[1])
        off_diag = (1. - diag)/(num_classes - 1)
        T = np.full((num_classes, num_classes), off_diag)
        np.fill_diagonal(T, diag)
        return NoahClade.transition_from_transition_matrix(T)

    # TODO: Jukes-Cantor transition generator
    # change gen_subtree_data to also pass the node itself
    # that way you can look up the branch length
    # from scipy.linalg import expm
    # T = expm(Q*t)

    @staticmethod
    def affine_transition_gaussian(w, b, std):
        def transition(data):
            data = np.array(data)
            return np.random.normal(loc=data*w+b, scale=std, size=data.shape[0])
        return transition

    @staticmethod
    def gen_linear_transition(std_bounds=(0.3, 1)):
        w = np.random.gamma(shape=2., scale=1.)
        b = np.random.uniform(0, 1)
        std = np.random.uniform(*std_bounds)
        return NoahClade.affine_transition_gaussian(w, b, std)

# TODO: for convenience, write a subclass of Tree that calls these methods
# on the root so you don't have to constantly type tree.root.do_something

if __name__ == "__main__":
    tree = NoahClade.randomized(4)
    tree.root.ascii()
    #print(tree.root.find_splits())
    #print(tree.root.find_splits(False))
    #print(tree.root.find_splits(False, False))

    tree.root_with_outgroup("n0")
    tree.root.ascii()
    tree2 = NoahClade.randomized(4)
    tree2.root_with_outgroup("n1")
    tree2.root.ascii()
    labs = tree.root.reset_taxasets()
    tree2.root.reset_taxasets(labs)
    print(equal_topology(tree, tree2))
    print(equal_topology(tree, tree2, True))
    exit()

    b = NoahClade(branch_length=2.0, name="b")
    c = NoahClade(branch_length=3.0, name="c")
    a = NoahClade(branch_length=1.0, clades=[b,c])
    a.gen_subtree_data([0,0,0,0,1,1,1,1,2,2,2,2], num_classes=3, transition_generator=NoahClade.gen_symmetric_transition)
    print(a.data)
    print("b's transition\n", b.transition.matrix)
    print(b.data)
    print("c's transition\n", c.transition.matrix)
    print(c.data)
