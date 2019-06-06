    """# build the similarity matrix M
    # this is a HUGE time suck
    # figure a way to calculate M once then read it from disk
    classes = np.unique(observations) #only used for my_confusion_matrix
    M = np.full((m,m), np.nan)  # we should never use diagonals; init to nan so they'll throw an error if we do
    for i in range(m):
        for j in range(i):  # should be i to skip diagonals, i+1 to include them
            # TODO: allow for missing data (nans) when building M
            Pij = confusion_matrix(observations[i,:], observations[j,:]) / n
            #Pij = my_confusion_matrix(observations[i,:], observations[j,:], classes=classes) / n
            M[i,j] = np.linalg.det(Pij)
            M[j,i] = M[i,j]"""

# like sklearn's confusion_matrix, but optimized
def my_confusion_matrix(x, y, classes):
    xp = np.array([x==cls for cls in classes], dtype='int')  # int32? int16?
    yp = np.array([y==cls for cls in classes], dtype='int')
    return np.einsum("ij,kj",xp,yp)


def gen_transition_matrix(num_classes, proba_bounds=(0.05, 0.95)):
    # TODO: refactor this for jesus sake
    # this is too slow
    T = np.zeros((num_classes,num_classes))
    for i in range(num_classes):
        diag = np.random.uniform(proba_bounds[0], proba_bounds[1])
        off_diags = np.random.uniform(size=num_classes-1)
        off_diags *= (1. - diag)/off_diags.sum()
        T[i,:] = np.concatenate((off_diags[:i], [diag], off_diags[i:]))
    return T

def gen_simple_transition_matrix(num_classes, proba_bounds=(0.05, 0.95)):
    diag = np.random.uniform(proba_bounds[0], proba_bounds[1])
    off_diag = (1. - diag)/(num_classes - 1)
    T = np.full((num_classes, num_classes), off_diag)
    np.fill_diagonal(T, diag)
    return T

def gen_multiclass_tree(num_terminals, num_samples, num_classes, proba_bounds=(0.5, 0.95), simple_transitions=True):
    def gen_subtree_data(clade, root_data):
        clade.data = root_data
        if not clade.is_terminal():
            # TODO: save transition_matrix as a member of the clade so we can rerun without generating new matrices?
            transition_matrix = gen_simple_transition_matrix(num_classes, proba_bounds=proba_bounds) if simple_transitions else gen_transition_matrix(num_classes, proba_bounds=proba_bounds)
            for child in clade:
                new_states = [np.random.choice(a=num_classes, p=transition_matrix[old_state,:]) for old_state in clade.data]
                gen_subtree_data(child, new_states)

    tree = Phylo.BaseTree.Tree.randomized(num_terminals)
    tree.root.name = "root"
    gen_subtree_data(tree.root, np.random.choice(a=num_classes, size=num_samples))
    return tree

def negate_bitstring(bs):
    #https://biopython.org/DIST/docs/api/Bio.Phylo.Consensus-pysrc.html#_BitString.__and__
    selfint = literal_eval('0b' + bs)
    resultint = 2**(len(bs)) - 1 - selfint
    return _BitString(bin(resultint)[2:].zfill(len(bs)))


    def bitstring_topology(self, symmetric=True):
        bitstring_topo = _bitstring_topology(self)
        if symmetric:
            flipped = {negate_bitstring(bitstr): length for bitstr, length in bitstring_topo.items()}
            bitstring_topo.update(flipped)
        return bitstring_topo

def format_tree(tree, outgroup=None):
    if outgroup is not None:
        tree.root_with_outgroup(outgroup)
    for c in tree.find_elements():
        c.branch_length = 1.0
    tree.ladderize()

def run_trial(reference_tree_path=None, observation_matrix_path=None, comparison_tree_path=None, format='newick', m=None, n=None, k=None, proba_bounds=None):
    if reference_tree_path is not None:
        reference_tree = Phylo.read(reference_tree_path, format)
    else:
        # this will generate fake data too, but we might ignore it if observation_matrix_path is given
        reference_tree = gen_multiclass_tree(m, n, k, proba_bounds=proba_bounds)

    if observation_matrix_path is not None:
        assert reference_tree_path is not None # otherwise how could external observation matrix match the tree
        obs = np.genfromtxt(observation_matrix_path, delimiter=",")
        assert obs.shape[0] == reference_tree.count_terminals() # one row per leaf
    else:
        obs = np.array([leaf.data for leaf in reference_tree.get_terminals()])

    if comparison_tree_path is not None:
        assert observation_matrix_path is not None
        matlab_inf = Phylo.read(comparison_tree_path, format)

        format_tree(matlab_inf, "_3")
        Phylo.draw_ascii(matlab_inf)

    inf = estimate_tree_topology_multiclass(obs, labels=["_"+str(i) for i in range(1, obs.shape[0]+1)])


    format_tree(reference_tree, "_3")
    Phylo.draw_ascii(reference_tree)

    format_tree(inf, "_3")
    Phylo.draw_ascii(inf)

    print(reference_tree.clade == inf.clade) # well this clearly doesn't work
