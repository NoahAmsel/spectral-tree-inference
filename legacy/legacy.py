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

if __name__=="__main__":
    inf = estimate_tree_topology_multiclass(obs, labels=["_"+str(i) for i in range(1, obs.shape[0]+1)])


    format_tree(reference_tree, "_3")
    Phylo.draw_ascii(reference_tree)

    format_tree(inf, "_3")
    Phylo.draw_ascii(inf)

    print(reference_tree.clade == inf.clade) # well this clearly doesn't work


    # def tracking_estimate_tree_topology(observations, labels=None, discrete=True, bifurcating=False, scorer=score_split, good_splits):
    #     m, n = observations.shape
    #
    #     """if labels is None:
    #         labels = [str(i) for i in range(m)]
    #     assert len(labels) == m"""
    #
    #     if discrete:
    #         M = similarity_matrix(observations)
    #     else:
    #         # TODO: should this be biased or not?
    #         #M = np.cov(observations, bias=False)
    #         M = np.corrcoef(observations)
    #
    #     # initialize leaf nodes
    #     G = [NoahClade.leaf(i, labels=labels, data=observations[i,:]) for i in range(m)]
    #
    #
    #     available_clades = set(range(len(G)))   # len(G) == m
    #     # initialize Sigma
    #     Sigma = np.full((2*m,2*m), np.nan)  # we should only use entries that we set later; init to nan so they'll throw an error if we do
    #     sv1 = np.full((2*m,2*m), np.nan)
    #     sv2 = np.full((2*m,2*m), np.nan)
    #     for i,j in combinations(available_clades, 2):
    #         A = G[i].taxa_set | G[j].taxa_set
    #         Sigma[i,j] = scorer(A, M)
    #         Sigma[j,i] = Sigma[i,j]    # necessary b/c sets have unstable order, so `combinations' could return either one
    #         sv1[i,j] = first_sv(A, M)
    #         sv1[j,i] = sv1[i,j]
    #         sv2[i,j] = second_sv(A, M)
    #         sv2[j,i] = sv2[i,j]
    #
    #     records = []
    #
    #     # merge
    #     while len(available_clades) > (2 if bifurcating else 3): # this used to be 1
    #         left, right = min(combinations(available_clades, 2), key=lambda pair: Sigma[pair])
    #         G.append(NoahClade(clades=[G[left], G[right]], score=Sigma[left, right]))
    #         # G.append(merge_taxa(G[left], G[right], score=Sigma[left, right]))
    #         new_ix = len(G) - 1
    #         available_clades.remove(left)
    #         available_clades.remove(right)
    #         #
    #         real_possibilities = []
    #         for left, right in combinations(available_clades, 2):
    #             if NoahClade.taxaset2ixs(G[left].taxa_set | G[right].taxa_set) in good_splits:
    #                 real_possibilities.append((left, right))
    #         left_real, right_real = min(real_possibilities, key=lambda pair: Sigma[pair])
    #         records.append({"newclade": G[-1], "score": G[-1].score, "nleft": G[left].taxa_set.sum(), "nright": G[right].taxa_set.sum(), "nA": G[-1].taxa_set.sum(),
    #                         "bestreal": NoahClade(clades=[G[left_real], G[right_real]], score=Sigma[left_real, right_real]), "scorereal": Sigma[left_real, right_real]})
    #         #
    #         for other_ix in available_clades:
    #             A = G[other_ix].taxa_set | G[-1].taxa_set
    #             Sigma[other_ix, new_ix] = scorer(A, M)
    #             Sigma[new_ix, other_ix] = Sigma[other_ix, new_ix]    # necessary b/c sets have unstable order, so `combinations' could return either one
    #             sv1[other_ix, new_ix] = first_sv(A, M)
    #             sv1[new_ix, other_ix] = sv1[other_ix, new_ix]
    #             sv2[other_ix, new_ix] = second_sv(A, M)
    #             sv2[new_ix, other_ix] = sv2[other_ix, new_ix]
    #         available_clades.add(new_ix)
    #
    #     # HEYYYY why does ariel do something special for the last THREE groups? (rather than last two)
    #     # think about what would happen if at the end we have three groups: 1 leaf, 1 leaf, n-2 leaves
    #     # how would we know which leaf to attach, since score would be 0 for both??
    #
    #     # return Phylo.BaseTree.Tree(G[-1])
    #
    #     # for a bifurcating tree we're combining the last two available clades
    #     # for an unrooted one it's the last three because
    #     # if we're making unrooted comparisons it doesn't really matter which order we attach the last three
    #     return Phylo.BaseTree.Tree(NoahClade(clades=[G[i] for i in available_clades])) #, Sigma, sv1, sv2


def R0(R):
    u, s, vh = np.linalg.svd(R, full_matrices=False)
    return np.outer(u[:,0]*s, vh[0,:])

R = np.random.randn(10,30)
R0 = R0(R)
