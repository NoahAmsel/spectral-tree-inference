import dendropy
import numpy as np
from dendropy.interop import seqgen

def new_default_namespace(num_taxa):
    return dendropy.TaxonNamespace([str(i) for i in range(num_taxa)])

def leaf(i, namespace, **kwargs):
    # for now, `i` will specify the taxon but in the future can make it more flexible
    assert 'taxon' not in kwargs
    kwargs['taxon'] = namespace[i]
    node = dendropy.Node(**kwargs)
    node.taxa_set = np.full(len(namespace), False)
    node.taxa_set[i] = True
    return node

def merge_children(children, **kwargs):
    node = dendropy.Node(**kwargs)
    for child in children:
        node.add_child(child)
    if all(hasattr(child,'taxa_set') for child in children):
        node.taxa_set = np.logical_or.reduce(tuple(child.taxa_set for child in children))
    return node

def balanced_binary(depth, namespace=None):
    num_taxa = 2**depth
    if namespace is None:
        namespace = new_default_namespace(num_taxa)
    else:
        assert num_taxa == len(namespace)

    nodes = [leaf(i, namespace, edge_length=1) for i in range(num_taxa)]
    while len(nodes) > 1:
        nodes = [merge_children(nodes[2*i : 2*i+2], edge_length=1) for i in range(len(nodes)//2)]

    return dendropy.Tree(taxon_namespace=namespace, seed_node=nodes[0])

def temp_dataset_maker(tree_list, seq_len, scaler):
    s = seqgen.SeqGen()
    #s.model = "GTR"
    s.seq_len = seq_len
    #s.num_partitions = 1 # what does this even do?
    s.scale_branch_lens = scaler
    d0 = s.generate(tree_list)
    #print(s._compose_arguments())
    return d0.char_matrices

def charmatrix2array(charmatrix):
    #charmatrix[taxon].values()
    alphabet = charmatrix.state_alphabets[0]
    return np.array([[state_id.index for state_id in charmatrix[taxon].values()]for taxon in charmatrix]), alphabet

def array2charmatrix(matrix, alphabet):
    pass

if __name__ == "__main__":
    t = balanced_binary(4)
    t.print_plot()

    trees = dendropy.TreeList()
    trees.extend([t, t])
    trees.taxon_namespace is trees[0].taxon_namespace
    trees.taxon_namespace is trees[-1].taxon_namespace

    trees.taxon_namespace[2]

    all_data = temp_dataset_maker(trees, 1000)
    t0_data = all_data[0]
    observations, _ = charmatrix2array(t0_data)

    """model = dendropy.model.discrete.Jc69() #
    model = dendropy.model.discrete.DiscreteCharacterEvolver()
    mat = dendropy.model.discrete.simulate_discrete_chars(100, t, model)
    hh = mat[0]
    hh.symbols_as_string()"""

    pdm = dendropy.PhylogeneticDistanceMatrix(t)
    print(pdm)
    tt = pdm.as_data_table()



    def similarity_matrix(observations, classes=None):
        # build the similarity matrix M -- this is pretty optimized, at least assuming k << n
        # setting classes allows you to ignore missing data. e.g. set class to 1,2,3,4 and it will
        # treat np.nan, -1, and 5 as missing data
        if classes is None:
            classes = np.unique(observations)
        observations_one_hot = np.array([observations==cls for cls in classes], dtype='int')
        # this command makes an (m x m) array of (k x k) confusion matrices
        # where m = number of leaves and k = number of classes
        confusion_matrices = np.einsum("jik,mlk->iljm", observations_one_hot, observations_one_hot)
        # valid_observations properly accounts for missing data I think
        valid_observations = confusion_matrices.sum(axis=(2,3), keepdims=True)
        M = np.linalg.det(confusion_matrices/valid_observations)     # same as `confusion_matrices/n' when there aren't null entries
        #M = np.linalg.det(confusion_matrices/sqrt(n))
        return M


    similarity_matrix(observations)
