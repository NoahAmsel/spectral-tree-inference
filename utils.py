from collections import defaultdict
import numpy as np
import dendropy
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

# TODO: change depth arg to #leaves arg
def balanced_binary(num_taxa, namespace=None):
    assert num_taxa == 2**int(np.log2(num_taxa))
    if namespace is None:
        namespace = new_default_namespace(num_taxa)
    else:
        assert num_taxa == len(namespace)

    nodes = [leaf(i, namespace, edge_length=1.) for i in range(num_taxa)]
    while len(nodes) > 1:
        nodes = [merge_children(nodes[2*i : 2*i+2], edge_length=1.) for i in range(len(nodes)//2)]

    return dendropy.Tree(taxon_namespace=namespace, seed_node=nodes[0])

# TODO:
def lopsided_tree(num_taxa, namespace=None):
    # one node splits off at each step
    if namespace is None:
        namespace = new_default_namespace(num_taxa)
    else:
        assert num_taxa == len(namespace)

    nodes = [leaf(i, namespace, edge_length=1.) for i in range(num_taxa)]
    while len(nodes) > 1:
        a = nodes.pop()
        b = nodes.pop()
        nodes.append(merge_children((a,b), edge_length=1.0))

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
    return np.array([[state_id.index for state_id in charmatrix[taxon].values()] for taxon in charmatrix]), alphabet

def array2charmatrix(matrix, alphabet):
    pass

def array2distance_matrix(matrix, namespace=None):
    m, m2 = matrix.shape
    assert m == m2, "Distance matrix must be square"
    if namespace is None:
        namespace = new_default_namespace(m)
    else:
        assert len(namespace) >= m, "Namespace too small for distance matrix"

    dict_form = defaultdict(dict)
    for i in range(m):
        for j in range(m):
            dict_form[namespace[i]][namespace[j]] = matrix[i,j]
    dm = dendropy.calculate.phylogeneticdistance.PhylogeneticDistanceMatrix()
    dm.compile_from_dict(dict_form, namespace)
    return dm

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
