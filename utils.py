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

# TODO: change depth arg to #leaves arg
def balanced_binary(depth, namespace=None):
    num_taxa = 2**depth
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

    nodes = [leaf(i, namespace, edge_length=1) for i in range(num_taxa)]
    while len(nodes) > 1:
        a = nodes.pop()
        b = nodes.pop()
        nodes.append(merge_children((a,b)))

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

if __name__ == "__main__":
    t = balanced_binary(4)
    t.print_plot()

    trees = dendropy.TreeList([t])
    trees
    trees.extend([t, t])
    trees.taxon_namespace is trees[0].taxon_namespace
    trees.taxon_namespace is trees[-1].taxon_namespace

    trees.taxon_namespace[2]

    all_data = temp_dataset_maker(trees, 1000, 1)
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
