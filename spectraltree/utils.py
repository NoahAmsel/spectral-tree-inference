from collections import defaultdict
from itertools import combinations
import scipy.spatial.distance
import numpy as np
import dendropy

from dendropy.simulate.treesim import birth_death_tree, pure_kingman_tree, mean_kingman_tree # for convenience

def default_namespace(num_taxa):
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

def lopsided_tree(num_taxa, namespace=None, edge_length=1.):
    # one node splits off at each step
    if namespace is None:
        namespace = default_namespace(num_taxa)
    else:
        assert num_taxa == len(namespace), "The number of leaves must match the size of the given namespace."

    nodes = [leaf(i, namespace, edge_length=edge_length) for i in range(num_taxa)]
    while len(nodes) > 1:
        a = nodes.pop()
        b = nodes.pop()
        nodes.append(merge_children((a,b), edge_length=edge_length))

    return dendropy.Tree(taxon_namespace=namespace, seed_node=nodes[0], is_rooted=False)

def set_edge_lengths(tree, value=None, fun=None, uniform_range=None):
    for e in tree.edges():
        if value is not None:
            assert fun is None and uniform_range is None
            e.length = value
        elif fun is not None:
            assert uniform_range is None
            e.length = fun(e.length)
        else:
            assert uniform_range is not None
            e.length = np.random.uniform(*uniform_range)

def charmatrix2array(charmatrix):
    #charmatrix[taxon].values()
    alphabet = charmatrix.state_alphabets[0]
    return np.array([[state_id.index for state_id in charmatrix[taxon].values()] for taxon in charmatrix]), alphabet

def array2charmatrix(matrix, alphabet=None, namespace=None):
    if namespace is None:
        namespace = default_namespace(matrix.shape[0])
    else:
        assert len(namespace) == matrix.shape[0]
    if alphabet is None:
        alphabet = dendropy.new_standard_state_alphabet(val for val in np.unique(matrix))

    str_matrix = matrix.astype('str') # there must be a faster way...
    return dendropy.StandardCharacterMatrix.from_dict({taxon: str_matrix[i,:] for (i, taxon) in enumerate(namespace)}, default_state_alphabet=alphabet)

def array2distance_matrix(matrix, namespace=None):
    m, m2 = matrix.shape
    assert m == m2, "Distance matrix must be square"
    if namespace is None:
        namespace = default_namespace(m)
    else:
        assert len(namespace) >= m, "Namespace too small for distance matrix"

    dict_form = defaultdict(dict)
    for i in range(m):
        for j in range(m):
            dict_form[namespace[i]][namespace[j]] = matrix[i,j]
    dm = dendropy.calculate.phylogeneticdistance.PhylogeneticDistanceMatrix()
    dm.compile_from_dict(dict_form, namespace)
    return dm

def distance_matrix2array(dm):
    """
    This is patristic distance: adding the branch lengths. If we set branches to
    have different transitions, this won't be the paralinear distance
    """
    taxa = list(dm.taxon_namespace)
    return scipy.spatial.distance.squareform([dm.distance(taxon1, taxon2) for taxon1, taxon2 in combinations(taxa,2)])

def tree2distance_matrix(tree):
    return distance_matrix2array(tree.phylogenetic_distance_matrix())

##########################################################
##               Tree Generation
##########################################################

def balanced_binary(num_taxa, namespace=None, edge_length=1.):
    assert num_taxa == 2**int(np.log2(num_taxa)), "The number of leaves in a balanced binary tree must be a power of 2."
    if namespace is None:
        namespace = default_namespace(num_taxa)
    else:
        assert num_taxa == len(namespace), "The number of leaves must match the size of the given namespace."

    nodes = [leaf(i, namespace, edge_length=edge_length) for i in range(num_taxa)]
    while len(nodes) > 1:
        nodes = [merge_children(nodes[2*i : 2*i+2], edge_length=edge_length) for i in range(len(nodes)//2)]

    return dendropy.Tree(taxon_namespace=namespace, seed_node=nodes[0], is_rooted=False)

def lopsided_tree(num_taxa, namespace=None, edge_length=1.):
    # one node splits off at each step
    if namespace is None:
        namespace = default_namespace(num_taxa)
    else:
        assert num_taxa == len(namespace), "The number of leaves must match the size of the given namespace."

    nodes = [leaf(i, namespace, edge_length=edge_length) for i in range(num_taxa)]
    while len(nodes) > 1:
        a = nodes.pop()
        b = nodes.pop()
        nodes.append(merge_children((a,b), edge_length=edge_length))

    return dendropy.Tree(taxon_namespace=namespace, seed_node=nodes[0], is_rooted=False)

def unrooted_birth_death_tree(birth_rate, death_rate, birth_rate_sd=0.0, death_rate_sd=0.0, **kwargs):
    tree = birth_death_tree(birth_rate, death_rate, birth_rate_sd=birth_rate_sd, death_rate_sd=death_rate_sd, **kwargs)
    tree.is_rooted = False
    return tree

def unrooted_pure_kingman_tree(taxon_namespace, pop_size=1, rng=None):
    tree = pure_kingman_tree(taxon_namespace, pop_size=pop_size, rng=rng)
    tree.is_rooted = False
    return tree

def unrooted_mean_kingman_tree(taxon_namespace, pop_size=1, rng=None):
    tree = mean_kingman_tree(taxon_namespace, pop_size=pop_size, rng=rng)
    tree.is_rooted = False
    return tree

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

if __name__ == "__main__":
    tree = treesim.birth_death_tree(birth_rate=1., death_rate=0., num_total_tips=len(taxa), taxon_namespace=taxa)


    tree = treesim.birth_death_tree(birth_rate=1., death_rate=2., is_retain_extinct_tips=True, num_total_tips=len(taxa), taxon_namespace=taxa)

    tree.minmax_leaf_distance_from_root()

    print(tree.as_python_source())

    print(tree.as_ascii_plot())


    for e in tree.edges():
        print(e.length)

    for n in tree.nodes():
        print(n.edge_length)

    e = tree.edges()[5]
    e.length


    n0 = tree.nodes()[0]
    n0.branch
