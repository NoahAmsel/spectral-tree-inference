import dendropy
import numpy
from dendropy.interop import seqgen

def new_default_namespace(num_taxa):
    return dendropy.TaxonNamespace([str(i) for i in range(num_taxa)])

def balanced_binary(depth, namespace=None):
    num_taxa = 2**depth
    if namespace is None:
        namespace = new_default_namespace(num_taxa)
    else:
        assert num_taxa == len(namespace)

    nodes = [dendropy.Node(taxon=namespace[i], edge_length=1) for i in range(num_taxa)]
    while len(nodes) > 1:
        next_level = []
        for i in range(len(nodes)//2):
            node = dendropy.Node(edge_length=1)
            node.add_child(nodes[2*i])
            node.add_child(nodes[2*i+1])
            next_level.append(node)
        nodes = next_level

    return dendropy.Tree(taxon_namespace=namespace, seed_node = nodes[0])

def temp_dataset_maker(tree_list, seq_len):
    s = seqgen.SeqGen()
    #s.model = "GTR"
    s.seq_len = seq_len
    #s.num_partitions = 1 # what does this even do?
    d0 = s.generate(tree_list)
    #print(s._compose_arguments())
    return d0.char_matrices


if __name__ == "__main__":
    t = balanced_binary(4)
    t.print_plot()
    trees = dendropy.TreeList()
    trees.extend([t, t])
    trees.taxon_namespace is trees[0].taxon_namespace
    trees.taxon_namespace is trees[-1].taxon_namespace

    all_data = temp_dataset_maker(trees, 25)
    t0_data = all_data[0]
    type(t0_data)

    """model = dendropy.model.discrete.Jc69() #
    model = dendropy.model.discrete.DiscreteCharacterEvolver()
    mat = dendropy.model.discrete.simulate_discrete_chars(100, t, model)
    hh = mat[0]
    hh.symbols_as_string()"""
