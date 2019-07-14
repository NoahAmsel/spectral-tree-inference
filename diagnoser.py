import numpy as np
import Bio.Phylo as Phylo
import NoahClade
from reconstruct_tree import estimate_tree_topology_Jukes_Cantor
from reconstruct_tree import similarity_matrix, JC_similarity_matrix
from matplotlib.pyplot import get_cmap
from itertools import cycle

def resolve_group(tree, node_group, symmetric=True):
    # node_group contains labels
    good = set(node_group)
    leaf_names = set(leaf.name for leaf in tree.get_terminals())
    assert all(node in leaf_names for node in node_group)
    queue = [tree.root]
    all_yes_clades = []
    all_no_clades = []
    for clade in queue:
        clade_descendents = [leaf.name for leaf in clade.get_terminals()]
        if all(descendent in good for descendent in clade_descendents):
            all_yes_clades.append(clade)
        elif all(descendent not in good for descendent in clade_descendents):
            all_no_clades.append(clade)
        else:
            queue += clade.clades

    if symmetric:
        return min(all_yes_clades, all_no_clades, key=len)
    else:
        return all_yes_clades

def diagnose(inferred, reference, labels):
    m = len(labels)

    term_names_inferred = set(term.name for term in inferred.find_clades(terminal=True))
    term_names_reference = set(term.name for term in reference.find_clades(terminal=True))
    assert term_names_inferred == term_names_reference
    # these are set objects
    splits_inferred = inferred.root.find_splits(symmetric=False, branch_lengths=False)
    splits_inferred_sym = inferred.root.find_splits(symmetric=True, branch_lengths=False)
    splits_reference = reference.root.find_splits(symmetric=False, branch_lengths=False)
    splits_reference_sym = reference.root.find_splits(symmetric=True, branch_lengths=False)

    good = Phylo.BaseTree.BranchColor(0,0,0)
    bad = Phylo.BaseTree.BranchColor(255,0,0)
    missed = cycle(get_cmap('Dark2').colors)

    for clade in inferred.root.find_elements(order='preorder'):
        clade.width = 2
        if clade.my_taxaset2ixs(False) in splits_reference_sym:
            pass #clade.color = good
        else:
            #clade.color = bad
            clade.branch_length *= 3
            clade.name = "***"+clade.name

    missed_splits = splits_reference - inferred.root.find_splits(symmetric=True, branch_lengths=False)
    pure_clades_per_split = [resolve_group(inferred_tree, [labels[leaf_ix] for leaf_ix in missed_split]) for missed_split in missed_splits]
    for pure_clades in sorted(pure_clades_per_split, key=len, reverse=True):
        color = tuple(int(255*x) for x in next(missed))
        for pure_clade in pure_clades:
            pure_clade.color = color

    print("Missed splits ({0}):".format(len(missed_splits)))
    for x in sorted([tuple(labels[leaf][-2:] for leaf in spl) for spl in missed_splits], key=lambda x:len(x)):
        print("\t", len(x), x)

    bad_splits = splits_inferred - splits_reference_sym
    print("Bad splits ({0}):".format(len(bad_splits)))
    for x in sorted([tuple(labels[leaf][-2:] for leaf in spl) for spl in bad_splits], key=lambda x:len(x)):
        print("\t", len(x), x)

    RF = len(splits_inferred_sym ^ splits_reference_sym) # Robinson-Foulds is size of symmetric difference
    both = len(splits_inferred_sym & splits_reference_sym)
    precision = float(both)/len(splits_inferred_sym)
    recall = float(both)/len(splits_reference_sym)
    F1 = 2*(precision*recall)/(precision+recall)
    return F1, precision, recall, RF

def score_stable_rank(A,M):
    M_A = M[np.ix_(A, ~A)]
    s = np.linalg.svd(M_A, compute_uv=False)
    s_sq = s**2
    return s_sq.sum()/s_sq[0] #frobenius norm sq / operator norm sq

if __name__ == "__main__":
    ref_tree = NoahClade.random_discrete_tree(64, 1000, 4, proba_bounds=(0.75, 0.95))
    #ref_tree = NoahClade.NoahClade.complete_binary(6, n=500, k=4, proba_bounds=(0.75, 0.95))
    observations, labels = ref_tree.root.observe()
    inferred_tree = estimate_tree_topology_Jukes_Cantor(observations, labels=labels)
    F1, _, _, RF, = diagnose(inferred_tree, ref_tree, labels)
    print("F1: {0:.3f}\nRF: {1}".format(F1, RF))
    Phylo.draw(inferred_tree, label_func=lambda x: None, branch_labels=lambda x: "***" if x.name[:3]=="***" else None)
    #Phylo.draw(ref_tree, label_func=lambda x: x.name[-2:] if x.is_terminal() else None)

    inferred_tree2 = estimate_tree_topology_Jukes_Cantor(observations, labels=labels, scorer=score_stable_rank)
    F12, _, _, RF2, = diagnose(inferred_tree2, ref_tree, labels)
    print("F1: {0:.3f}\nRF: {1}".format(F12, RF2))
    Phylo.draw(inferred_tree2, label_func=lambda x: None, branch_labels=lambda x: "***" if x.name[:3]=="***" else None)
