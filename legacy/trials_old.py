# %%
import numpy as np
import Bio.Phylo as Phylo
import cProfile
import NoahClade

from reconstruct_tree import estimate_tree_topology_multiclass
from baselines import NJ_JC, NJ_hamming

def grow_tree_recover(m, n, k, proba_bounds, verbose=True):
    reference_tree = random_discrete_tree(m, n, k, proba_bounds=proba_bounds)
    if verbose:
        reference_tree.root.ascii()
    observations, labels = reference_tree.root.observe()
    inferred_tree = estimate_tree_topology_multiclass(observations, labels=labels)
    if verbose:
        Phylo.draw_ascii(inferred_tree)
    NoahClade.tree_Fscore(inferred_tree, reference_tree)
    print(NoahClade.equal_topology(inferred_tree, reference_tree))

def load_tree_recover(reference_tree, n, k=None, proba_bounds=(0.50, 0.95), verbose=True, format='newick'):
    # if tree is a file, load the tree it contains
    # if tree is a tree object, generate data, possibly using preexisting transition matrices
    if isinstance(reference_tree, str):
        reference_tree = Phylo.read(reference_tree, format)
        reference_tree.root = NoahClade.NoahClade.convertClade(reference_tree.root)
    if hasattr(reference_tree.root, 'transition') and hasattr(reference_tree.root.transition, 'matrix'):
        k = reference_tree.root.transition.shape[1]
    root_data = np.random.choice(a=k, size=n)
    transition_maker = NoahClade.NoahClade.gen_symmetric_transition
    reference_tree.root.gen_subtree_data(root_data, transition_maker, num_classes=k, proba_bounds=proba_bounds)

    if verbose:
        reference_tree.root.ascii()
    observations, labels = reference_tree.root.observe()
    reference_tree.root.reset_taxasets(labels=labels)
    inferred_tree = estimate_tree_topology_multiclass(observations, labels=labels)
    inferred_tree.root.reset_taxasets(labels=labels)
    if verbose:
        Phylo.draw_ascii(inferred_tree)
    NoahClade.tree_Fscore(inferred_tree, reference_tree)
    NoahClade.equal_topology(inferred_tree, reference_tree)

def even_branch_lengths(tree):
    for node in tree.find_elements():
        node.branch_length = 1

def load_observations_recover(reference_tree_path=None, observation_matrix_path=None, comparison_tree_path=None, verbose=False, format='newick', method=estimate_tree_topology_multiclass):
    reference_tree = Phylo.read(reference_tree_path, format)
    reference_tree.root = NoahClade.NoahClade.convertClade(reference_tree.root)
    if verbose:
        print("="*40)
        print("REFERENCE")
        even_branch_lengths(reference_tree)
        reference_tree.root.ascii()

    observations = np.genfromtxt(observation_matrix_path, delimiter=",")
    assert observations.shape[0] == reference_tree.count_terminals() # one row per leaf
    labels = ["_"+str(i) if i < 10 else str(i) for i in range(1, observations.shape[0]+1)]
    inferred_tree = method(observations, labels=labels)
    reference_tree.root.reset_taxasets(labels=labels)
    inferred_tree.root.reset_taxasets(labels=labels)
    if verbose:
        print("INFERRED")
        even_branch_lengths(inferred_tree)
        inferred_tree.root.ascii()

    print("=== Compare to gold standard ===")
    F1_gold, _, _, _ = NoahClade.tree_Fscore(inferred_tree, reference_tree)
    print(NoahClade.equal_topology(inferred_tree, reference_tree))
    print(F1_gold)

    if comparison_tree_path is not None:
        external_inferred = Phylo.read(comparison_tree_path, format)
        external_inferred.root = NoahClade.NoahClade.convertClade(external_inferred.root)
        external_inferred.root.reset_taxasets(labels=labels)
        print("=== Compare to Matlab version ===")
        F1_matlab, _, _, _ = NoahClade.tree_Fscore(inferred_tree, external_inferred)
        print(NoahClade.equal_topology(inferred_tree, external_inferred))
        print(F1_matlab)
        if verbose:
            print("MATLAB")
            even_branch_lengths(external_inferred)
            external_inferred.root.ascii()

def main():
    #cProfile.run('generate_and_recover(m=64, n=10**4, k=4, proba_bounds=(0.8, 0.95))')
    grow_tree_recover(m=64, n=10**4, k=4, proba_bounds=(0.80, 0.95))
    #cProfile.run('generate_and_recover(m=200, n=4000, k=4, proba_bounds=(0.8, 0.95))')
    #generate_and_recover(m=200, n=4000, k=4, proba_bounds=(0.8, 0.95))

def main2():
    path = '/Users/noah/Dropbox/reconstructing_trees/code/output_trees/phy_mc_ref.tree'
    load_tree_recover(reference_tree=path, n=10**4, k=4, proba_bounds=(0.80, 0.95))

def main3():
    load_observations_recover(reference_tree_path='/Users/noah/Dropbox/reconstructing_trees/code/output_trees/phy_mc_ref.tree',
                observation_matrix_path='/Users/noah/Dropbox/reconstructing_trees/code/observations.txt',
                comparison_tree_path='/Users/noah/Dropbox/reconstructing_trees/code/output_trees/spec_mc.tree', verbose=True)
# %%

if __name__ == "__main__":
    main()
    main2()
    main3()

    from scipy.spatial.distance import squareform, pdist
    from baselines import distance_correction

    load_observations_recover(reference_tree_path="./NJ_ref.tree",
                                observation_matrix_path="obs.csv",
                                comparison_tree_path="NJ_test.tree", method=NJ_JC, verbose=True)

    dists = np.genfromtxt("dists.csv", delimiter=",")
    matlab_dm = squareform(dists)

    obs = np.genfromtxt("obs.csv", delimiter=",")
    classes = np.unique(obs)
    k = len(classes)
    hamming = squareform(pdist(obs, metric='hamming'))
    dm = distance_correction(hamming, k)

    dm[dm > 5]


    pdist(obs[], metric='hamming')

    pdist(obs[(0,5),:], metric='hamming')

    dm[0,:6]
    matlab_dm[0,:6]
    np.max(dm)
    np.max(matlab_dm)
    np.max(np.abs(matlab_dm - dm))
    np.argmax(dm - matlab_dm)
    dm[0,4]
    matlab_dm[0,4]

    np.allclose(dm, matlab_dm, atol=0.01)

    load_observations_recover(reference_tree_path="./NJ_ref.tree",
                                observation_matrix_path="obs.csv",
                                comparison_tree_path="NJ_test.tree", method=NJ_hamming, verbose=True)
