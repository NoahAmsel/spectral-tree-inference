import numpy as np

import Bio.Phylo as Phylo
import Bio
import Bio.Phylo.TreeConstruction
import scipy.spatial.distance

import pandas as pd

def distance_correction(hamming_ratio, k):
    # hamming_ratio must be between 0 and 1

    #return -np.log(1 - hamming_ratio*k/(k-1))/k
    inside_log = 1 - hamming_ratio*k/(k-1)
    return -np.log(np.clip(inside_log, a_min=1e-16, a_max=None))/k
    # /\ this is vectorized, plus maybe its good to be warned
    #if ratio >= (k-1)/k:
    #    return np.inf
    #else:
    #    return -np.log(1 - ratio*k/(k-1))/k

def neighbor_joining(observations, labels=None):
    if labels is None:
        m = observations.shape[0]
        labels = ["_"+str(j) for j in range(1, m+1)]
    align = Bio.Align.MultipleSeqAlignment([Bio.SeqRecord.SeqRecord(Bio.Seq.Seq("".join(str(char) for char in observations[i,:])), id=labels[i]) for i in range(observations.shape[0])])
    calculator = Bio.Phylo.TreeConstruction.DistanceCalculator('identity')
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor(calculator, 'nj')
    return constructor.build_tree(align)

def NJ(observations, labels=None, classes=None):
    """
    Applies Neighbor Joining to discrete data. Generating the distance matrix myself
    """
    m = observations.shape[0]
    n = observations.shape[1]
    if labels is None:
        labels = ["_"+str(j) for j in range(1, m+1)]
    if classes is None:
        classes = np.unique(observations)
        w = np.ones(n)
    else:
        assert(False)
        # w = # set to 0 whenever one of the elements isn't in the classes?
    hamming = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    dm = Bio.Phylo.TreeConstruction._DistanceMatrix(labels, [list(hamming[i,:i+1]) for i in range(0,m)])
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor()
    return constructor.nj(dm)

def NJ_JC(observations, labels=None, classes=None):
    """
    Applies Neighbor Joining to discrete data but does a distance correction based on the Jukes Cantor model.
    """
    m = observations.shape[0]
    n = observations.shape[1]
    if labels is None:
        labels = ["_"+str(j) for j in range(1, m+1)]
    if classes is None:
        classes = np.unique(observations)
        w = np.ones(n)
    else:
        assert(False)
        # w =  set to 0 whenever one of the elements isn't in the classes?
    # TODO: should k just be another argument?
    k = len(classes)

    hamming = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    corrected = distance_correction(hamming, k)
    dm = Bio.Phylo.TreeConstruction._DistanceMatrix(labels, [list(corrected[i,:i+1]) for i in range(0,m)])
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor()
    return constructor.nj(dm)

def NJ_continuous(observations, labels=None):
    m = observations.shape[0]
    if labels is None:
        labels = ["_"+str(j) for j in range(1, m+1)]
    distances = -np.log(np.corrcoef(observations)**2)
    dm = Bio.Phylo.TreeConstruction._DistanceMatrix(labels, [list(distances[i,:i+1]) for i in range(0,m)])
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor()
    return constructor.nj(dm)

# %%
if __name__ == "__main__":
    from trials import random_JC_tree
    import NoahClade

    ref_tree = random_JC_tree(m=200, n=30_000, k=2)
    ref_tree.root.ascii()
    observations, labels = ref_tree.root.observe()
    T2 = NJ(observations, labels)
    T3 = NJ_JC(observations, labels)

    hamming = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    corrected = distance_correction(hamming, 4)
    pd.DataFrame(corrected)
    hamming.max()
    corrected.max()
    distance_correction(0.74, k=4)
    1 - 0.9666666666666667*4/3
    -np.log(1e-16)


    # how come the Jukes-Cantor correction does worse even under a Jukes Cantor model

    ref_tree.root = NoahClade.NoahClade.convertClade(ref_tree.root)
    ref_tree.root.reset_taxasets(labels)
    T2.root = NoahClade.NoahClade.convertClade(T2.root)
    T2.root.reset_taxasets(labels)
    T3.root = NoahClade.NoahClade.convertClade(T3.root)
    T3.root.reset_taxasets(labels)
    NoahClade.tree_Fscore(T2, T3)
    NoahClade.tree_Fscore(T2, ref_tree)
    NoahClade.tree_Fscore(T3, ref_tree)


    T3.root.ascii()
