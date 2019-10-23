import dendropy

pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
        src=open("pythonidae.mle.weighted.pdm.csv"),
        delimiter=",")
nj_tree = pdm.nj_tree()
print(nj_tree.as_string("newick"))

# %%

import numpy as np
import pandas as pd

import Bio.Phylo as Phylo
import Bio
import Bio.Phylo.TreeConstruction
import scipy.spatial.distance
import NoahClade

from NoahClade import random_JC_tree, random_discrete_tree
from reconstruct_tree import similarity_matrix
from scipy.special import xlogy
from sklearn.metrics import confusion_matrix
from itertools import product

def neighbor_joining(observations, labels=None):
    if labels is None:
        m = observations.shape[0]
        labels = ["_"+str(j) for j in range(1, m+1)]
    align = Bio.Align.MultipleSeqAlignment([Bio.SeqRecord.SeqRecord(Bio.Seq.Seq("".join(str(char) for char in observations[i,:])), id=labels[i]) for i in range(observations.shape[0])])
    calculator = Bio.Phylo.TreeConstruction.DistanceCalculator('identity')
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor(calculator, 'nj')
    return constructor.build_tree(align)

def NJ(distance_matrix, labels=None):
    """
    Applies Neighbor Joining to discrete data. Generating the distance matrix myself
    """
    m = distance_matrix.shape[0]
    if labels is None:
        labels = ["_"+str(j) for j in range(1, m+1)]
    dm = Bio.Phylo.TreeConstruction._DistanceMatrix(labels, [list(distance_matrix[i,:i+1]) for i in range(0,m)])
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor()
    inferred_tree = constructor.nj(dm)
    inferred_tree.root = NoahClade.NoahClade.convertClade(inferred_tree.root)
    inferred_tree.root.reset_taxasets(labels)
    return inferred_tree

def NJ_hamming(observations, labels=None):
    dm = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    return NJ(dm, labels=labels)

# d = -3/4 log(1-p * 4/3)
def distance_correction(hamming_ratio, k):
    # hamming_ratio must be between 0 and 1

    #return -np.log(1 - hamming_ratio*k/(k-1))/k
    inside_log = 1 - hamming_ratio*k/(k-1)
    return -(k-1)*np.log(np.clip(inside_log, a_min=1e-16, a_max=None))/k
    # /\ this is vectorized, plus maybe its good to be warned
    #if ratio >= (k-1)/k:
    #    return np.inf
    #else:
    #    return -np.log(1 - ratio*k/(k-1))/k

def NJ_JC(observations, labels=None):
    classes = np.unique(observations)
    k = len(classes)
    hamming = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    dm = distance_correction(hamming, k)
    return NJ(dm, labels=labels)

def NJ_logdet(observations, labels=None, classes=None):
    similarity = similarity_matrix(observations, classes=classes)
    dm = -np.log(np.clip(similarity, a_min=1e-20, a_max=None))
    return NJ(dm, labels=labels)

def NJ_continuous(observations, labels=None):
    m = observations.shape[0]
    if labels is None:
        labels = ["_"+str(j) for j in range(1, m+1)]
    distances = -np.log(np.corrcoef(observations)**2)
    dm = Bio.Phylo.TreeConstruction._DistanceMatrix(labels, [list(distances[i,:i+1]) for i in range(0,m)])
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor()
    inferred_tree = constructor.nj(dm)
    inferred_tree.root = NoahClade.NoahClade.convertClade(inferred_tree.root)
    inferred_tree.root.reset_taxasets(labels)
    return inferred_tree

# %%
from sklearn.metrics import confusion_matrix, mutual_info_score
from scipy.stats import entropy

if __name__ == "__main__":
    t = 0.8
    q = 0.8
    r = 0.6
    Tca = np.array([[1-q, q],[q, 1-q]])
    Tcb = np.array([[1-r, r],[r, 1-r]])
    pc = np.array([t, 1-t])
    pab = Tca.dot(np.diag(pc)).dot(Tcb)
    pa_alt = Tca.dot(pc)
    pb_alt = Tcb.dot(pc)
    pa = pab.sum(axis=1, keepdims=True)
    pb = pab.sum(axis=0, keepdims=True)
    print(pa, pa_alt)
    print(pb, pb_alt)
    print(pab)
    pa.dot(pb)

    pac = Tca.dot(np.diag(pc))
    pbc = Tcb.dot(np.diag(pc))

    D(pab)
    D(pac)+D(pbc)

    mut_info(pab)
    mut_info(pac)
    mut_info(pbc)

def joint_prob(X1, X2):
    p12 = confusion_matrix(X1, X2)
    return p12 / p12.sum()

def xlgx(x):
    return xlogy(x, x)/np.log(2) # convert log to log_2

def joint_ent(pxy):
    return -(xlgx(pxy)).sum()


from sklearn.metrics import mutual_info_score

def mut_info(pxy):
    px = pxy.sum(axis=1, keepdims=True)
    px = px / px.sum()
    py = pxy.sum(axis=0, keepdims=True)
    py = py / py.sum()
    denom = px.dot(py)
    # return (pxy*np.nan_to_num(np.log2(pxy/denom))).sum()
    # for numerical reasons this is cleaner:
    return (xlgx(pxy/denom)*denom).sum()

def D(pxy):
    Hxy = joint_ent(pxy)
    Ixy = mut_info(pxy)
    d = Hxy - Ixy
    D = d / Hxy if d>0 else 0
    return d

def mut_info_pdist(observations):
    m, n = observations.shape
    return [[D(joint_prob(observations[i,:], observations[j,:])) for j in range(m)] for i in range(m)]




# %%
def test_correction(m, n, k, proba_bounds):
    score = 0
    NNN = 50
    for _ in range(NNN):
        ref_tree = random_JC_tree(m=m, n=n, k=k, proba_bounds=proba_bounds)
        observations, labels = ref_tree.root.observe()
        T2 = NJ_hamming(observations, labels)
        T3 = NJ_JC(observations, labels)
        ref_tree.root = NoahClade.NoahClade.convertClade(ref_tree.root)
        ref_tree.root.reset_taxasets(labels)
        T2.root = NoahClade.NoahClade.convertClade(T2.root)
        T2.root.reset_taxasets(labels)
        T3.root = NoahClade.NoahClade.convertClade(T3.root)
        T3.root.reset_taxasets(labels)
        plain_F1, _, _, _ = NoahClade.tree_Fscore(T2, ref_tree)
        jc_F1, _, _, _ = NoahClade.tree_Fscore(T3, ref_tree)
        if jc_F1 > plain_F1:
            score += 1
        print("plain {0:.3f}%\t\tjc {1:.3f}%".format(100*plain_F1, 100*jc_F1))
    return score/NNN

if __name__ == "__main__":
    test_correction(m=64, n=10_000, k=4, proba_bounds=(0.85, 0.95))


# %%
if __name__ == "__main__":

    #ref_tree = random_JC_tree(m=32, n=40_000, k=20, proba_bounds=(0.75, 0.95))
    ref_tree = random_JC_tree(m=32, n=20_000, k=2, proba_bounds=(0.75, 0.95))
    ref_tree.root.ascii()
    observations, labels = ref_tree.root.observe()
    T1 = neighbor_joining(observations, labels)
    T2 = NJ_hamming(observations, labels)
    T3 = NJ_JC(observations, labels)

    hamming = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    corrected = distance_correction(hamming, 4)
    #pd.DataFrame(corrected)
    hamming.max()
    corrected.max()
    #distance_correction(0.74, k=4)
    #1 - 0.9666666666666667*4/3
    #-np.log(1e-16)

    # how come the Jukes-Cantor correction does worse even under a Jukes Cantor model

    ref_tree.root = NoahClade.NoahClade.convertClade(ref_tree.root)
    _ = ref_tree.root.reset_taxasets(labels)
    T1.root = NoahClade.NoahClade.convertClade(T1.root)
    _ = T1.root.reset_taxasets(labels)
    T2.root = NoahClade.NoahClade.convertClade(T2.root)
    _ = T2.root.reset_taxasets(labels)
    T3.root = NoahClade.NoahClade.convertClade(T3.root)
    _ = T3.root.reset_taxasets(labels)
    NoahClade.tree_Fscore(T1, T2)
    NoahClade.tree_Fscore(T2, T3)
    NoahClade.tree_Fscore(T2, ref_tree)
    NoahClade.tree_Fscore(T3, ref_tree)

    T3.root.ascii()
