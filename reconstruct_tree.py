from itertools import combinations
import numpy as np
import scipy.spatial.distance

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

    diag = M.diagonal()
    return M / np.sqrt(np.outer(diag,diag))

def JC_similarity_matrix(observations, classes=None):
    # TODO: use classes argument to make this robust to missing data
    if classes is None:
        classes = np.unique(observations)
    else:
        assert False
    k = len(classes)

    normalized_hamming_distance = scipy.spatial.distance.pdist(observations, metric='hamming')
    # under J-C model, even independent sequences have a maximum hamming distance of (k-1)/k
    # in which case determinant is 0
    normalized_hamming_distance = np.clip(normalized_hamming_distance, a_min=None, a_max=(k-1)/k)
    expected_dets = ((1/k)**k)*(1 - normalized_hamming_distance*k/(k-1))**(k-1)
    #expected_dets = (1/k - normalized_hamming_distance/(k-1))**(k-1) / k  # this is the same
    dm = scipy.spatial.distance.squareform(expected_dets)
    # under jukes-cantor model, base frequencies are equal at all nodes
    # so for diag elements, take determinant of (1/k)* I_k
    np.fill_diagonal(dm, (1/k)**k)
    return dm

def linear_similarity_matrix(observations):
    return np.corrcoef(observations)**2
