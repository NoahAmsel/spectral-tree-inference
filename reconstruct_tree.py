from itertools import combinations
import numpy as np
import scipy.spatial.distance

def paralinear_distance(observations, classes=None):
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
    similarity = M / np.sqrt(np.outer(diag,diag))
    return -np.log(np.clip(similarity, a_min=1e-20, a_max=None))

def JC_distance_matrix(observations, classes=None):
    assert classes is None
    if classes is None:
        classes = np.unique(observations)
    k = len(classes)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    inside_log = 1 - hamming_matrix*k/(k-1)
    inside_log = np.clip(inside_log, a_min=1e-16, a_max=None)
    return - ((k-1)/k) * np.log(inside_log)

def correlation_distance_matrix(observations):
    corr = np.corrcoef(observations)
    corr = np.clip(corr, a_min=1e-16, a_max=None)
    return np.log(corr)


if __name__ == "__main__":
    from utils import *
    t = balanced_binary(4)
    all_data = temp_dataset_maker(t, 1000, 0.1)
    observations, _ = charmatrix2array(all_data[0])
    dist = JC_distance_matrix(observations)
    dist[:5,:5]

    dist = paralinear_distance(observations)
    dist[:5,:5]

    dist = JC_similarity_matrix(observations)
    np.log(dist[:5,:5])

    scipy.spatial.distance.pdist(observations, metric='hamming')
