from itertools import product, combinations

import dendropy
import numpy as np
import scipy.spatial.distance

def sv2(A1, A2, M):
    """Second Singular Value"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    if s.size <= 1:
        return 0    # automatically rank 1, so we say "the second eigenvalue" is 0
    else:
        return s[1] # the second eigenvalue

def max_quartet(A1,A2,M):
    "maximal quartet criterion"
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]
    m_1,m_2 = M_A.shape
    max_w = 0
    for x in combinations(range(m_1), 2):
        for y in combinations(range(m_2),2):
            w = np.abs(M_A[x[0],y[0]]*M_A[x[1],y[1]]-M_A[x[0],y[1]]*M_A[x[1],y[0]])
            if w>max_w:
                max_w = w
    return max_w
    #for (i,j) in  

def sum_sigular_values(A1, A2, M):
    """Normalized Sum of Squared Quartets"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    s = np.linalg.svd(M_A, compute_uv=False)
    return np.linalg.norm(s[1:])

def sum_squared_quartets(A1, A2, M):
    """Normalized Sum of Squared Quartets"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    norm_sq = np.linalg.norm(M_A)**2
    num = norm_sq**2 - np.linalg.norm(M_A.T.dot(M_A))**2
    return num / norm_sq

def average_quartets(A1, A2, M):
    """Normalized Sum of Squared Quartets"""
    A = A1 | A2
    M_A = M[np.ix_(A, ~A)]        # same as M[A,:][:,~A]
    #norm_sq = np.linalg.norm(M_A)**2
    m_1 = np.sum(A1)+np.sum(A2)
    m_2 = len(A1)-m_1
    norm_sq = m_1*(m_1-1)*m_2*(m_2-1)
    num = norm_sq**2 - np.linalg.norm(M_A.T.dot(M_A))**2
    return num / norm_sq

def paralinear_similarity(observations, taxa_metadata=None):
    # build the similarity matrix M -- this is pretty optimized, at least assuming k << n
    # setting classes allows you to ignore missing data. e.g. set class to 1,2,3,4 and it will
    # treat np.nan, -1, and 5 as missing data
    
    # TODO: clean and then delete these two
    #if classes is None:
    #    classes = np.unique(observations)
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
    # assert np.count_nonzero(diag) > 0, "Sequence was so short that some characters never appeared"
    similarity = M / np.sqrt(np.outer(diag,diag))
    return similarity

def paralinear_distance(observations, taxa_metadata=None):
    similarity = paralinear_similarity(observations)
    similarity = np.clip(similarity, a_min=1e-20, a_max=None)
    return -np.log(similarity)

def JC_similarity_matrix(observations, taxa_metadata=None,params=None):
    """Jukes-Cantor Corrected Similarity"""
    classes = np.unique(observations)
    if classes.dtype == np.dtype('<U1'):
        # needed to use hamming distance with string arrays
        vord = np.vectorize(ord)
        observations = vord(observations)
    k = len(classes)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    inside_log = 1 - hamming_matrix*k/(k-1)
    return inside_log**(k-1)
    #return inside_log**(3)

def gamma_func(P,a):
    return (3/4)*a*( (1-(4/3)*P)**(-1/a)-1 )

def JC_gamma_similarity_matrix(observations, taxa_metadata=None,params = None):
    """Jukes-Cantor Corrected Similarity"""
    classes = np.unique(observations)
    if classes.dtype == np.dtype('<U1'):
        # needed to use hamming distance with string arrays
        vord = np.vectorize(ord)
        observations = vord(observations)
    k = len(classes)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(observations, metric='hamming'))
    #inside_log = 1 - hamming_matrix*k/(k-1)
    a = params.alpha
    return np.exp(gamma_func(hamming_matrix,a))

def JC_distance_matrix(observations, taxa_metadata=None):
    """Jukes-Cantor Corrected Distance"""
    inside_log = JC_similarity_matrix(observations, taxa_metadata)
    inside_log = np.clip(inside_log, a_min=1e-16, a_max=None)
    return - np.log(inside_log)

def hamming_dist_missing_values(vals, missing_val):
    classnames, indices = np.unique(vals, return_inverse=True)
    num_arr = indices.reshape(vals.shape)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(num_arr, metric='hamming'))
    missing_array = (vals==missing_val)
    pdist_xor = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_xor(u,v))))
    pdist_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_or(u,v))))
    
    return (hamming_matrix*vals.shape[1] - pdist_xor) / (np.ones_like(hamming_matrix) * vals.shape[1] - pdist_or)

def HKY_similarity_matrix(observations, taxa_metadata, verbose = False):
    m, N = observations.shape

    assert taxa_metadata.alphabet in [dendropy.DNA_STATE_ALPHABET, dendropy.RNA_STATE_ALPHABET]

    if verbose: print("Computing the average base frequency for each pair of sequences...")
    g = {}

    # TODO: ignore other indice too, not just "-"
    not_missing = observations != 4
    not_missing_sum = np.sum(not_missing, axis = 1) # # not missing for each taxon
    not_missing_pair = np.array([a + b for a, b in product(not_missing_sum, repeat = 2)]).reshape((m, m))

    for x, ix in [("A",0), ("C",1), ("G",2), ("T",3)]:
        obs_x = (observations == ix)
        g[x] = np.array([np.sum(np.hstack([a, b])) for a, b in product(obs_x, repeat = 2)]).reshape((m, m))
        g[x] = g[x] / not_missing_pair

    # in DNA_STATE_ALPHABET, 0 => A, 1 => C, 2 => G, 3 => T (or U in RNA)
    # Purines are A and G
    # Pyrimidines are C and T (or U)

    g["purines"] = g["A"] + g["G"]
    g["pyrimidines"] = g["C"] + g["T"]

    # compute transition and transversion proportion
    if verbose: print("Computing transition and transversion proportion for each pair of sequences...")

    P_1 = np.zeros((m,m))
    P_2 = np.zeros((m,m))

    A = hamming_dist_missing_values(observations, missing_val = 4)

    for i in range(m):
        for j in range(i + 1, m):
            neither_missing = np.logical_and(not_missing[i,:], not_missing[j,:])
            a = observations[i,:][neither_missing]
            b = observations[j,:][neither_missing]
            
            # in DNA_STATE_ALPHABET, 0 => A, 1 => C, 2 => G, 3 => T (or U in RNA)
            A_G = np.mean(np.logical_and(a == 0, b == 2) + np.logical_and(a == 2, b == 0))
            P_1[i, j] = P_1[j, i] = A_G
            
            C_T = np.mean(np.logical_and(a == 1, b == 3) + np.logical_and(a == 3, b == 1))
            P_2[i, j] = P_2[j, i] = C_T
            
    Q = A - P_1 - P_2
    
    #print("P", P_1, P_2)
    #print("Q", Q)
    # compute the similarity (formula 7)
    if verbose: print("Computing similarity matrix")
    R = (1 - g["purines"]/(2 * g["A"] * g["G"]) * P_1 - 1 / (2 * g["purines"]) * Q)
    Y = (1 - g["pyrimidines"]/(2 * g["T"] * g["C"]) * P_2 - 1 / (2 * g["pyrimidines"]) * Q)
    T = (1 - 1/(2 * g["purines"] * g["pyrimidines"]) * Q)
    S = np.sign(R) * (np.abs(R))**(8 * g["A"] * g["G"] / g["purines"])
    S *= np.sign(Y) * (np.abs(Y))**(8 * g["T"] * g["C"] / g["pyrimidines"])
    S *= np.sign(T) * (np.abs(T))**(8 * (g["purines"] * g["pyrimidines"] - g["A"] * g["G"] * g["pyrimidines"] / g["purines"] - g["T"] * g["C"] * g["purines"] / g["pyrimidines"]))

    return np.maximum(S,np.zeros_like(S))

"""
def HKY_similarity_matrix(observations, classes=None, verbose = False):
    m, N = observations.shape
    if classes is None:
        classes = np.unique(observations)
    k = len(classes)
    # From Tamura, K., and M. Nei. 1993
    # for each pair of sequences, 
    # 1. estimate the average base frequency for pairs of sequences
    # 2. compute purine transition proportion P1 (A <-> G)
    # 3. compute pyrimidine transition proportion P2 (T <-> C)
    # 3. compute transversion proportion Q (A <-> C, A <-> T, G <-> C, G <-> T)

    if verbose: print("Computing the average base frequency for each pair of sequences...")
    g = {}
    
    not_missing = observations != "-"
    not_missing_sum = np.sum(not_missing, axis = 1) # # not missing for each taxon
    not_missing_pair = np.array([a + b for a, b in product(not_missing_sum, repeat = 2)]).reshape((m, m))
    
    for x in classes:
        obs_x = observations == x 
        g[x] = np.array([np.sum(np.hstack([a, b])) for a, b in product(obs_x, repeat = 2)]).reshape((m, m))
        g[x] = g[x] / not_missing_pair

    
    g["R"] = g["A"] + g["G"]
    g["Y"] = g["T"] + g["C"]
    
    # compute transition and transversion proportion
    if verbose: print("Computing transition and transversion proportion for each pair of sequences...")
        
    P_1 = np.zeros((m,m))
    P_2 = np.zeros((m,m))
    
    A = hamming_dist_missing_values(observations, missing_val = "-")
    
    for i in range(m):
        for j in range(i + 1, m):
            neither_missing = np.logical_and(not_missing[i,:], not_missing[j,:])
            a = observations[i,:][neither_missing]
            b = observations[j,:][neither_missing]
            
            A_G = np.mean(np.logical_and(a == "A", b == "G") + np.logical_and(a == "G", b == "A"))
            P_1[i, j] = P_1[j, i] = A_G
            
            C_T = np.mean(np.logical_and(a == "C", b == "T") + np.logical_and(a == "T", b == "C"))
            P_2[i, j] = P_2[j, i] = C_T
            
    Q = A - P_1 - P_2
    
    #print("P", P_1, P_2)
    #print("Q", Q)
    # compute the similarity (formula 7)
    if verbose: print("Computing similarity matrix")
    R = (1 - g["R"]/(2 * g["A"] * g["G"]) * P_1 - 1 / (2 * g["R"]) * Q)
    Y = (1 - g["Y"]/(2 * g["T"] * g["C"]) * P_2 - 1 / (2 * g["Y"]) * Q)
    T = (1 - 1/(2 * g["R"] * g["Y"]) * Q)
    S = np.sign(R) * (np.abs(R))**(8 * g["A"] * g["G"] / g["R"])
    S *= np.sign(Y) * (np.abs(Y))**(8 * g["T"] * g["C"] / g["Y"])
    S *= np.sign(T) * (np.abs(T))**(8 * (g["R"] * g["Y"] - g["A"] * g["G"] * g["Y"] / g["R"] - g["T"] * g["C"] * g["R"] / g["Y"]))

    return np.maximum(S,np.zeros_like(S))

"""