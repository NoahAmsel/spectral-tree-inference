import numpy as np
import seaborn as sns
import NoahClade
import Bio.Phylo as Phylo
from reconstruct_tree import estimate_tree_topology_Jukes_Cantor
from reconstruct_tree import similarity_matrix, JC_similarity_matrix, normalized_similarity_matrix
from matplotlib.pyplot import get_cmap
from itertools import cycle

import matplotlib.pyplot as plt

def det_Pr(x):
    _, counts = np.unique(x, return_counts=True)
    return (counts/len(x)).prod()

def exact_similarity(ref_tree):
    for node in ref_tree.root.find_clades():
        if node is not ref_tree.root:
            node.log_len = np.log(np.linalg.det(node.transition.matrix))
    leaves = list(ref_tree.get_terminals())
    exact_sim = np.full((len(leaves), len(leaves)), np.nan)
    for i in range(len(leaves)):
        for j in range(len(leaves)):
            left = leaves[i]
            right = leaves[j]
            ancestor = ref_tree.common_ancestor(left, right)
            exact_sim[i,j] = np.log(0.25**4) + np.array([clade.log_len for clade in ancestor.get_path(left)]).sum() + np.array([clade.log_len for clade in ancestor.get_path(right)]).sum()

    return np.exp(exact_sim)

def normalized_exact_similarity(ref_tree):
    return exact_similarity(ref_tree) * (4**4)

from scipy.spatial.distance import pdist
from scipy.linalg import expm
# like sklearn's confusion_matrix, but optimized
def my_confusion_matrix(x, y, classes=None):
    if classes is None:
        classes = np.unique(x)
    xp = np.array([x==cls for cls in classes], dtype='int')  # int32? int16?
    yp = np.array([y==cls for cls in classes], dtype='int')
    return np.einsum("ij,kj",xp,yp)

def factored_similarity(ref_tree):
    n = len(ref_tree.root.data)
    for node in ref_tree.root.find_clades():
        node.P_A = (np.unique(node.data, return_counts=True)[1]/n).prod()
        #P_A = (1/4)**4
        for child in node:
            child.len_empirical = np.linalg.det(my_confusion_matrix(node.data, child.data)/n)/node.P_A

    leaves = list(ref_tree.get_terminals())
    factored_sim = np.full((len(leaves), len(leaves)), np.nan)
    for i in range(len(leaves)):
        for j in range(len(leaves)):
            left = leaves[i]
            right = leaves[j]
            ancestor = ref_tree.common_ancestor(left, right)
            #factored_sim[i,j] = P_root*np.array([node.len_empirical for clade in ancestor.get_path(left)]).prod() * np.array([node.len_empirical for clade in ref_tree.root.get_path(ancestor)]).prod() * np.array([node.len_empirical for clade in ancestor.get_path(right)]).prod()
            factored_sim[i,j] = np.array([clade.len_empirical for clade in ancestor.get_path(left)]).prod() * ancestor.P_A * np.array([clade.len_empirical for clade in ancestor.get_path(right)]).prod()

    diag = [leaf.P_A for leaf in leaves]
    return factored_sim
    #return factored_sim / np.sqrt(np.outer(diag,diag))

def normalized_factored_similarity2(ref_tree):
    n = len(ref_tree.root.data)
    for node in ref_tree.root.find_clades():
        node.P_A = (np.unique(node.data, return_counts=True)[1]/n).prod()
        #P_A = (1/4)**4
        for child in node:
            child.len_empirical = np.linalg.det(my_confusion_matrix(node.data, child.data)/n)/node.P_A

    leaves = list(ref_tree.get_terminals())
    factored_sim = np.full((len(leaves), len(leaves)), np.nan)
    for i in range(len(leaves)):
        for j in range(len(leaves)):
            left = leaves[i]
            right = leaves[j]
            ancestor = ref_tree.common_ancestor(left, right)
            #factored_sim[i,j] = P_root*np.array([node.len_empirical for clade in ancestor.get_path(left)]).prod() * np.array([node.len_empirical for clade in ref_tree.root.get_path(ancestor)]).prod() * np.array([node.len_empirical for clade in ancestor.get_path(right)]).prod()
            factored_sim[i,j] = np.array([clade.len_empirical for clade in ancestor.get_path(left)]).prod() * np.array([clade.len_empirical for clade in ancestor.get_path(right)]).prod()

    diag = [leaf.P_A for leaf in leaves]
    #return factored_sim
    return factored_sim

if __name__ == "__main__":
    depth = 3
    m = 2**depth
    k = 4
    n = 1_000
    ref_tree = NoahClade.NoahClade.complete_binary(depth, proba_bounds=(0.85, 0.95), k=k, n=n)
    ref_tree.root.ascii()
    print(ref_tree)
    obs, labels = ref_tree.root.observe()
    print(labels)
    A = ref_tree.common_ancestor(labels[0], labels[1])
    Pxy_matrix = my_confusion_matrix(obs[0,:], obs[1,:])/n
    Pxy = np.linalg.det(Pxy_matrix)
    Pxy_norm = np.linalg.det(Pxy_matrix/np.sqrt(Pxy_matrix.sum(axis=0).prod()*Pxy_matrix.sum(axis=1).prod()))
    T_xy = Pxy / det_Pr(obs[0,:])
    P_A = det_Pr(A.data)
    P_Ax_matrix = my_confusion_matrix(obs[0,:], A.data)/n
    P_Ax = np.linalg.det(P_Ax_matrix)
    P_Ay_matrix = my_confusion_matrix(obs[1,:], A.data)/n
    P_Ay = np.linalg.det(P_Ay_matrix)
    P_A_inv = np.diag(1/(np.unique(obs[0,:], return_counts=True)[1]/n))
    T_Ax = P_Ax/P_A
    T_Ay = P_Ay/P_A
    T_xA = P_Ax / det_Pr(obs[0,:])

    T_xy
    T_xA*T_Ay

    Pxy
    T_Ax*P_A*T_Ay
    P_Ax*P_Ay/P_A
    P_Ax*P_Ay/(0.25)**4

    # how are these different???
    n*Pxy_matrix
    n*P_Ax_matrix.dot(P_A_inv).dot(P_Ay_matrix.T)


    print("="*30)
    print(exact_similarity(ref_tree)[:5,:5])
    print("+"*20)
    print(similarity_matrix(obs)[:5,:5])
    print("+"*20)
    print(factored_similarity(ref_tree)[:5,:5])

    np.allclose(factored_similarity2(ref_tree), factored_similarity(ref_tree))

    normalized_similarity_matrix(obs) - exact_similarity(ref_tree)

    hamming = pdist(obs[(0,1),:], metric='hamming')[0]
    expected_det = (1 - hamming*k/(k-1))**(k-1)
    t = -np.log(Pxy_norm)/(k*(k-1)) # this is txy

    Pxs = [(np.unique(obs[leaf,:], return_counts=True)[1]/n).prod() for leaf in range(m)]
    np.outer(Pxs,Pxs)
    np.exp(np.log(Pxs).mean())
    (1/4)**4
    one_est = (np.unique(obs, return_counts=True)[1]/(n*m)).prod()

    Q = np.full((k, k), 1)
    np.fill_diagonal(Q, (1-k)*1)
    T = expm(Q*0.2)
    -np.log(np.linalg.det(T))
    #-np.log(np.linalg.det(expm(Q)))
    0.2*(k-1)*k
    T

    ref_tree.find_any("_5").taxa_set

if __name__ == "__main__":
    depth = 6
    m = 2**depth
    n = 100
    iters = 200

    ref_tree = NoahClade.NoahClade.complete_binary(depth, proba_bounds=(0.85, 0.95), k=4, n=1)
    exact_sim = exact_similarity(ref_tree)
    samples = np.full((iters, m, m), np.nan)
    for i in range(iters):
        root_data = np.random.choice(a=4, size=n)
        ref_tree.root.gen_subtree_data(root_data)
        observations, labels = ref_tree.root.observe()
        samples[i,:,:] = JC_similarity_matrix(observations)
        #samples[i,:,:] = similarity_matrix(observations)
    means = np.mean(samples, axis=0)
    vars = np.var(samples, axis=0)

    # corrs[i,j][k,l] = corr (P_ij, P_kl)
    corrs = np.corrcoef((samples-means).reshape((iters, -1)), rowvar=False).reshape((m,m,m,m))
    corrs = np.nan_to_num(corrs)
    for i in range(m):
        for j in range(m):
            # remove the perfect correlations
            corrs[i,j,i,j] = 0
            corrs[i,j,j,i] = 0

    corrs.argmax() #692457
    692457 // (64*64*64)
    (692457 - 2*(64*64*64)) // (64*64)
    (((692457 - 2*(64*64*64)) - 41*(64*64))) // (64)
    (((692457 - 2*(64*64*64)) - 41*(64*64))) - 3*64
    corrs[2,41,3,41]

    (np.abs(corrs) > 0.35).mean()

    sns.distplot(corrs.flatten(), kde=False)

    sns.heatmap(corrs[2,41,:,:], robust=True, center=0, cmap='RdBu')
    sns.distplot(corrs[2,41,:,:].flatten())

    sns.heatmap(corrs[0,1,:,:], robust=True, center=0, cmap='RdBu')
    sns.distplot(corrs[0,1,:,:].flatten())

    sns.heatmap(corrs[0,40,:,:], robust=True, center=0, cmap='RdBu')
    sns.distplot(corrs[0,40,:,:].flatten())

    sns.heatmap(corrs.mean(axis=(2,3)), center=0, cmap='RdBu')
    sns.heatmap(corrs.var(axis=(2,3)), cmap='RdBu')

    corrs[23,32,:,:].mean()

    variability = np.var(corrs, axis=(2,3)).flatten()

    sns.distplot(variability[np.nonzero(variability)])
    # these correlations are almost normal...
    tree_dist = np.full((m,m), np.nan)
    for i in range(m):
        for j in range(m):
            tree_dist[i,j] = abs(i-j)
    sns.regplot(tree_dist.flatten()[np.nonzero(variability)], variability[np.nonzero(variability)], scatter_kws={"alpha":0.1})

    print(corrs[0,1,:,:])
    np.fill_diagonal(corrs, 0)
    sns.heatmap(corrs[::31, ::31], robust=True, center=0, cmap='RdBu')
    sns.distplot(corrs[7, :])

    true_thetas = 3/4 - 3*(4*exact_sim)**(1/3)
    exact_variances = (1/4)**(8) * (16/n) * ((1 - true_thetas*4/3)**4)*true_thetas*(1-true_thetas)

    sns.heatmap(-np.log(exact_sim))
    sns.heatmap(true_thetas)

    np.var(samples[:, 6, 30])
    true_thetas[6,30]
    exact_variances[6,30]

    logratio = np.log2(np.clip(means, a_min=1e-7, a_max=None)/exact_sim)
    sns.heatmap(logratio, robust=True, center=0, cmap='RdBu')

    sns.heatmap(exact_variances, robust=True, center=0, cmap='RdBu')
    sns.heatmap(vars, robust=True, center=0, cmap='RdBu')

if __name__ == "__main__":
    ref_tree = NoahClade.NoahClade.complete_binary(6, proba_bounds=(0.85, 0.95), k=4, n=1)

    root_data = np.random.choice(a=4, size=1000)
    ref_tree.root.gen_subtree_data(root_data)
    observations, labels = ref_tree.root.observe()
    empirical_sim = JC_similarity_matrix(observations)
    #empirical_sim = similarity_matrix(observations)
    exact_sim = exact_similarity(ref_tree)
    np.corrcoef(exact_sim.flatten(), empirical_sim.flatten())[0,1]

    logratio = np.log2(np.clip(empirical_sim, a_min=1e-7, a_max=None)/exact_sim)
    sns.heatmap(logratio, robust=True, center=0, cmap='RdBu')

    #error = empirical_sim - exact_sim
    #sns.heatmap(error, robust=True, center=0, cmap='RdBu')

    accuracy = 100*(empirical_sim - exact_sim)/exact_sim
    print("We underestimate {0}%".format(100*(accuracy<0).astype(int).mean()))
    sns.heatmap(accuracy, robust=True, center=0, cmap='RdBu')
