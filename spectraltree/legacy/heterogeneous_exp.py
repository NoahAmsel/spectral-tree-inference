from sklearn.metrics import confusion_matrix
import numpy as np

def cond_cov(X, EX):
    return np.cov(X - EX)

def from_parent(parent, rate, time, w = 8.0):
    expec = parent*rate*time
    return expec + w*(np.random.random()-0.5) #don't forget the minus .5 or the expectation will be wrong

def from_parent_expec(parent, expec, rate, time, w=1.0):
    return from_parent(parent, rate=rate, time=time, w=w), from_parent(expec, rate=rate, time=time, w=0)

def sample(rate, root=None, ret_E=False):
    if root is None:
        root = 10*np.random.random()
    hA, EhA = from_parent_expec(root, root, rate=rate, time=3)
    x, Ex = from_parent_expec(hA,EhA, rate=rate, time=2)
    y, Ey = from_parent_expec(hA,EhA, rate=rate, time=1)

    hB, EhB = from_parent_expec(root, root, rate=rate, time=8)
    w, Ew = from_parent_expec(hB, EhB, rate=rate, time=30)
    z, Ez = from_parent_expec(hB, EhB, rate=rate, time=1)

    if ret_E:
        return [x, y, w, z], [Ex, Ey, Ew, Ez]
    else:
        return [x, y, w, z]

def rank1(M_A):
    s = np.linalg.svd(M_A, compute_uv=False)
    #return 1 - (s[0]/np.sum(s))
    #return s[1]
    print("svs ",s[:3])
    return s[1:].sum()

if __name__ == "__main__":

    # use a different rate for each row
    obs = np.array([sample(np.random.choice(range(10))) for _ in range(100)])
    cov = np.cov(obs, rowvar=False)
    print(rank1(cov[:2,:][:,2:]))
    print(rank1(cov[(0,2),:][:,(1,3)]))
    print(rank1(cov[(0,3),:][:,(1,2)]))
    # so it really does still work!

    # this shows that under homogeneity (rate is always 2) the covariance btwn nodes x,y given hA is 0 (when A is btwn x and y)
    obsexp2 = np.array([sample(2, ret_E=True) for _ in range(1000_000)])
    obs2, exp2  = obsexp2[:,0,:], obsexp2[:,1,:]
    print(np.cov(obs2-exp2, rowvar=False))
    # and the covariance between the conditional expectations is rank 1
    covexp2 = np.cov(exp2, rowvar=False)
    print(rank1(covexp2[:2,:][:,2:]))
    cov2 = np.cov(obs2, rowvar=False)
    print(rank1(cov2[:2,:][:,2:]))
    print(rank1(cov2[(0,2),:][:,(1,3)]))
    print(rank1(cov2[(0,3),:][:,(1,2)]))

    # ... but the above does NOT hold under heterogeneity
    obsexp3 = np.array([sample(np.random.choice(range(100)), ret_E=True) for _ in range(1000_000)])
    obs3, exp3  = obsexp3[:,0,:], obsexp3[:,1,:]
    cov3 = np.cov(obs3-exp3, rowvar=False)
    print(cov3)
    # tho the cov between conditional expectations is still rank 1
    covexp3 = np.cov(exp3, rowvar=False)
    print(rank1(covexp3[:2,:][:,2:]))
    cov3 = np.cov(obs3, rowvar=False)
    print(rank1(cov3[:2,:][:,2:]))
    print(rank1(cov3[(0,2),:][:,(1,3)]))
    print(rank1(cov3[(0,3),:][:,(1,2)]))
