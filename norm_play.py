import numpy as np
from numpy.linalg import norm
from spectraltree import *


b = balanced_binary(16)

R = np.exp(-0.1*tree2distance_matrix(b))
R

A = [0,1]
B = [6,7]
C = [A, B]
rest = [[2,3], [4,5], list(range(8,16)),]
RA = [R[np.ix_(A, r)] for r in rest]
RB = [R[np.ix_(B, r)] for r in rest]

RC = R[np.ix_(sum(C, []), sum(rest, []))]
rest
RC == R[np.ix_([0,1,6,7], [2,3,4,5,8,9,10,11,12,13,14,15])]

########

sum([RC[i1,j1]**2 * RC[i2,j2]**2 for i1 in range(RC.shape[0]) for j1 in range(RC.shape[1]) for i2 in range(RC.shape[0]) for j2 in range(RC.shape[1])])
norm(RC)**4

sum([RC[i1,j1] * RC[i1,j2] * RC[i2,j1] * RC[i2,j2] for i1 in range(RC.shape[0]) for j1 in range(RC.shape[1]) for i2 in range(RC.shape[0]) for j2 in range(RC.shape[1])])
norm(np.dot(RC.T, RC))**2
########

RA[0] == R[np.ix_(A, [2,3])]
RB[1] == R[np.ix_(B, [4,5])]

def num(M):
    return norm(M)**4 - norm(np.dot(M.T, M))**2

def quartet(M, j1, j2, k1, k2):
    return (M[j1, k1] * M[j2, k2] - M[j1, k2] * M[j2, k1])**2

def dets(M):
    n, m = M.shape
    return sum([quartet(RC, j1, j2, k1, k2) for j1 in range(n) for j2 in range(n) for k1 in range(m) for k2 in range(m)])

num(RC)
dets(RC)

# lemma 5.1
assert np.isclose(   num(RC),   (1/2)*dets(RC)    )


def by_part(RA, RB):
    return sum([(norm(RA[j])*norm(RB[k]) - norm(RB[j])*norm(RA[k]))**2 for j in range(len(rest)) for k in range(len(rest))])

by_part(RA, RB)
num(RC)

# lemma 4.6
assert np.isclose(   num(RC),  by_part(RA, RB)    )











# B.5
norm(RC)**2
sum([norm(RA[j])**2 + norm(RB[j])**2 for j in range(len(rest))])

# B.6
norm(RC)**4
sum([(norm(RA[j])**2 + norm(RB[j])**2)*(norm(RA[k])**2 + norm(RB[k])**2) for j in range(len(rest)) for k in range(len(rest))])

# B.7
norm(np.dot(RC.T, RC))**2
sum([norm(RA[j].T.dot(RA[k]) + RB[j].T.dot(RB[k]))**2 for j in range(len(rest)) for k in range(len(rest))])
sum([norm(RA[j].T.dot(RA[k]))**2 + norm(RB[j].T.dot(RB[k]))**2 + 2*norm(RA[k].T.dot(RA[j]).dot(RB[j].T).dot(RB[k])) for j in range(len(rest)) for k in range(len(rest))])

# line 585
norm(np.dot(RC.T, RC))**2
sum([norm(RA[j])**2 * norm(RA[k])**2 + norm(RB[j])**2 * norm(RB[k])**2 + 2*norm(RA[j])*norm(RA[k])*norm(RB[j])*norm(RB[k])   for j in range(len(rest)) for k in range(len(rest))])

norm(RC)**4 - norm(np.dot(RC.T, RC))**2
sum([norm(RA[j])**2 * norm(RB[k])**2 + norm(RA[k])**2 * norm(RB[j])**2 - 2*norm(RA[j])*norm(RA[k])*norm(RB[j])*norm(RB[k])   for j in range(len(rest)) for k in range(len(rest))])
sum([(norm(RA[j]) * norm(RB[k]) - norm(RA[k]) * norm(RB[j]))**2  for j in range(len(rest)) for k in range(len(rest))])
