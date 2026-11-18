import numpy as np

from numpy import linalg as LA
from sklearn.cluster import KMeans

data_matrix = np.array([[2,2],
                               [8,2]])

eig_val, eig_vect = LA.eig(data_matrix)

# print(eig_val)
# print(eig_vect)

A = np.array([
                    [0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                    ])


D = np.diag(A.sum(axis=1))
print(D)

L = D-A
print(L)

print("Adjacency Matrix:")
print(A)

D = np.diag(A.sum(axis=1))

L = D-A

vals, vecs = np.linalg.eig(L)

vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]

kmeans = KMeans(n_clusters=4)
kmeans.fit(vecs[:,1:4])
colors = kmeans.labels_

print("Clusters:", colors)

