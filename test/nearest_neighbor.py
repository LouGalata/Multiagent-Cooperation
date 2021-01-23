import numpy as np
import scipy.spatial
import math

def nearest_neighbors(arr, k):
    k_lst = list(range(k + 2))[2:]  # [2,3]
    neighbors = []
    # construct a kd-tree
    tree = scipy.spatial.cKDTree(arr)
    for row in arr:
        # stack the data so each element is in its own row
        data = row #np.vstack(row)
        # find k nearest neighbors for each element of data, squeezing out the zero result (the first nearest neighbor is always itself)
        dd, ii = tree.query(data, k=k_lst)
        # apply an index filter on data to get the nearest neighbor elements
        neighbors.append(ii)
    return np.stack(neighbors)



N = 10
k = 2
A = np.random.random((N, 2))

print(nearest_neighbors(A, k))
