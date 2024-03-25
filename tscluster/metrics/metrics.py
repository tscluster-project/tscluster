import numpy as np

from tscluster.preprocessing.utils import infer_data

def _broadcast_data(cluster_centers, labels, T):
    if cluster_centers.ndim == 2:
        cluster_centers = np.array([z for z in range(T)])

    if labels.ndim == 1:
        labels = np.array([l for l in range(T)]).T

    return cluster_centers, labels

@infer_data
def _get_inferred_data(_, X):
    return X

def inertia(X, cluster_centers, labels, ord=2):

    X = _get_inferred_data(None, X)

    cluster_centers, labels = _broadcast_data(cluster_centers, labels, X.shape[0])

    running_sum = 0

    for t in range(X.shape[0]):
       for k in range(cluster_centers.shape[1]):
            is_assigned = labels[:, t] == k
            dist = np.linalg.norm(X[t, :, :] - cluster_centers[t, k, :].reshape(-1, X.shape[2]), ord=ord, axis=1)

            running_sum += np.sum(dist * is_assigned)

    return running_sum

def max_dist(X, cluster_centers, labels, ord=2):

    X = _get_inferred_data(None, X)

    cluster_centers, labels = _broadcast_data(cluster_centers, labels, X.shape[0])

    running_max = -np.inf

    for t in range(X.shape[0]):
       for k in range(cluster_centers.shape[1]):
            is_assigned = labels[:, t] == k
            dist = np.linalg.norm(X[t, :, :] - cluster_centers[t, k, :].reshape(-1, X.shape[2]), ord=ord, axis=1)

            max_d = np.max(dist * is_assigned)

            if max_d > running_max:
                running_max = max_d

    return running_max