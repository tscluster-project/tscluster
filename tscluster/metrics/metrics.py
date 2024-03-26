from __future__ import annotations
from typing import Tuple, Any, List

import numpy as np
import numpy.typing as npt

from tscluster.preprocessing.utils import _get_inferred_data, broadcast_data

def inertia(
        X: npt.NDArray[np.float64], 
        cluster_centers: npt.NDArray[np.float64], 
        labels:npt.NDArray[np.int64], 
        ord: int = 2
        ) -> np.float64:
    
    """Calculate the inertia score"""

    X = _get_inferred_data(None, X)

    cluster_centers, labels = broadcast_data(X.shape[0], cluster_centers, labels)

    running_sum = 0

    for t in range(X.shape[0]):
       for k in range(cluster_centers.shape[1]):
            is_assigned = labels[:, t] == k
            dist = np.linalg.norm(X[t, :, :] - cluster_centers[t, k, :].reshape(-1, X.shape[2]), ord=ord, axis=1)

            running_sum += np.sum(dist * is_assigned)

    return running_sum

def max_dist(
        X: npt.NDArray[np.float64], 
        cluster_centers: npt.NDArray[np.float64], 
        labels: npt.NDArray[np.int64], 
        ord: int = 2) -> np.float64:

    """Calculate the max_dist score"""

    X = _get_inferred_data(None, X)

    cluster_centers, labels = broadcast_data(X.shape[0], cluster_centers, labels)

    running_max = -np.inf

    for t in range(X.shape[0]):
       for k in range(cluster_centers.shape[1]):
            is_assigned = labels[:, t] == k
            dist = np.linalg.norm(X[t, :, :] - cluster_centers[t, k, :].reshape(-1, X.shape[2]), ord=ord, axis=1)

            max_d = np.max(dist * is_assigned)

            if max_d > running_max:
                running_max = max_d

    return running_max