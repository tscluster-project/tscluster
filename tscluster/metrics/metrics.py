from __future__ import annotations
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd

from tscluster.preprocessing.utils import broadcast_data

def inertia(
        X: npt.NDArray[np.float64], 
        cluster_centers: npt.NDArray[np.float64]|List, 
        labels:npt.NDArray[np.int64]|pd.DataFrame, 
        ord: int = 2
        ) -> np.float64:
    
    """
    inertia(X, cluster_centers, labels, ord=2)
    
    Calculates the inertia score
    
    This calculates the sum of the distance between all points and their cluster centers across the different time steps. See note.

    Parameters
    -----------
    X : numpy array
        Input time series data. Should be a 3 dimensional array in TNF fromat.
    cluster_centers : numpy array or list
        If numpy array, it is expected to a 3D, use `arr_format` to specify its format.
        A list of k pandas DataFrames. Where k is the number of clusters. The i-th dataframe in the list is a T x F dataframe of the values of the cluster centers of the i-th cluster. 
    labels : numpy array or pandas dataframe
        It is expected to be a N x T 2D array or pandas DataFrame. Where N is the number of entities and T is the number of time steps. The value of the ith row at the t-th column is the label (cluster index) entity i was assigned to at time t.
    ord : int, default : 2
        The distance metric to use. 1 is l1 distance, 2 is l2 distance etc.

    Returns
    --------
    float
        The intertia value.
        
    Notes
    ------
    The inertia is calculated as: 
    
    .. math::
        \sum_{t=1}^{T} \sum_{i=1}^{N} d(X_{ti}, Z_t) 
    Where 
    `T`, `N` are the number of time steps and entities respectively, 
    `d` is a distance function (or metric e.g :math:`L_1`, :math:`L_2` etc), 
    :math:`X_{ti} \in \mathbf{R}^f` is the feature vector of entity `i` at time `t`,
    `f` is the number of features, and 
    :math:`Z_t \in \mathbf{R}^f` is the cluster center :math:`X_{ti}` is assigned to at time `t`

    See Also
    --------
    max_dist : Calculates the maximum distance
    """

    # X, _ = get_inferred_data(X)

    if isinstance(cluster_centers, list):
        cluster_centers = np.array([df.values for df in cluster_centers])

    if isinstance(labels, pd.DataFrame):
        labels = labels.values

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

    """
    Calculate the max_dist score
        
    This calculates the maximum of the distance between all points and their cluster centers across the different time steps. See note.

    Parameters
    -----------
    X : numpy array
        Input time series data. Should be a 3 dimensional array in TNF fromat.
    cluster_centers : numpy array or list
        If numpy array, it is expected to a 3D, use `arr_format` to specify its format.
        A list of k pandas DataFrames. Where k is the number of clusters. The i-th dataframe in the list is a T x F dataframe of the values of the cluster centers of the i-th cluster. 
    labels : numpy array or pandas dataframe
        It is expected to be a N x T 2D array or pandas DataFrame. Where N is the number of entities and T is the number of time steps. The value of the ith row at the t-th column is the label (cluster index) entity i was assigned to at time t.
    ord : int, default : 2
        The distance metric to use. 1 is l1 distance, 2 is l2 distance etc
    arr_format : str, default 'TNF'
        format of the loaded data. 'TNF' means the data dimension is Time x Number of observations x Features
        'NTF' means the data dimension is Number OF  observations x Time x Features
    suffix_sep : str, default '_'
        separator separating the file number from the filename.
    file_reader : str, default 'infer'
        file loader to use. Can be any of np.load, pd.read_csv, pd.read_json, and pd.read_excel. If 'infer', decorator will attempt to infer the file type from the file name 
        and use the approproate loader.
    read_file_args : dict, default empty dictionary.
        parameters to be passed to the data loader.

    Returns
    --------
    float
        The max distance value.
        
    Notes
    ------
    The max_dist is calculated as: 
    
    .. math::
        max(d(X_{ti}, Z_t)) 

    Where 
    `d` is a distance function (or metric e.g :math:`L_1`, :math:`L_2` etc), 
    :math:`X_{ti} \in \mathbf{R}^f` is the feature vector of entity `i` at time `t`,
    `f` is the number of features, and 
    :math:`Z_t \in \mathbf{R}^f` is the cluster center :math:`X_{ti}` is assigned to at time `t`

    See Also
    --------
    interia : Calculates the inertia score
    """

    # X, _ = get_inferred_data(X)

    if isinstance(cluster_centers, list):
        cluster_centers = np.array([df.values for df in cluster_centers])

    if isinstance(labels, pd.DataFrame):
        labels = labels.values

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