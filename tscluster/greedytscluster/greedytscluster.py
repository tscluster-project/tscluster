from __future__ import annotations

from tscluster.base import TSCluster
from tscluster.interface import TSClusterInterface
from sklearn.cluster import kmeans_plusplus
from sklearn.utils.validation import _check_sample_weight
from scipy.spatial.distance import cdist
from tscluster.metrics import inertia, max_dist

from typing import TextIO, Tuple, List
import sys
import numpy as np
import numpy.typing as npt
import copy

class GreedyTSCluster(TSCluster, TSClusterInterface):
    """
    (Under development)
    Class for Maxima Minimation (MM) algorithm (a.k.a. greedy algorithm) for time-series clustering.
    Throughout this doc and code, ‘z’ refers to cluster centers, while ‘c’ to label assignment. This creates an GreedyTSCluster object.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to generate.
    scheme: {'z0c0', 'z0c1', 'z1c0', 'z1c1'}, default='z1c0'
        The scheme to use for tsclustering. Could be one of:
            - 'z0c0' means fixed center, fixed assignment
            - 'z0c1' means fixed center, changing assignment
            - 'z1c0' means changing center, fixed assignment
            - 'z1c1' means changing center, changing assignment
        Scheme needs to be a dynamic label assignment scheme (either 'z1c1' or 'z0c1') when using constrained cluster change (either with `n_allow_assignment_change`)
    n_allow_assignment_change : int or None, default=None
        Penalty added to changing assignments over time for 'c1' schemes.
    random_state : int or None, default=None
        Random seed for reproducibility.
    initialization : str, default='kmeans++'
        Method to initialize cluster centers. Must be one of {'kmeans++', 'random'}.
    """
    def __init__(self,     
                n_clusters: int, 
                scheme: str = 'z1c0', 
                *,
                n_allow_assignment_change: None|int = None,  
                random_state: None|int = None,
                initialization: str = 'kmeans++'
                ) -> None:
        
        self.n_clusters = n_clusters
        self.n_allow_assignment_change = n_allow_assignment_change
        self.random_state = np.random.RandomState(random_state)
        self.initialization_choices = {'kmeans++', 'random'}
        if initialization not in  self.initialization_choices:
            raise ValueError(f'Invalid value for initialization Expected any of {self.initialization_choices}, but got {initialization}')
        else:
            self.initialization = initialization

        self.solver_schemes = {'z0c0', 'z1c0', 'z0c1', 'z1c1'}
        if scheme not in self.solver_schemes:
            raise ValueError(f"Invalid value for scheme. Expected any of {self.solver_schemes}, but got '{scheme}'")
        else:
            self.scheme = scheme
        
        self._label_dict_ = None
        self.verbose = True 
        # the cluster centers
        self.Zs = None
        # a binary matrix for the cluster labels
        self._Cs = None
        # a counter to keep track of the number of times the same cluster assignment is repeated
        self._converage_count = 0
        # dimensions of the input data
        self.T_, self.N_, self.F_ = None, None, None

    def _TSCplusplus(self) -> None:
        """
        Initialize cluster centers using the k-means++ strategy.

        For 'z1' schemes, centers are dynamic and initialized using flattened temporal features.
        For 'z0' schemes, centers are static and shared across time.
        """
        if self.T_ is None or self.N_ is None or self.F_ is None:
            self.T_, self.N_, self.F_ = self.X_.shape
        self.Zs = np.zeros((self.T_, self.n_clusters, self.F_))
        if self.scheme == 'z1c0' or self.scheme == 'z1c1':
            Xt = self.X_.transpose(1, 0, 2)
            x_squared_norms = cdist(Xt.reshape((self.N_, -1)), 
                                    np.zeros((1, self.T_ * self.F_)), 
                                    metric="sqeuclidean")
            sample_weight = _check_sample_weight(None, Xt, dtype=Xt.dtype)
            self.Zs = kmeans_plusplus(Xt.reshape((self.N_, -1)), self.n_clusters, 
                                                    x_squared_norms=x_squared_norms, 
                                                    sample_weight=sample_weight, 
                                                    random_state=self.random_state)[0].reshape((-1, self.T_, self.F_))
            self.Zs = self.Zs.transpose(1, 0, 2)
        elif self.scheme == 'z0c0' or self.scheme == 'z0c1':
            # if the cluster centers are static arcorss time, we used the mean of the time series
            Xt = self.X_.mean(axis=0)
            x_squared_norms = cdist(Xt, np.zeros((1, self.F_)), metric="sqeuclidean")
            sample_weight = _check_sample_weight(None, Xt, dtype=Xt.dtype)
            
            z = kmeans_plusplus(Xt.reshape((self.N_, -1)), self.n_clusters, 
                                x_squared_norms=x_squared_norms, 
                                sample_weight=sample_weight, 
                                random_state=self.random_state)[0].reshape((-1, self.F_))
            for k in range(self.n_clusters):
                for d in range(self.F_):
                    self.Zs[:, k , d] = z[k,d] * np.ones(self.T_)
        else:
            raise ValueError(f"Invalid value for scheme. Expected any of {self.solver_schemes}, but got '{self.scheme}'")
    
    def _random_init(self) -> None:
        """
        Randomly initialize the cluster assignments and zero-initialize cluster centers.

        Each sample is randomly assigned to one of the clusters across all time steps.
        """
        if self.T_ is None or self.N_ is None or self.F_ is None:
            self.T_, self.N_, self.F_ = self.X_.shape
        cluster_index = self.random_state.choice(range(self.n_clusters), size=self.N_)
        self._Cs = np.zeros(shape=(self.T_, self.N_, self.n_clusters), dtype=int)
        for i, cluster_assignment in enumerate(cluster_index):
            # assign a static cluster assignment across time for better convergence
            self._Cs[:, i, cluster_assignment] = 1
        self.Zs = np.zeros((self.T_, self.n_clusters, self.F_))
        
    
    def _update_cluster_assignment(self) -> tuple[bool, float, float, int]:
        """
        Update the cluster assignments based on distance to centers.

        For 'c1' schemes with penalty, continuity of assignment is encouraged over time.

        Returns
        -------
        tuple of (bool, float, float, int)
            - Converged flag
            - Sum of intra-cluster distances
            - Maximum intra-cluster distance
            - Number of samples with assignment changes
        """
        Cs_new = np.zeros(shape=(self.T_, self.N_ , self.n_clusters), dtype=int)
        for n in range(self.N_):
            if self.scheme == 'z1c0' or self.scheme == 'z0c0':
                distance = np.array([np.linalg.norm(self.X_[:, n, :] - self.Zs[:, k, :]) 
                                     for k in range(self.n_clusters)])
                Cs_new[:, n, distance.argmin()] = 1
            elif self.scheme == 'z1c1' or self.scheme == 'z0c1':
                if self.n_allow_assignment_change is None:
                    for t in range(self.T_):
                        distance = np.array([np.linalg.norm(self.X_[t, n, :] - self.Zs[t, k, :]) 
                                             for k in range(self.n_clusters)])
                        Cs_new[t, n, distance.argmin()] = 1
                else:
                    # using the smallest global distance to initialize the cluster label 
                    # to prevent first timestamp dominates the cluster labels
                    global_distance = np.array([np.linalg.norm(self.X_[:, n, :] - self.Zs[:, k, :]) 
                                                for k in range(self.n_clusters)])
                    for t in range(0, self.T_):
                        if t == 0:
                            # add penalty if it is different from the cluster with smallest global distance
                            # this way ensured if the first cluster is too far away from the rest time steps,
                            # it can still assign it to the closest cluster
                            distance = np.array([np.linalg.norm(self.X_[t, n, :] - self.Zs[t, k, :]) 
                                                if k == global_distance.argmin() 
                                                else np.linalg.norm(self.X_[0, n, :] - self.Zs[0, k, :]) + 
                                                self.n_allow_assignment_change for k in range(self.n_clusters)])
                        else:
                            # add penalty if it is different from the cluster label previous time step
                            distance = np.array([np.linalg.norm(self.X_[t, n, :] - self.Zs[t, k, :]) + 
                                                self.n_allow_assignment_change * (1 - Cs_new[t-1, n, k]) 
                                                for k in range(self.n_clusters)])
                        Cs_new[t, n, distance.argmin()] = 1
            else:
                raise ValueError(f"Invalid value for scheme. Expected any of {self.solver_schemes}, but got '{self.scheme}'")
        
        # calculate the statistics of the cluster assignment
        sum_of_distance = inertia(self.X_, self.cluster_centers_, np.argmax(Cs_new, axis=2).transpose())
        max_distance = max_dist(self.X_, self.cluster_centers_, np.argmax(Cs_new, axis=2).transpose())
        number_of_change = (np.sum(self._Cs != Cs_new, axis=(2, 0)) > 0).sum()

        if np.array_equal(self._Cs, Cs_new) and self._converage_count>5: 
        # To prevent local minima, stop if the same cluster assignment is repeated for more than 5 times
            return True, sum_of_distance, max_distance, number_of_change
        elif np.array_equal(self._Cs, Cs_new):
            self._converage_count += 1
            return False, sum_of_distance, max_distance, number_of_change
        else:
            self._Cs = copy.deepcopy(Cs_new)
            self._converage_count = 0
            return False, sum_of_distance, max_distance, number_of_change

    def _update_cluster_centers(self) -> None:
        """
        Update cluster centers using current cluster assignments.

        For 'z0' schemes, cluster centers are shared across time.
        For 'z1' schemes, centers are updated independently for each time step.
        """
        for k in range(self.n_clusters):
            for d in range(self.F_):
                if self.scheme  == 'z0c0':
                    z = np.mean([self._Cs[t,:, k].dot(self.X_[t, :, d].T)/(self._Cs[t,:, k].sum() + 1e-10) 
                                  for t in range(self.T_)]) # prevent division by zero
                    self.Zs[:, k, d] = [z]*self.T_
                elif self.scheme  == 'z0c1':
                    if self._Cs[:, :, k].sum() != 0: # if no data point is assigned to the cluster, keep the old value
                        self.Zs[:, k, d] = np.sum([self._Cs[t,:, k].dot(self.X_[t, :, d].T)
                                                    for t in range(self.T_)]) / (self._Cs[:, :, k].sum())
                elif self.scheme in ['z1c0','z1c1']:
                    for t in range(self.T_):
                        if self._Cs[t, :, k].sum() != 0: # if no data point is assigned to the cluster, keep the old value
                            self.Zs[t, k, d] = np.sum(self._Cs[t,:, k].dot(self.X_[t, :, d].T)) / self._Cs[t,:, k].sum()
    def fit(
            self, 
            X: npt.NDArray[np.float64], 
            label_dict: dict|None = None, 
            verbose: bool = True, 
            print_to: TextIO = sys.stdout,
            max_iter: int = 1000,
            **kwargs
            ) -> "GreedyTSCluster":
        """
        Fit the temporal clustering model using greedy optimization.

        Parameters
        ----------
        X : np.ndarray of shape (T, N, F)
            The input time series data, where T is the number of time steps,
            N is the number of samples, and F is the number of features.
        label_dict : dict, optional
            Optional dictionary of axis labels used for interpretability.
        verbose : bool, default=True
            If True, print progress and diagnostic information during fitting.
        print_to : TextIO, default=sys.stdout
            File-like stream to output verbose logs.
        max_iter : int, default=1000
            Maximum number of optimization iterations.

        Returns
        -------
        self : GreedyTSCluster
            The fitted model instance.
        """
        self._label_dict_ = label_dict
        self.verbose = verbose
        self._converage_count = 0
        self.T_, self.N_, self.F_ = X.shape
        self.X_ = copy.deepcopy(X)
        if self.n_allow_assignment_change is not None:
            # normalize the data to prevent the penalty from being too large
            self.n_allow_assignment_change *= np.max(np.abs(X[1:]-X[:-1]))/self.n_clusters

        if self.initialization == 'kmeans++':
            self._TSCplusplus()
            _ = self._update_cluster_assignment()
            self._converage_count = 0
        elif self.initialization == 'random':
            self._random_init()
        elif self.initialization == 'custom':
            if 'Zs' in kwargs and 'Cs' in kwargs:
                self.Zs = kwargs['Zs']
                self._Cs = kwargs['Cs']
            else:
                raise ValueError("Custom initialization is selected, but no cluster centers or labels are provided. Please provide 'Zs' and 'Cs' as keyword arguments.")

        if self.verbose:
            print(f'Initialization with {self.initialization}, Sum of Distance: {inertia(X, self.cluster_centers_, self.labels_):.4f}, Max Distance: {max_dist(X, self.cluster_centers_, self.labels_):.4f}',
                        file=print_to)
        for i in range(max_iter):
            # fix the assignment and update the cluster centers
            self._update_cluster_centers()
            converage, sum_of_distance, max_of_distance, number_of_change = self._update_cluster_assignment()
            # fix the cluster centers and update the assignment
            if converage:
                if self.verbose:
                    print(f'Converged at iteration {i}, Sum of distance: {sum_of_distance:.4f}, Max distance: {max_of_distance:.4f}', file=print_to)

                return self
            else:
                if self.verbose and i % 5 == 0: 
                    print(f'Iteration {i}, Sum of distance: {sum_of_distance:.4f}, Max distance: {max_of_distance:.4f}, Number of change: {number_of_change}', file=print_to)
        
        if self.verbose:
            print(f'Maximum iteration reached, Sum of distance: {sum_of_distance:.4f}, Max distance: {max_of_distance:.4f}', file=print_to)
        return self
    
    @property
    def cluster_centers_(self):
        """
        Cluster centers learned by the model.

        Returns
        -------
        np.ndarray of shape (T, K, F)
            The cluster centroids for each cluster k and time t.
        """
        if self.Zs is None:
            raise ValueError("The model is not fitted yet. Please fit the model first.")
        else:
            return self.Zs

    @property
    def labels_(self):
        """
        Cluster labels for each sample at each time step.

        Returns
        -------
        np.ndarray of shape (N, T)
            The cluster assignment for each sample and time.
        """
        if self._Cs is None:
            raise ValueError("The model is not fitted yet. Please fit the model first.")
        else:
            return np.argmax(self._Cs, axis=2).transpose()
    
    @property
    def fitted_data_shape_(self) -> Tuple[int, int, int]:
        """
        Shape of the data the model was fit on.

        Returns
        -------
        tuple of int
            Tuple (T, N, F) corresponding to time, samples, and features.
        """
        return self.T_, self.N_, self.F_