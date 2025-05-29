from __future__ import annotations
from typing import List, Any, Tuple


import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from tscluster.interface import TSClusterInterface
from tscluster.base import TSCluster
from tscluster.preprocessing.utils import tnf_to_ntf, infer_data

class TSGlobalKmeans(KMeans, TSCluster, TSClusterInterface):
    # def __init__(self, *args, **kwargs):
    #     self._labels_ = None
    #     self._cluster_centers_ = None
    #     super().__init__(*args, **kwargs)

    """
    Applies sklearn's K-Means clustering to the version of a data  reshaped from a 3D array of shape T x N x F to a 2D array of shape (TxN) x F

    Read more in the Sklearn's user guide. The follow parameters are from sklearn's K-Means constructor.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.

        For an example of how to choose an optimal value for n_clusters refer to
        sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py.

    init : {'k-means++', 'random'}, callable or array-like of shape (n_clusters, n_features), default='k-means++'
        Method for initialization:

    'k-means++' : selects initial cluster centroids using sampling based on an empirical probability distribution of the points' contribution to the overall inertia. This technique speeds up convergence. The algorithm implemented is "greedy k-means++". It differs from the vanilla k-means++ by making several trials at each sampling step and choosing the best centroid among them.

    'random': choose n_clusters observations (rows) at random from data for the initial centroids.

    If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

    If a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization.

        For an example of how to use the different init strategy, see the example
        entitled sphx_glr_auto_examples_cluster_plot_kmeans_digits.py.

    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid seeds. The final results is the best output of n_init consecutive runs in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see kmeans_sparse_high_dim).

        When n_init='auto', the number of runs depends on the value of init: 10 if using init='random' or init is a callable; 1 if using init='k-means++' or init is an array-like.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
        See Glossary <random_state>.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True (default), then the original data is not modified. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean. Note that if the original data is not C-contiguous, a copy will be made even if copy_x is False. If the original data is sparse, but not in CSR format, a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is "lloyd". The "elkan" variation can be more efficient on some datasets with well-defined clusters, by using the triangle inequality. However it's more memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).
    
    Attributes
    ----------
    cluster_centers_
    fitted_data_shape_
    labels_
    label_dict_
    n_changes_
    """

    def fit(
        self, 
        X: npt.NDArray[np.float64],
        label_dict: dict|None = None, 
        **kwargs
        ) -> 'TSGlobalKmeans':
        """
        Method for fitting the model on the data.

        Parameters
        ----------
        X : numpy array
            Input time series data. Should be a 3 dimensional array in TNF fromat.
        label_dict : dict, default=None
            A dictionary of the labels of X. Keys should be 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
                - 'T' is a list of names/labels of each time step used as index of each dataframe during fit. Default is range(0, T). Where T is the number of time steps in the fitted data
                - 'N' is a list of names/labels of each entity used as index of the dataframe. Default is range(0, N). Where N is the number of entities/observations in the fitted data 
                - 'F' is a list of names/labels of each feature used as column of each dataframe. Default is range(0, F). Where F is the number of features in the fitted data 
            data_loader function from tscluster.preprocessing.utils can help in getting label_dict of a data. 
        **kwargs keyword arguments to be passed to skleanr's kmeans fit method

        Returns
        -------
        self: 
            The fitted TSGlobalKmeans object. 
        """

        self._label_dict_ = label_dict

        self._labels_ = None
        self._cluster_centers_ = None

        self.Xt = tnf_to_ntf(X)

        self.N, self.T, self.F = self.Xt.shape

        self.Xt = np.vstack(self.Xt)

        super().fit(self.Xt, **kwargs) 

        return self

    @property
    def cluster_centers_(self) -> npt.NDArray[np.float64]: 
        return self._cluster_centers_
    
    @cluster_centers_.setter
    def cluster_centers_(self, new_value: npt.NDArray[np.float64]) -> None:
        self._cluster_centers_ = new_value

    @property
    def labels_(self) -> npt.NDArray[np.int64]:
        if self._labels_ is not None:
            return self._labels_.reshape(self.N, self.T)
        
        return self._labels_
    
    @labels_.setter
    def labels_(self, new_value: Any) -> None:
        self._labels_ = new_value

    @property
    def fitted_data_shape_(self) -> Tuple[int, int, int]:
        """
        returns a tuple of the shape of the fitted data in TNF format. E.g (T, N, F) where T, N, and F are the number of timesteps, observations, and features respectively. 
        """
        
        return self.T, self.N, self.F