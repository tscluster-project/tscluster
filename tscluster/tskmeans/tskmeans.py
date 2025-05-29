from __future__ import annotations
from typing import List, Any, Tuple

import numpy as np
import numpy.typing as npt
from tslearn.clustering import TimeSeriesKMeans

from tscluster.interface import TSClusterInterface
from tscluster.base import TSCluster
from tscluster.preprocessing.utils import tnf_to_ntf, ntf_to_tnf, infer_data

class TSKmeans(TimeSeriesKMeans, TSCluster, TSClusterInterface):
    """
    Applies tslearn's TimeSeriesKMeans clustering to the data. 

    Read more in the tslearn's user guide. The follow parameters are from tslearn's TimeSeriesKMeans constructor.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.

    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than this threshold between two consecutive iterations, the model is considered to have converged and the algorithm stops.

    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

    metric : {"euclidean", "dtw", "softdtw"} (default: "euclidean")
        Metric to be used for both cluster assignment and barycenter computation. If "dtw", DBA is used for barycenter computation.

    max_iter_barycenter : int (default: 100)
        Number of iterations for the barycenter computation process. Only used if metric="dtw" or metric="softdtw".

    metric_params : dict or None (default: None)
        Parameter values for the chosen metric. For metrics that accept parallelization of the cross-distance matrix computations, n_jobs key passed in metric_params is overridden by the n_jobs argument.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix computations. Ignored if the cross-distance matrix cannot be computed using parallelization.
        None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See scikit-learns'     Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>_ for more details.

    dtw_inertia: bool (default: False)
        Whether to compute DTW inertia even if DTW is not the chosen metric.

    verbose : int (default: 0)
        If nonzero, print information about the inertia while learning the model and joblib progress messages are printed.

    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it fixes the seed. Defaults to the global numpy random number generator.

    init : {'k-means++', 'random' or an ndarray} (default: 'k-means++')
        Method for initialization:
        'k-means++' : use k-means++ heuristic. See scikit-learn's k_init_ <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ cluster/k_means_.py>_ for more.
        'random': choose k observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, ts_size, d) and gives the initial centers.
    
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
        ) -> 'TSKmeans':
        """
        Method for fitting the model on the data.

        Parameters
        ----------
        X : numpy array
            Input time series data.Should be a 3 dimensional array in TNF fromat.
        label_dict : dict, default=None
            A dictionary of the labels of X. Keys should be 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
                - 'T' is a list of names/labels of each time step used as index of each dataframe during fit. Default is range(0, T). Where T is the number of time steps in the fitted data
                - 'N' is a list of names/labels of each entity used as index of the dataframe. Default is range(0, N). Where N is the number of entities/observations in the fitted data 
                - 'F' is a list of names/labels of each feature used as column of each dataframe. Default is range(0, F). Where F is the number of features in the fitted data 
            data_loader function from tscluster.preprocessing.utils can help in getting label_dict of a data. 
        **kwargs keyword arguments to be passed to tslearn's kmeans fit method

        Returns
        --------
        self: 
            The fitted TSKmeans object.
        """

        self._label_dict_ = label_dict

        self._labels_ = None
        self._cluster_centers_ = None

        self.Xt = tnf_to_ntf(X)

        self.N_, self.T_, self.F_ = self.Xt.shape

        super().fit(self.Xt, **kwargs) 

        self._cluster_centers_ = ntf_to_tnf(self._cluster_centers_)

        return self

    @property
    def cluster_centers_(self) -> npt.NDArray[np.float64]: 
        return self._cluster_centers_
    
    @cluster_centers_.setter
    def cluster_centers_(self, new_value: npt.NDArray[np.float64]) -> None:
        self._cluster_centers_ = new_value

    @property
    def labels_(self) -> npt.NDArray[np.int64]:
        """
        Returns the assignment labels. Values are integers in range [0, k-1], where k is the number of clusters. If scheme is fixed assignment, returns a 1D array of size N. Where N is the number of entities. A value of j at the i-th index means that entity i is assigned to the j-th cluster at all time steps. If scheme is changing assignment, returns a N x T 2D array. Where N is the number of entities and T is the number of time steps. A value of j at the i-th row and t-th column means that entity i is assigned to the j-th cluster at the t-th time step.
        """
        return self._labels_ 
    
    @labels_.setter
    def labels_(self, new_value: npt.NDArray[np.int64]) -> None:
        self._labels_ = new_value

    @property
    def fitted_data_shape_(self) -> Tuple[int, int, int]:
        """
        returns a tuple of the shape of the fitted data in TNF format. E.g (T, N, F) where T, N, and F are the number of timesteps,
        observations, and features respectively. 
        """
        return self.T_, self.N_, self.F_