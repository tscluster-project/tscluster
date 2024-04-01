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
    """
    
    @infer_data
    def fit(
        self, 
        X: npt.NDArray[np.float64]|str|List,
        y: npt.NDArray[np.float64] | npt.NDArray[np.int64] | None = None 
        ) -> 'TSKmeans':
        """
        Method for fitting the model on the data.

        Parameters
        ----------
        X : numpy array, string or list
            Input time series data. If ndarray, should be a 3 dimensional array, use `arr_format` to specify its format. If str and a file name, will use numpy to load file.
            If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
            Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
            Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
            If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
            If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
        y : None
            Ignored, not used. Only present as a convention for fit methods of most models.
        **kwargs keyword arguments, can be any of the following:
            - arr_format : str, default 'TNF'
                format of the loaded data. 'TNF' means the data dimension is Time x Number of observations x Features
                'NTF' means the data dimension is Number OF  observations x Time x Features
            - suffix_sep : str, default '_'
                separator separating the file number from the filename.
            - file_reader : str, default 'infer'
                file loader to use. Can be any of np.load, pd.read_csv, pd.read_json, and pd.read_excel. If 'infer', decorator will attempt to infer the file type from the file name 
                and use the approproate loader.
            - read_file_args : dict, default empty dictionary.
                parameters to be passed to the data loader.

        Returns
        --------
        self: 
            The fitted TSKmeans object.
        """

        self._labels_ = None
        self._cluster_centers_ = None

        self.Xt = tnf_to_ntf(X)

        self.N_, self.T_, self.F_ = self.Xt.shape

        super().fit(self.Xt) 

        self._cluster_centers_ = ntf_to_tnf(self._cluster_centers_)

        return self

    @property
    def cluster_centers_(self) -> npt.NDArray[np.float64]: 
        return self._cluster_centers_
    
    @cluster_centers_.setter
    def cluster_centers_(self, new_value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self._cluster_centers_ = new_value

    @property
    def labels_(self) -> npt.NDArray[np.int64]:
        """
        Returns the assignment labels. Values are integers in range [0, k-1], where k is the number of clusters. If scheme is fixed assignment, returns a 1D array of size N. Where N is the number of entities. A value of j at the i-th index means that entity i is assigned to the j-th cluster at all time steps. If scheme is changing assignment, returns a N x T 2D array. Where N is the number of entities and T is the number of time steps. A value of j at the i-th row and t-th column means that entity i is assigned to the j-th cluster at the t-th time step.
        """
        return self._labels_ 
    
    @labels_.setter
    def labels_(self, new_value: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        self._labels_ = new_value

    @property
    def fitted_data_shape_(self) -> Tuple[int, int, int]:
        """
        returns a tuple of the shape of the fitted data in TNF format. E.g (T, N, F) where T, N, and F are the number of timesteps,
        observations, and features respectively. 
        """
        return self.T_, self.N_, self.F_