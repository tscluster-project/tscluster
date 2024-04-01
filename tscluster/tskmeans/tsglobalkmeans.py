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
    """

    @infer_data
    def fit(
        self, 
        X: npt.NDArray[np.float64]|List|str,
        y: npt.NDArray[np.float64] | npt.NDArray[np.int64] | None = None 
        ) -> 'TSGlobalKmeans':
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
        -------
        self: 
            The fitted TSGlobalKmeans object. 
        """

        self._labels_ = None
        self._cluster_centers_ = None

        self.Xt = tnf_to_ntf(X)

        self.N, self.T, self.F = self.Xt.shape

        self.Xt = np.vstack(self.Xt)

        super().fit(self.Xt) 

        return self

    @property
    def cluster_centers_(self) -> npt.NDArray[np.float64]: 
        return self._cluster_centers_
    
    @cluster_centers_.setter
    def cluster_centers_(self, new_value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self._cluster_centers_ = new_value

    @property
    def labels_(self) -> npt.NDArray[np.int64]:
        if self._labels_ is not None:
            return self._labels_.reshape(self.N, self.T)
        
        return self._labels_
    
    @labels_.setter
    def labels_(self, new_value: Any) -> npt.NDArray[np.int64]:
        self._labels_ = new_value

    @property
    def fitted_data_shape_(self) -> Tuple[int, int, int]:
        """
        returns a tuple of the shape of the fitted data in TNF format. E.g (T, N, F) where T, N, and F are the number of timesteps, observations, and features respectively. 
        """
        
        return self.T, self.N, self.F