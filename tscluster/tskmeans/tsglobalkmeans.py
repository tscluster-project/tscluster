from __future__ import annotations
from typing import List, Any, Tuple


import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from tscluster.interface import TSClusterInterface
from tscluster.base import TSCluster
from tscluster.preprocessing.utils import TNF_to_NTF, infer_data

class TSGlobalKmeans(KMeans, TSCluster, TSClusterInterface):
    # def __init__(self, *args, **kwargs):
    #     self._labels_ = None
    #     self._cluster_centers_ = None
    #     super().__init__(*args, **kwargs)

    @infer_data
    def fit(self, X: npt.NDArray[np.float64]|List|str) -> 'TSGlobalKmeans':
        self._labels_ = None
        self._cluster_centers_ = None

        self.Xt = TNF_to_NTF(X)

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
        returns a tuple of the shape of the fitted data in TNF format. E.g (T, N, F) where T, N, and F are the number of timesteps,
        observations, and features respectively. 
        """
        return self.T, self.N, self.F