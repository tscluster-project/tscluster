from __future__ import annotations
from typing import List, Any, Tuple

import numpy as np
import numpy.typing as npt
from tslearn.clustering import TimeSeriesKMeans

from tscluster.interface import TSClusterInterface
from tscluster.base import TSCluster
from tscluster.preprocessing.utils import tnf_to_ntf, ntf_to_tnf, infer_data

class TSKmeans(TimeSeriesKMeans, TSCluster, TSClusterInterface):

    @infer_data
    def fit(self, X: npt.NDArray[np.float64]|str|List) -> 'TSKmeans':
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