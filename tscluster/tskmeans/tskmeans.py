from __future__ import annotations
from typing import List, Any

import numpy as np
import numpy.typing as npt
from tslearn.clustering import TimeSeriesKMeans

from tscluster.base import TSCluster
from tscluster.preprocessing.utils import TNF_to_NTF, NTF_to_TNF, infer_data

class TSKmeans(TimeSeriesKMeans, TSCluster):

    @infer_data
    def fit(self, X: npt.NDArray[np.float64]|str|List) -> 'TSKmeans':
        self._labels_ = None
        self._cluster_centers_ = None

        self.Xt = TNF_to_NTF(X)

        super().fit(self.Xt) 

        self._cluster_centers_ = NTF_to_TNF(self._cluster_centers_)

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