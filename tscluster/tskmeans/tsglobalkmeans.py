from __future__ import annotations
from typing import TextIO, Tuple, List
import sys
import re
from time import time
import copy

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from tscluster.base import TSCluster
from tscluster.preprocessing.utils import TNF_to_NTF, infer_data

class TSGlobalKmeans(KMeans, TSCluster):
    # def __init__(self, *args, **kwargs):
    #     self._labels_ = None
    #     self._cluster_centers_ = None
    #     super().__init__(*args, **kwargs)

    @infer_data
    def fit(self, X):
        self._labels_ = None
        self._cluster_centers_ = None

        self.Xt = TNF_to_NTF(X)

        self.N, self.T, _ = self.Xt.shape

        self.Xt = np.vstack(self.Xt)

        super().fit(self.Xt) 

        return self

    @property
    def cluster_centers_(self): 
        return self._cluster_centers_
    
    @cluster_centers_.setter
    def cluster_centers_(self, new_value):
        self._cluster_centers_ = new_value

    @property
    def labels_(self):
        if self._labels_ is not None:
            return self._labels_.reshape(self.N, self.T)
        
        return self._labels_
    
    @labels_.setter
    def labels_(self, new_value):
        self._labels_ = new_value