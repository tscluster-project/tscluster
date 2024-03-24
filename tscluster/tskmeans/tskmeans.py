from __future__ import annotations
from typing import TextIO, Tuple, List
import sys
import re
from time import time
import copy

import numpy as np
import numpy.typing as npt
from tslearn.clustering import TimeSeriesKMeans

from tscluster.base import TSCluster
from tscluster.preprocessing.utils import TNF_to_NTF, infer_data

class TSKmeans(TimeSeriesKMeans, TSCluster):

    @infer_data
    def fit(self, X):
        self._labels_ = None
        self._cluster_centers_ = None
        self.Xt = TNF_to_NTF(X)
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
        return self._labels_ 
    
    @labels_.setter
    def labels_(self, new_value):
        self._labels_ = new_value