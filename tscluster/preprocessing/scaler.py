from __future__ import annotations
from typing import TextIO, Tuple, List
import sys
import re
from time import time
import copy

import numpy as np
from numpy import ndarray
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin

from tscluster.preprocessing.base import TSPreprocessor
from tscluster.preprocessing.utils import reshape_for_transform, infer_data

class TSScaler(TransformerMixin, TSPreprocessor):
    def __init__(self, scaler, per_time: bool = True, **kwargs) -> None:
        self._scaler = scaler
        self.per_time = per_time
        self.kwargs = kwargs 
    
    @infer_data
    def fit(self, X):

        X, n = reshape_for_transform(X, self.per_time)

        self._scalers = [self._scaler(**self.kwargs).fit(X[i]) for i in range(n)] 

        return self
    
    @infer_data
    def transform(self, X):
        _shape = X.shape

        X, _ = reshape_for_transform(X, self.per_time)

        return np.array([scaler.transform(X[i]) for i, scaler in enumerate(self._scalers)]).reshape(*_shape)
    
    @infer_data
    def inverse_transform(self, X):
        _shape = X.shape

        X, _ = reshape_for_transform(X, self.per_time)

        return np.array([scaler.inverse_transform(X[i]) for i, scaler in enumerate(self._scalers)]).reshape(*_shape)

class TSStandardScaler(TSScaler):
    def __init__(self, per_time: bool = True, **kwargs) -> None:
        scaler = StandardScaler
        super().__init__(scaler, per_time, **kwargs)

class TSMinMaxScaler(TSScaler):
    def __init__(self, per_time: bool = True, **kwargs) -> None:
        scaler = MinMaxScaler
        super().__init__(scaler, per_time, **kwargs)
