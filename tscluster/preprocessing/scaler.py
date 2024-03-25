from __future__ import annotations
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tscluster.preprocessing.base import TSPreprocessor
from tscluster.preprocessing.utils import reshape_for_transform, infer_data

class TSScaler(TSPreprocessor):
    def __init__(self, scaler, per_time: bool = True, **kwargs) -> None:
        "parent class for transformers"
        self._scaler = scaler # scaler object (e.g. sklearn's scaler obejct for each time step)
        self.per_time = per_time
        self.kwargs = kwargs 
    
    @infer_data
    def fit(self, X: npt.NDArray[np.float64]|str|List) -> 'TSScaler':
        """
        Fit method of transformer. Should be deocrated with infer_data function located in tscluster.preprocessing.utils
        X: ndarray, string or list. 
            Input time series data. If ndarray, should be a 3 dimensional array. If str and a file name, will use numpy to load file.
            If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
            Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
            Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
            If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
            If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
        
        Returns the tranformed data
        """
        X, n = reshape_for_transform(X, self.per_time)

        self._scalers = [self._scaler(**self.kwargs).fit(X[i]) for i in range(n)] 

        return self
    
    @infer_data
    def transform(self, X: npt.NDArray[np.float64]|str|List) -> npt.NDArray[np.float64]:
        """
        transform method for  transformer. Should be deocrated with infer_data function located in tscluster.preprocessing.utils
        X: ndarray, string or list. 
            Input time series data. If ndarray, should be a 3 dimensional array. If str and a file name, will use numpy to load file.
            If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
            Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
            Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
            If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
            If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
        
        Returns the transform of the data
        """
       
        _shape = X.shape

        X, _ = reshape_for_transform(X, self.per_time)

        return np.array([scaler.transform(X[i]) for i, scaler in enumerate(self._scalers)]).reshape(*_shape)
    
    @infer_data
    def inverse_transform(self, X: npt.NDArray[np.float64]|str|List) -> npt.NDArray[np.float64]:
        """
        inverse transform method for  transformer. Should be deocrated with infer_data function located in tscluster.preprocessing.utils
        X: ndarray, string or list. 
            Input time series data. If ndarray, should be a 3 dimensional array. If str and a file name, will use numpy to load file.
            If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
            Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
            Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
            If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
            If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
        
        Returns the inverse-transform of the data
        """

        _shape = X.shape

        X, _ = reshape_for_transform(X, self.per_time)

        return np.array([scaler.inverse_transform(X[i]) for i, scaler in enumerate(self._scalers)]).reshape(*_shape)

    @infer_data
    def fit_transform(self, X: npt.NDArray[np.float64]|str|List) -> npt.NDArray[np.float64]:
        self.fit(X)

        return self.transform(X)

class TSStandardScaler(TSScaler):
    def __init__(self, per_time: bool = True, **kwargs) -> None:
        """
        Uses zscore to scale a time series data.

        Args:

            per_time: bool, default True
            indicates if to compute zscore per time step.
            
            kwargs: keyword arugments to be passed to data loader or scaler
        """
        scaler = StandardScaler
        super().__init__(scaler, per_time, **kwargs)

class TSMinMaxScaler(TSScaler):
    def __init__(self, per_time: bool = True, **kwargs) -> None:
        """
        Uses min-max to scale a time series data to [0, 1].

        Args:

            per_time: bool, default True
            indicates if to scale per time step.
            
            kwargs: keyword arugments to be passed to data loader or scaler
        """

        scaler = MinMaxScaler
        super().__init__(scaler, per_time, **kwargs)
