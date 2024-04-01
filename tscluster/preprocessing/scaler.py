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
        # parent class for transformers

        self._scaler = scaler # scaler object (e.g. sklearn's scaler obejct for each time step)
        self.per_time = per_time
        self.kwargs = kwargs 
    
    @infer_data
    def fit(self, X: npt.NDArray[np.float64]|str|List) -> 'TSScaler':
        """
        Fit method of transformer. 

        Parameters
        ----------
        X: ndarray, string or list. 
            Input time series data. If ndarray, should be a 3 dimensional array. If str and a file name, will use numpy to load file.
            If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
            Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
            Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
            If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
            If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
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
        self
            the fitted transformer object
        """
        X, n = reshape_for_transform(X, self.per_time)

        self._scalers = [self._scaler(**self.kwargs).fit(X[i]) for i in range(n)] 

        return self
    
    @infer_data
    def transform(self, X: npt.NDArray[np.float64]|str|List) -> npt.NDArray[np.float64]:
        """
        transform method for  transformer. 

        Parameters
        ----------
        X: ndarray, string or list. 
            Input time series data. If ndarray, should be a 3 dimensional array. If str and a file name, will use numpy to load file.
            If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
            Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
            Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
            If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
            If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
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
        numpy array 
            the transformed data in TNF format
        """
       
        _shape = X.shape

        X, _ = reshape_for_transform(X, self.per_time)

        return np.array([scaler.transform(X[i]) for i, scaler in enumerate(self._scalers)]).reshape(*_shape)
    
    @infer_data
    def inverse_transform(self, X: npt.NDArray[np.float64]|str|List) -> npt.NDArray[np.float64]:
        """
        inverse transform method for  transformer. 

        Parameters
        ----------
        X: ndarray, string or list. 
            Input time series data. If ndarray, should be a 3 dimensional array. If str and a file name, will use numpy to load file.
            If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
            Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
            Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
            If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
            If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
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
        numpy array 
            the inverse-transform of the data in TNF format
        """

        _shape = X.shape

        X, _ = reshape_for_transform(X, self.per_time)

        return np.array([scaler.inverse_transform(X[i]) for i, scaler in enumerate(self._scalers)]).reshape(*_shape)

    @infer_data
    def fit_transform(self, X: npt.NDArray[np.float64]|str|List) -> npt.NDArray[np.float64]:
        """
        fit and transform the data

        Parameters
        ----------
        X: ndarray, string or list. 
            Input time series data. If ndarray, should be a 3 dimensional array. If str and a file name, will use numpy to load file.
            If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
            Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
            Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
            If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
            If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
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
        numpy array 
            the transformed data in TNF format
        """

        self.fit(X)

        return self.transform(X)

class TSStandardScaler(TSScaler):
    """
    Uses sklearn's StandardScaler to scale a time series data.

    Parameters
    -----------
    per_time : bool, default=True
        If True, compute zscore per time step. If False, compute zscore per feature across all timesteps.
        
    **kwargs : keyword arugments to be passed to sklearn's StandardScaler.
    """

    def __init__(self, per_time: bool = True, **kwargs) -> None:

        scaler = StandardScaler
        super().__init__(scaler, per_time, **kwargs)

class TSMinMaxScaler(TSScaler):
    """
    Uses sklearn's MinMaxScaler to scale a time series data.

    Parameters
    -----------
    per_time : bool, default=True
        If True, compute zscore per time step. If False, compute zscore per feature across all timesteps.
        
    **kwargs : keyword arugments to be passed to sklearn's MinMaxScaler.
    """

    def __init__(self, per_time: bool = True, **kwargs) -> None:

        scaler = MinMaxScaler
        super().__init__(scaler, per_time, **kwargs)
