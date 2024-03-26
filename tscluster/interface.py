from __future__ import annotations
from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy.typing as npt
import numpy as np
import pandas as pd 

class TSClusterInterface(ABC):
    @abstractmethod
    def fit(self, 
            X: npt.NDArray[np.float64]|str|List, 
            *args, 
            **kwargs
            ) -> 'TSClusterInterface':
        """
        Fit method of model. Should be decorated with infer_data function located in tscluster.preprocessing.utils
        X: ndarray, string or list. 
            Input time series data. If ndarray, should be a 3 dimensional array. If str and a file name, will use numpy to load file.
            If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
            Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
            Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
            If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
            If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.

        args: 
            any other positional arguments for fit method.

        kwargs:
            any keyword argument for fit method or data loader. e.g.
            arr_format: str, default 'TNF'
                format of the loaded data. 'TNF' means the data dimension is Time x Number of observations x Features
                'NTF' means the data dimension is Number OF  observations x Time x Features
            suffix_sep: str, default '_'
                separator separating the file number from the filename.
            file_reader: str, default 'infer'
                file loader to use. Can be any of np.load, pd.read_csv, pd.read_json, and pd.read_excel. If 'infer', decorator will attempt to infer the file type from the file name 
                and use the approproate loader.
            read_file_args: dict, default empty dictionary.
                parameters to be passed to the data loader.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def cluster_centers_(self) -> npt.NDArray[np.float64]:
        """
        getter method for cluster_centers) property
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def labels_(self) -> npt.NDArray[np.float64]:
        """setter method for cluster_centers) property"""
        raise NotImplementedError

    @property
    @abstractmethod
    def fitted_data_shape_(self) -> Tuple[int, int, int]:
        """
        returns a tuple of the shape of the fitted data in TNF format. E.g (T, N, F) where T, N, and F are the number of timesteps,
        observations, and features respectively. 
        """
        raise NotImplementedError

    @abstractmethod
    def get_named_cluster_centers(self, 
                                   time: List[str]|None = None, 
                                   features: List|None = None
                                   ) -> List[pd.DataFrame]:
        """
        Method to return the cluster centers with custom names of time steps and features.

        Args:
            time: list, default: None.
                A list of names of each time step to be used as index of each dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
            features: list, default: None
                A list of names of each feature to be used as column of each dataframe. If None, range(0, F) is used. Where F is the number of features in the fitted data 

        Return: A list of k pandas DataFrames. Where k is the number of clusters. The i-th dataframe in the list is a T x F dataframe of the values of the cluster centers of the i-th cluster.   
        """
        raise NotImplementedError 

    @abstractmethod
    def get_named_labels(self, 
                        time: List[str]|None = None, 
                        entities: List|None = None
                        ) -> pd.DataFrame:
        """
        Method to return the a data frame of the label assignments with custom names of time steps and entities.

        Args:
            time: list, default: None.
                A list of names of each time step to be used as column names of the dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
            entities: list, default: None
                A list of names of each entity to be used as index of the dataframe. If None, range(0, N) is used. Where N is the number of entities/observations in the fitted data 

        Return: A pandas DataFrame with shape (N, T). The value in the n-th row and t-th column is an integer indicating the custer assignment of the n-th entity/observation at time t.  
        """
        raise NotImplementedError 