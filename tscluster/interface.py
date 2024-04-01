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

        Parameters
        ----------
        X : ndarray, string or list. 
            Input time series data. If ndarray, should be a 3 dimensional array. If str and a file name, will use numpy to load file.
            If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
            Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
            Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
            If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
            If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.

        *args 
            any other positional arguments for fit method.

        **kwargs
            Any keyword argument for fit method or infer_data. Keyword arguments to be passed to the decorator (infer_data) are
            arr_format : str, default 'TNF'
                format of the loaded data. 'TNF' means the data dimension is Time x Number of observations x Features
                'NTF' means the data dimension is Number OF  observations x Time x Features
            suffix_sep : str, default '_'
                separator separating the file number from the filename.
            file_reader : str, default 'infer'
                file loader to use. Can be any of np.load, pd.read_csv, pd.read_json, and pd.read_excel. If 'infer', decorator will attempt to infer the file type from the file name 
                and use the approproate loader.
            read_file_args : dict, default empty dictionary.
                parameters to be passed to the data loader.

        Returns
        -------
        self
        """
        
        raise NotImplementedError
    
    @property
    @abstractmethod
    def cluster_centers_(self) -> npt.NDArray[np.float64]:
        """
        returns the cluster centers. If scheme is fixed centers, returns a k x F 2D array. Where k is the number of clusters and F is the number of features. If scheme is changing centers, returns a T x k x F 3D array. Where T is the number of time stesp, k is the number of clusters and F is the number of features.
        """

        raise NotImplementedError
    
    @property
    @abstractmethod
    def labels_(self) -> npt.NDArray[np.float64]:
        """
        returns the assignment labels. values are integers in range [0, k-1], where k is the number of clusters. If scheme is fixed assignment, returns a 1D array of size N. Where N is the number of entities. A value of j at the i-th index means that entity i is assigned to the j-th cluster at all time steps. If scheme is changing assignment, returns a N x T 2D array. Where N is the number of entities and T is the number of time steps. A value of j at the i-th row and t-th column means that entity i is assigned to the j-th cluster at the t-th time step.
        """
        
        raise NotImplementedError

    @property
    @abstractmethod
    def fitted_data_shape_(self) -> Tuple[int, int, int]:
        """
        returns a tuple of the shape of the fitted data in TNF format. E.g (T, N, F) where T, N, and F are the number of timesteps, observations, and features respectively. 
        """
        
        raise NotImplementedError

    @abstractmethod
    def get_named_cluster_centers(
                                self, 
                                label_dict: dict|None = None
                                   ) -> List[pd.DataFrame]:
        """
        Method to return the cluster centers with custom names of time steps and features.

        Method to return the a data frame of the label assignments with custom names of time steps and entities.

        Parameters
        -----------
        label_dict dict, default=None
            a dictionary whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
            - 'T' is a list of names/labels of each time step to be used as index of each dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
            - 'N' is a list of names/labels of each entity to be used as index of the dataframe. If None, range(0, N) is used. Where N is the number of entities/observations in the fitted data 
            - 'F' is a list of names/labels of each feature to be used as column of each dataframe. If None, range(0, F) is used. Where F is the number of features in the fitted data 
            If label_dict is None, the result of self.label_dict_ is used.

        Returns
        --------
        list
            A list of k pandas DataFrames. Where k is the number of clusters. The i-th dataframe in the list is a T x F dataframe of the values of the cluster centers of the i-th cluster.   
        """
        
        raise NotImplementedError 

    @abstractmethod
    def get_named_labels(
                        self, 
                        label_dict: dict|None = None
                        ) -> pd.DataFrame:
        """
        Method to return the a data frame of the label assignments with custom names of time steps and entities.

        Parameters
        -----------
        label_dict dict, default=None
            a dictionary whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
            - 'T' is a list of names/labels of each time step to be used as index of each dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
            - 'N' is a list of names/labels of each entity to be used as index of the dataframe. If None, range(0, N) is used. Where N is the number of entities/observations in the fitted data 
            - 'F' is a list of names/labels of each feature to be used as column of each dataframe. If None, range(0, F) is used. Where F is the number of features in the fitted data 
            If label_dict is None, the result of self.label_dict_ is used.

        Returns
        -------- 
        pd.DataFrame
            A pandas DataFrame with shape (N, T). The value in the n-th row and t-th column is an integer indicating the custer assignment of the n-th entity/observation at time t.  
        """
        
        raise NotImplementedError 