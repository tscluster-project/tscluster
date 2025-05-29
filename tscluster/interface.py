from __future__ import annotations
from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy.typing as npt
import numpy as np
import pandas as pd 

class TSClusterInterface(ABC):
    @abstractmethod
    def fit(self, 
            X: npt.NDArray[np.float64], 
            label_dict: dict|None = None
            ) -> 'TSClusterInterface':
        """
        Fit method of model. 

        Parameters
        ----------
        X : ndarray
            Input time series data. Should be a 3 dimensional array in TNF fromat.
        label_dict : dict, default=None
            A dictionary of the labels of X. Keys should be 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
                - 'T' is a list of names/labels of each time step used as index of each dataframe during fit. Default is range(0, T). Where T is the number of time steps in the fitted data
                - 'N' is a list of names/labels of each entity used as index of the dataframe. Default is range(0, N). Where N is the number of entities/observations in the fitted data 
                - 'F' is a list of names/labels of each feature used as column of each dataframe. Default is range(0, F). Where F is the number of features in the fitted data 

            data_loader function from tscluster.preprocessing.utils can help in getting label_dict of a data. 
            
        Returns
        -------
        self
            The fitted model object
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

    @property
    @abstractmethod
    def label_dict_(self) -> dict:
        """
        returns a dictionary of the labels whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
            - 'T' is a list of names/labels of each time step used as index of each dataframe during fit. Default is range(0, T). Where T is the number of time steps in the fitted data
            - 'N' is a list of names/labels of each entity used as index of the dataframe. Default is range(0, N). Where N is the number of entities/observations in the fitted data 
            - 'F' is a list of names/labels of each feature used as column of each dataframe. Default is range(0, F). Where F is the number of features in the fitted data 
        """

        raise NotImplementedError
    
    @abstractmethod
    def set_label_dict(self, value: dict) -> None:
        """
        Method to manually set the label_dict_.

        Parameters
        ----------
        value : dict
            the value to set as label_dict_. Should be a dict with all of 'T', 'N', and 'F' (case sensitive, which are number of time steps, entities, and features respectively) as key. The value of each key is a list of labels for the key in the data.  If your data don't have values for any of the keys, set its value to None.
        """

        raise NotImplementedError

    @abstractmethod
    def get_named_cluster_centers(
                                self, 
                                label_dict: dict|None = None
                                   ) -> List[pd.DataFrame]:
        """
        Method to return the cluster centers with custom names of time steps and features.

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