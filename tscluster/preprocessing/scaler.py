from __future__ import annotations
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tscluster.preprocessing.interface import TSPreprocessorInterface
from tscluster.preprocessing.utils import reshape_for_transform, infer_data

class TSScaler(TSPreprocessorInterface):
    def __init__(self, scaler, per_time: bool = True, **kwargs) -> None:
        # parent class for transformers

        self._scaler = scaler # scaler object (e.g. sklearn's scaler obejct for each time step)
        self.per_time = per_time
        self.kwargs = kwargs 
    
    @property
    def label_dict_(self) -> dict:
        """
        returns a dictionary of the labels. Keys are: 'T', 'N', and 'F' which are the number of time steps, entities, and features respectively. Each key's value is a list of its labels seen during the last time the data was fitted, transformed or inverse_transformed. 
        """
        return self._label_dict_
    
    def set_label_dict_(self, value: dict) -> None:
        """
        Method to manually set the label_dict_.

        Parameters
        ----------
        value : dict
            the value to set as label_dict_. Should be a dict with all of 'T', 'N', and 'F' (case sensitive, which are number of time steps, entities, and features respectively) as key. The value of each key is a list of labels for the key in the data.  If your data don't have values for any of the keys, set its value to None.
        """

        valid_keys = {'T', 'N', 'F'}

        if not isinstance(value, dict):
            raise TypeError(f"Expected value to be of type 'dict', but got '{type(value).__name__}'")
        elif any(k not in valid_keys for k in value.keys()):
            raise ValueError(f"Expected dict to have all of {valid_keys} as key.")
        
        self._label_dict_ = value

    def fit(
        self, 
        X: npt.NDArray[np.float64],
        **kwargs
        ) -> 'TSScaler':
        """
        Fit method of transformer. 

        Parameters
        ----------
        X: ndarray
            Input time series data. Should be a 3 dimensional array in TNF fromat.
       **kwargs keyword arguments to be passed to fit method.
            
        Returns 
        -------
        self
            the fitted transformer object
        """
        
        X, n = reshape_for_transform(X, self.per_time)

        self._scalers = [self._scaler(**self.kwargs).fit(X[i], *kwargs) for i in range(n)] 

        return self
    
    def transform(
        self, 
        X: npt.NDArray[np.float64], 
        **kwargs
        ) -> npt.NDArray[np.float64]:
        """
        transform method for  transformer. 

        Parameters
        ----------
        X: ndarray
            Input time series data. Should be a 3 dimensional array in TNF fromat.
       **kwargs keyword arguments.

        Returns
        -------
        numpy array 
            the transformed data in TNF format
        """
       
        _shape = X.shape

        X, _ = reshape_for_transform(X, self.per_time)

        return np.array([scaler.transform(X[i]) for i, scaler in enumerate(self._scalers)]).reshape(*_shape)
    
    def inverse_transform(
        self, 
        X: npt.NDArray[np.float64],
        **kwargs
        ) -> npt.NDArray[np.float64]:
        """
        inverse transform method for  transformer. 

        Parameters
        ----------
        X: ndarray. 
            Input time series data. Should be a 3 dimensional array in TNF fromat.
       **kwargs keyword arguments.
    
        Returns
        -------
        numpy array 
            the inverse-transform of the data in TNF format
        """

        _shape = X.shape

        X, _ = reshape_for_transform(X, self.per_time)

        return np.array([scaler.inverse_transform(X[i]) for i, scaler in enumerate(self._scalers)]).reshape(*_shape)

    def fit_transform(
        self, 
        X: npt.NDArray[np.float64],
        **kwargs
        ) -> npt.NDArray[np.float64]:
        """
        fit and transform the data

        Parameters
        ----------
        X: ndarray
            Input time series data. Should be a 3 dimensional array in TNF fromat.
       **kwargs keyword arguments
       
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
