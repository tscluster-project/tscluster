from __future__ import annotations
from typing import List

from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

class TSPreprocessorInterface(ABC):
    @abstractmethod
    def fit(self, 
            X: npt.NDArray[np.float64],
            *args, 
            **kwargs
            ) -> 'TSPreprocessorInterface':
       """
        Fit method of transformer. 

        Parameters
        ----------
        X: ndarray. 
            Input time series data. Should be a 3 dimensional array in TNF fromat.
        args: 
            any other positional arguments for fit method.
        kwargs:
            any keyword argument for fit method 

        Returns
        ----------
        self
            the fitted transformer object
        """
       
       raise NotImplementedError
    
    @abstractmethod
    def transform(self, 
                X: npt.NDArray[np.float64], 
                *args, 
                **kwargs
                ) -> npt.NDArray[np.float64]:
        """
        Transform method. 

        Parameters
        ----------
        X: ndarray
            Input time series data. Should be a 3 dimensional array in TNF fromat.
        args: 
            any other positional arguments for fit method.
        kwargs:
            any keyword argument for fit method.

        Returns
        ----------
        numpy array
            The transformed data as numpy array of dimension T x N x F
        """

        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, 
            X: npt.NDArray[np.float64], 
            *args, 
            **kwargs
            ) -> npt.NDArray[np.float64]:
        """
        Inverse transform method. 

        Parameters
        ----------
        X: ndarray 
            Input time series data. Should be a 3 dimensional array in TNF fromat.
        args: 
            any other positional arguments for fit method.
        kwargs:
            any keyword argument for fit method.

        Returns
        ----------
        numpy array
            The inverse transform of the input data as numpy array of dimension T x N x F
        """
        
        raise NotImplementedError