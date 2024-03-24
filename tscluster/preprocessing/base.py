from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

class TSPreprocessor(ABC):
    @abstractmethod
    def fit(self, X:npt.NDArray[np.float64], *args, **kwargs) -> 'TSPreprocessor':
        """should be deocrated with infer_data in tscluster.preprocessing.utils"""
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, X:npt.NDArray[np.float64], *args, **kwargs) -> npt.NDArray[np.float64]:
        """should be deocrated with infer_data in tscluster.preprocessing.utils"""
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, X:npt.NDArray[np.float64], *args, **kwargs) -> npt.NDArray[np.float64]:
        """should be deocrated with infer_data in tscluster.preprocessing.utils"""
        raise NotImplementedError