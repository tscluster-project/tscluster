from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

class TSCluster(ABC):
    @abstractmethod
    def fit(self, X:npt.NDArray[np.float64], *args, **kwargs) -> 'TSCluster':
        """should be deocrated with infer_data in tscluster.preprocessing.utils"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def cluster_centers_(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def labels_(self):
        raise NotImplementedError
