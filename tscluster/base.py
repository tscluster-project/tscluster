from __future__ import annotations
from typing import List, Tuple

import numpy.typing as npt
import numpy as np
import pandas as pd 

from tscluster.preprocessing.utils import broadcast_data, TNF_to_NTF

class TSCluster():
    """Class that contains common implementation of methods for a temporal clustering model"""

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
        cluster_centers, _ = broadcast_data(self.cluster_centers_, self.labels_, self.fitted_data_shape_[0])

        cluster_centers = TNF_to_NTF(cluster_centers)
    
        return [pd.DataFrame(cluster_centers_k, columns=features, index=time) for cluster_centers_k in cluster_centers]

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
        _, labels = broadcast_data(self.cluster_centers_, self.labels_, self.fitted_data_shape_[0])
    
        return pd.DataFrame(labels, columns=time, index=entities)
