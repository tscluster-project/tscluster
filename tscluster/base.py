from __future__ import annotations
from typing import List, Tuple

import numpy.typing as npt
import numpy as np
import pandas as pd 

from tscluster.preprocessing.utils import broadcast_data, tnf_to_ntf

class TSCluster():
    """
    Class that contains implementations of common methods for a temporal clustering model
    """


    @property
    def label_dict_(self) -> dict:
        """
        returns a dictionary of the labels whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
            - 'T' is a list of names/labels of each time step used as index of each dataframe during fit. Default is range(0, T). Where T is the number of time steps in the fitted data
            - 'N' is a list of names/labels of each entity used as index of the dataframe. Default is range(0, N). Where N is the number of entities/observations in the fitted data 
            - 'F' is a list of names/labels of each feature used as column of each dataframe. Default is range(0, F). Where F is the number of features in the fitted data 

        """
        keys = ('T', 'N', 'F')
        if self._label_dict_ is None:
            self._label_dict_ = {}
        
        for i, k in enumerate(keys):
            _ = self._label_dict_.setdefault(k, list(range(self.fitted_data_shape_[i])))

        return self._label_dict_
    
    def set_label_dict(self, value: dict) -> None:
        """
        Method to manually set the label_dict_.

        Parameters
        ----------
        value : dict
            the value to set as label_dict_. Should be a dict with all of 'T', 'N', and 'F' (case sensitive, which are number of time steps, entities, and features respectively) as key. The value of each key is a list of labels for the key in the data.  If your data don't have values for any of the keys, set its value to None.
        
        Returns
        -------
        dict 
            a dictionary whose keys are 'T', 'N', and 'F'; and values are lists of the labels of each key.
        """

        valid_keys = {'T', 'N', 'F'}

        if not isinstance(value, dict):
            raise TypeError(f"Expected value to be of type 'dict', but got '{type(value).__name__}'")
        elif any(k not in valid_keys for k in value.keys()):
            raise ValueError(f"Expected dict to have all of {valid_keys} as key.")
        
        self._label_dict_ = value


    def get_named_cluster_centers(
                                self, 
                                label_dict : dict|None = None
                                   ) -> List[pd.DataFrame]:
        """
        Method to return the cluster centers with custom names of time steps and features.

        Parameters
        ----------
        label_dict dict, default=None
            a dictionary whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
            - 'T' is a list of names/labels of each time step to be used as index of each dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
            - 'N' is a list of names/labels of each entity to be used as index of the dataframe. If None, range(0, N) is used. Where N is the number of entities/observations in the fitted data 
            - 'F' is a list of names/labels of each feature to be used as column of each dataframe. If None, range(0, F) is used. Where F is the number of features in the fitted data 
            If label_dict is None, the result of self.label_dict_ is used.

        Returns
        ------ 
        list    
            A list of k pandas DataFrames. Where k is the number of clusters. The i-th dataframe in the list is a T x F dataframe of the values of the cluster centers of the i-th cluster.   
        """

        cluster_centers, _ = broadcast_data(self.fitted_data_shape_[0], cluster_centers=self.cluster_centers_)

        cluster_centers = tnf_to_ntf(cluster_centers)
    
        if label_dict is None:
            time, features = self.label_dict_['T'], self.label_dict_['F']

        else:
            time, features = label_dict['T'], label_dict['F']

            if time is None:
                time = list(range(self.fitted_data_shape_[0]))
            if features is None:
                features = list(range(self.fitted_data_shape_[2]))

        return [pd.DataFrame(cluster_centers_k, columns=features, index=time) for cluster_centers_k in cluster_centers]

    def get_named_labels(
                        self, 
                        label_dict: dict|None = None
                        ) -> pd.DataFrame:
        """
        Method to return the a data frame of the label assignments with custom names of time steps and entities.

        Parameters
        -----------
        label_dict : dict, default=None
            a dictionary whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
            - 'T' is a list of names/labels of each time step to be used as index of each dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
            - 'N' is a list of names/labels of each entity to be used as index of the dataframe. If None, range(0, N) is used. Where N is the number of entities/observations in the fitted data 
            - 'F' is a list of names/labels of each feature to be used as column of each dataframe. If None, range(0, F) is used. Where F is the number of features in the fitted data 
            If label_dict is None, the result of self.label_dict_ is used.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with shape (N, T). The value in the n-th row and t-th column is an integer indicating the custer assignment of the n-th entity/observation at time t.  
        """
        
        _, labels = broadcast_data(self.fitted_data_shape_[0], labels=self.labels_)
    
        if label_dict is None:
            time, entities = self.label_dict_['T'], self.label_dict_['N']

        else:
            time, entities = label_dict['T'], label_dict['N']

            if time is None:
                time = list(range(self.fitted_data_shape_[0]))
            if entities is None:
                entities = list(range(self.fitted_data_shape_[1]))

        return pd.DataFrame(labels, columns=time, index=entities)

    def _get_changes(self) -> npt.NDArray[np.int64]:
        labels = self.get_named_labels().values
        return np.sum(labels[:, :-1] != labels[:, 1:], axis=1)
    
    def get_dynamic_entities(self) -> Tuple[List[np.int64], List[np.int64]]:
        """
        returns the dynamic entities and their number of changes. Both lists are sorted by the number of cluster changes in descending order.

        Returns
        -------
        dynamic entities : list
            a 1-D array of the indexes of the entities that change cluster at least once.
        number of changes : list
            a 1-D array of the number of changes for each dynamic entity such that the i-th element is the number of cluster changes for the i-th dynamic entity
        """
        
        n_changes = self._get_changes()
        changes = n_changes > 0

        entities = self.label_dict_['N']
        dynamic_entities = np.where(changes)[0]

        sort_filter = np.argsort(n_changes[changes])[::-1]

        return [entities[i] for i in dynamic_entities[sort_filter]], list(np.sort(n_changes[changes])[::-1])

    def get_index_of_label(self, labels: List[str], axis: str = 'N') -> List[int]:
        """
        function to return the integer indexes of some given labelled items in `self.label_dict_`. The indexes are assumed to be 0-indexed.

        Parameters
        ----------
        labels : list
            a list of the label(s) whose integer indexes should be returned.
        axis : str, default='N'
            can be any of {'T', 'N', 'F'}. 
            - If 'T', the values in the `labels` parameter are interpreted as time labels (as stored in `self.label_dict_['T']`). 
            - If 'N', the values in the `labels` parameter are interpreted as entity labels (as stored in `self.label_dict_['N']`). 
            - If 'F', the values in the `labels` parameter are interpreted as feature labels (as stored in `self.label_dict_['F']`). 
    
        Returns
        -------
        list
            a list of the integer indexes of the labels in the given axis dimension.
        """ 

        return [self.label_dict_[axis].index(label) for label in labels]

    def get_label_of_index(self, indexes: List[int], axis: str = 'N') -> List[str]:
        """
        function to return the labels of some given integer indexes as labelled in `self.label_dict_`. The indexes are assumed to be 0-indexed.

        Parameters
        ----------
        indexes : list
            a list of the index(es) whose labels should be returned.
        axis : str, default='N'
            can be any of {'T', 'N', 'F'}. 
            - If 'T', the values in the `indexes` parameter are interpreted as the time indexes whose labels (as stored in `self.label_dict_['T']`) should be returned. 
            - If 'N', the values in the `indexes` parameter are interpreted as the entity indexes whose labels (as stored in `self.label_dict_['N']`) should be returned. 
            - If 'F', the values in the `indexes` parameter are interpreted as the feature indexes whose labels (as stored in `self.label_dict_['F']`) should be returned. 
    
        Returns
        -------
        list
            a list of the labels of the given integer indexes in the given axis dimension.
        """ 
        
        return list(pd.Series(self.label_dict_[axis]).values[indexes])
    
    @property
    def n_changes_(self) -> int:
        """
        returns the total number of label changes
        """
        return np.sum(self._get_changes())