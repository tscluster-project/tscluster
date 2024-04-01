from __future__ import annotations
from typing import List

from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

class TSPreprocessorInterface(ABC):
    @abstractmethod
    def fit(self, 
            X: npt.NDArray[np.float64]|str|List,
            *args, 
            **kwargs
            ) -> 'TSPreprocessorInterface':
       """
        Fit method of transformer. Should be deocrated with infer_data function located in tscluster.preprocessing.utils
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
    
    @abstractmethod
    def transform(self, 
                X: npt.NDArray[np.float64]|str|List, *args, **kwargs) -> npt.NDArray[np.float64]:
        """
        Transform method. Should be deocrated with infer_data function located in tscluster.preprocessing.utils
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

        return
            A numpy array of dimension T x N x F
        """

        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, 
            X: npt.NDArray[np.float64]|str|List, *args, **kwargs) -> npt.NDArray[np.float64]:
        """
        Inverse transform method. Should be deocrated with infer_data function located in tscluster.preprocessing.utils
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

        return
            A numpy array of dimension T x N x F
        """
        
        raise NotImplementedError