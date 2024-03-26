from __future__ import annotations
from typing import List, Callable, Any, Tuple
import os 

import numpy as np 
import numpy.typing as npt
import pandas as pd

valid_data_load_types = {np.ndarray, pd.DataFrame, str, list}
valid_data_load_types_names = tuple[npt.NDArray, int](map(lambda x: x.__name__, valid_data_load_types))

default_data_loader_args = {
    "arr_format": "TNF", 
    "suffix_sep": "_", 
    "read_file_args": {},
    "file_reader": "infer" # can be one of ("infer", "np_load", "pd_read_csv", "pd_read_json", pd_read_excel)
    }

file_readers = {
    "np_load": np.load,
    "pd_read_csv": pd.read_csv,
    "pd_read_json": pd.read_json,
    "pd_read_excel": pd.read_excel
}

def TNF_to_NTF(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Utility function to convert an array from Time x Number of observation x Feature format to 
        Number of observation x Time x Feature format
    """
    T, N, F = X.shape 

    Xt = np.zeros(shape=(N, T, F))

    for t in range(T):
        Xt[:, t, :] = X[t, :, :]

    return Xt 

def NTF_to_TNF(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Utility function to convert an array from Number of observation x Time x Feature format to 
       Time x Number of observation x Feature format         
    """
    N, T, F = X.shape 

    Xt = np.zeros(shape=(T, N, F))

    for t in range(T):
        Xt[t, :, :] = X[:, t, :]

    return Xt 

def reshape_for_transform(X: npt.NDArray[np.float64], per_time: bool) -> Tuple[npt.NDArray, int]:
    """
    Reshape to appropriate shape for transformation. Assumes input shape is TNF
    """
    if per_time:
        n = X.shape[0]

    else:
        n = 1
        X = np.array([np.vstack(X)])

    return X, n 

def to_TNF(X: npt.NDArray[np.float64], arr_format: str) -> npt.NDArray[np.float64]:
    """
    Utility function to check the format of an array and converts it to TNF format. Raises ValueError if the array is not 3d.
    """
    if X.ndim != 3:
        raise ValueError(f"Invalid dimension of array. Expected array with 3 dimensions but got {X.ndim}")
    
    elif arr_format.upper() == 'NTF':
        return NTF_to_TNF(X)
    
    return X


def broadcast_data(
        cluster_centers: npt.NDArray[np.float64], 
        labels: npt.NDArray[np.int], 
        T: int
        ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    
    "function to make cluster_centers and labels both of size T x N x F"
    if cluster_centers.ndim == 2:
        cluster_centers = np.array([cluster_centers for _ in range(T)])

    if labels.ndim == 1:
        labels = np.array([labels for _ in range(T)]).T

    return cluster_centers, labels


def is_all_type(lst: List, data_type: type, type_checker: Callable[[Any, type], bool]=isinstance) -> bool:
    """
    Utility function to check if all elements of a list are of the same data type
    """
    return all(map(lambda x: type_checker(x, data_type), lst))

def get_lst_of_filenames(dir: str) -> List[str]:
    """
    Utility function to return a list of all filenames in a directory
    """
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

def get_default_header(file_reader: str, kwargs: dict) -> dict:
    """
    Utility function to set the default value header for pandas.read_csv or read_excel to None
    """
    if file_reader in ('pd_read_csv', 'pd_read_excel') and 'header' not in kwargs:
        kwargs['header'] = None

    return kwargs

def read_all_files(lst: List[str], file_reader: str, **kwargs) -> npt.NDArray[np.float64]:
    """
    Utility function to read all the files in a list
    """
    if file_reader == 'infer':

        file_extension = {filename.split('.')[-1].lower() for filename in lst}

        if 'npy' in file_extension or 'npz' in file_extension:
            return np.array([file_readers['np_load'](file_path, **kwargs) for file_path in lst])

        elif 'json' in file_extension:
            return np.array([file_readers['pd_read_json'](file_path, **kwargs).values for file_path in lst])

        elif 'xls' in file_extension or 'xlsx' in file_extension:
            kwargs = get_default_header('pd_read_excel', kwargs)
            return np.array([file_readers['pd_read_excel'](file_path, **kwargs).values for file_path in lst])
        
        else: # assume csv
            kwargs = get_default_header('pd_read_csv', kwargs)
            return np.array([file_readers['pd_read_csv'](file_path, **kwargs).values for file_path in lst])    
        
    elif file_reader == 'np_load': # if numpy
        return np.array([file_readers['np_load'](file_path, **kwargs) for file_path in lst])
    
    else: # if pandas
        kwargs = get_default_header(file_reader, kwargs)
        return np.array([file_readers[file_reader](file_path, **kwargs).values for file_path in lst])    

def get_infer_data_wrapper_args(arg: str, kwargs: dict) -> Any:
    """
    Utility function to get the value of arg if arg is an argument for the decorator e.g. argument for data loader
    """
    try:
        arg_value = kwargs.pop(arg)
    except KeyError:
        arg_value = default_data_loader_args[arg] 

    return arg_value   

def infer_data(func: Callable) -> Callable:
    """
    Decorator to infer the data type of X, load it and return it in TNF format.
    """

    def args_selector(
            self: Any, 
            X: npt.NDArray[np.float64]|str|List, 
            *args, 
            **kwargs
            ) -> Any:
    
        def data_loader(
                self: Any, 
                X: npt.NDArray[np.float64]|str|List, 
                arr_format: str, 
                suffix_sep: str, 
                file_reader: str,
                read_file_args: dict, 
                *args, 
                **kwargs
                ) -> Any:
            """
            X: ndarray, string or list. 
                Input time series data. If ndarray, should be a 3 dimensional array. If str and a file name, will use numpy to load file.
                If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
                Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
                Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
                If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
                If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.

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
            args: 
                any other positional arguments for fit method or function to be decorated.

            kwargs:
                any keyword argument for fit method, or function to be decorated.
            """
            if isinstance(X, np.ndarray):
                X_arr = X 
            
            elif isinstance(X, list):

                if is_all_type(X, np.ndarray):
                    X_arr = np.array(X)

                elif is_all_type(X, pd.DataFrame):
                    X_arr = pd.concat(X, axis=0, sort=False).values

                elif is_all_type(X, str):
                    X_arr = read_all_files(X, file_reader, **read_file_args)

            elif isinstance(X, str):

                if os.path.isfile(X):
                    X_arr = file_readers['np_load'](X, **read_file_args)

                else:
                    file_names = get_lst_of_filenames(X)

                    file_list_sort_key = lambda filename: int("".join(filename.split(".")[0]).split(suffix_sep)[-1])

                    sorted_filenames = sorted(file_names, key=file_list_sort_key)

                    lst_of_filepaths = [os.path.join(X, f) for f in sorted_filenames]

                    X_arr = read_all_files(lst_of_filepaths, file_reader, **read_file_args)
            
            else:
                raise TypeError(f"Invalid type! Expected any of {valid_data_load_types_names}, but got '{type(X).__name__}'")
            
            X_arr = to_TNF(X_arr, arr_format) 

            return func(self, X_arr, *args, **kwargs)
        
        arr_format = get_infer_data_wrapper_args('arr_format', kwargs)
        suffix_sep = get_infer_data_wrapper_args('suffix_sep', kwargs)
        file_reader = get_infer_data_wrapper_args('file_reader', kwargs)
        read_file_args = get_infer_data_wrapper_args('read_file_args', kwargs)
            
        return data_loader(self, X, arr_format, suffix_sep, file_reader, read_file_args, *args, **kwargs)
    
    return args_selector
