
import os 

import numpy as np 
import numpy.typing as npt
import pandas as pd

valid_data_load_types = {np.ndarray, pd.DataFrame, str, list}
valid_data_load_types_names = tuple(map(lambda x: x.__name__, valid_data_load_types))

default_data_loader_args = {"arr_format": "TNF", "suffix_sep": "_", 'pd_read_csv_args': {}}

def TNF_to_NTF(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    T, N, F = X.shape 

    Xt = np.zeros(shape=(N, T, F))

    for t in range(T):
        Xt[:, t, :] = X[t, :, :]

    return Xt 

def NTF_to_TNF(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    N, T, F = X.shape 

    Xt = np.zeros(shape=(T, N, F))

    for t in range(T):
        Xt[t, :, :] = X[:, t, :]

    return Xt 

def reshape_for_transform(X, per_time):
    """
    Reshape to appropriate shape for transformation. Assumes input shape is TNF
    """
    if per_time:
        n = X.shape[0]

    else:
        n = 1
        X = np.array([np.vstack(X)])

    return X, n 

def to_TNF(X, arr_format):
    if X.ndim != 3:
        raise ValueError(f"Invalid dimension of array. Expected array with 3 dimensions but got {X.ndim}")
    
    elif arr_format.upper() == 'NTF':
        return NTF_to_TNF(X)
    return X

def is_all_type(lst, data_type, type_checker=isinstance):
    return all(map(lambda x: type_checker(x, data_type), lst))

def get_lst_of_filenames(dir):
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

def pandas_read_all_files(lst, **kwargs):
    return np.array([pd.read_csv(file_path, **kwargs).values for file_path in lst])

def get_infer_data_wrapper_args(arg, kwargs):
        try:
            arr_format = kwargs.pop(arg)
        except KeyError:
            arr_format = default_data_loader_args[arg] 

        return arr_format   

def infer_data(func):

    def args_selector(self, X, *args, **kwargs):
    
        def data_loader(self, X, arr_format, suffix_sep, pd_read_csv_args, *args, **kwargs):
            """**pd_read_csv_args is passed to pd.read_csv"""

            if isinstance(X, np.ndarray):
                X_arr = X 
            
            elif isinstance(X, list):

                if is_all_type(X, np.ndarray):
                    X_arr = np.array(X)

                elif is_all_type(X, pd.DataFrame):
                    X_arr = pd.concat(X, axis=0, sort=False).values

                elif is_all_type(X, str):
                    X_arr = pandas_read_all_files(X, **pd_read_csv_args)

            elif isinstance(X, str):
                file_paths = get_lst_of_filenames(X)

                file_list_sort_key = lambda filename: int("".join(filename.split(".")[0]).split(suffix_sep)[-1])

                X_arr = pandas_read_all_files(sorted(file_paths, key=file_list_sort_key), **pd_read_csv_args)
            
            else:
                raise TypeError(f"Invalid type! Expected any of {valid_data_load_types_names}, but got '{type(X).__name__}'")
            
            X_arr = to_TNF(X_arr, arr_format) 

            return func(self, X_arr, *args, **kwargs)
        
        arr_format = get_infer_data_wrapper_args('arr_format', kwargs)
        suffix_sep = get_infer_data_wrapper_args('suffix_sep', kwargs)
        pd_read_csv_args = get_infer_data_wrapper_args('pd_read_csv_args', kwargs)
            
        return data_loader(self, X, arr_format, suffix_sep, pd_read_csv_args, *args, **kwargs)
    
    return args_selector
