
import os 

import numpy as np 
import numpy.typing as npt
import pandas as pd

valid_data_load_types = {np.ndarray, pd.DataFrame, str, list}
valid_data_load_types_names = tuple(map(lambda x: x.__name__, valid_data_load_types))

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

def get_default_header(file_reader, kwargs):
    if file_reader in ('pd_read_csv', 'pd_read_excel') and 'header' not in kwargs:
        kwargs['header'] = None

    return kwargs

def read_all_files(lst, file_reader, **kwargs):

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
            print(kwargs)
            return np.array([file_readers['pd_read_csv'](file_path, **kwargs).values for file_path in lst])    
        
    elif file_reader == 'np_load': # if numpy
        return np.array([file_readers['np_load'](file_path, **kwargs) for file_path in lst])
    
    else: # if pandas
        kwargs = get_default_header(file_reader, kwargs)
        return np.array([file_readers[file_reader](file_path, **kwargs).values for file_path in lst])    

def get_infer_data_wrapper_args(arg, kwargs):
        try:
            arg_value = kwargs.pop(arg)
        except KeyError:
            arg_value = default_data_loader_args[arg] 

        return arg_value   

def infer_data(func):

    def args_selector(self, X, *args, **kwargs):
    
        def data_loader(self, X, arr_format, suffix_sep, read_file_args, *args, **kwargs):
            """**read_file_args is passed to pd.read_csv"""

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
        read_file_args = get_infer_data_wrapper_args('read_file_args', kwargs)
        file_reader = get_infer_data_wrapper_args('file_reader', kwargs)
            
        return data_loader(self, X, arr_format, suffix_sep, read_file_args, *args, **kwargs)
    
    return args_selector
