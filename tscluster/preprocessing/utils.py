from __future__ import annotations
from typing import List, Callable, Any, Tuple
import os 
from functools import wraps

import numpy as np 
import numpy.typing as npt
import pandas as pd

valid_data_load_types = {np.ndarray, pd.DataFrame, str, list}
valid_data_load_types_names = tuple[npt.NDArray, int](map(lambda x: x.__name__, valid_data_load_types))

default_data_loader_args = {
    "arr_format": "TNF", 
    "suffix_sep": "_", 
    "read_file_args": {},
    "file_reader": "infer", # can be one of ("infer", "np_load", "pd_read_csv", "pd_read_json", pd_read_excel)
    'use_suffix_as_label': False
    }

file_readers = {
    "np_load": np.load,
    "pd_read_csv": pd.read_csv,
    "pd_read_json": pd.read_json,
    "pd_read_excel": pd.read_excel
}

def tnf_to_ntf(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Utility function to convert an array from Time x Number of observation x Feature format to Number of observation x Time x Feature format
    """

    T, N, F = X.shape 

    Xt = np.zeros(shape=(N, T, F))

    for t in range(T):
        Xt[:, t, :] = X[t, :, :]

    return Xt 

def ntf_to_tnf(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Utility function to convert an array from Number of observation x Time x Feature format to Time x Number of observation x Feature format         
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

def to_tnf(X: npt.NDArray[np.float64], arr_format: str, label_dict: dict) -> Tuple[npt.NDArray[np.float64], dict]:
    """
    Utility function to check the format of an array and converts it to TNF format. Raises ValueError if the array is not 3d.
    """
    if X.ndim != 3:
        raise ValueError(f"Invalid dimension of array. Expected array with 3 dimensions but got {X.ndim}")
    
    elif arr_format.upper() == 'NTF':
        label_dict['T'], label_dict['N'] = label_dict['N'], label_dict['T']
        
        return ntf_to_tnf(X), label_dict
    
    return X, label_dict

def broadcast_data(
        T: int,
        *,
        cluster_centers: npt.NDArray[np.float64]|None = None, 
        labels: npt.NDArray[np.int]|None = None, 
        ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    
    """function to make cluster_centers of shape T x N x F and labels of shape N x F """

    if cluster_centers is not None:
        if cluster_centers.ndim == 2:
            cluster_centers = np.array([cluster_centers for _ in range(T)])

    if labels is not None:
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

def read_all_files(lst: List[str], file_reader: str, **kwargs) -> Tuple[npt.NDArray[np.float64], dict]:
    """
    Utility function to read all the files in a list
    """
    if file_reader == 'infer':

        file_extension = {filename.split('.')[-1].lower() for filename in lst}

        if 'npy' in file_extension or 'npz' in file_extension:
            df_lst = [pd.DataFrame(file_readers['np_load'](file_path, **kwargs)) for file_path in lst]

        elif 'json' in file_extension:
            df_lst = [file_readers['pd_read_json'](file_path, **kwargs).sort_index() for file_path in lst]


        elif 'xls' in file_extension or 'xlsx' in file_extension:
            kwargs = get_default_header('pd_read_excel', kwargs)
            df_lst = [file_readers['pd_read_excel'](file_path, **kwargs).sort_index() for file_path in lst]
        
        else: # assume csv
            kwargs = get_default_header('pd_read_csv', kwargs)
            df_lst = [file_readers['pd_read_csv'](file_path, **kwargs).sort_index() for file_path in lst]   
        
    elif file_reader == 'np_load': # if numpy
        df_lst = [pd.DataFrame(file_readers['np_load'](file_path, **kwargs)) for file_path in lst]
    
    else: # if pandas
        kwargs = get_default_header(file_reader, kwargs)
        df_lst = [file_readers[file_reader](file_path, **kwargs).sort_index() for file_path in lst]   

    df = pd.concat(df_lst, axis=0, sort=False)
    label_dict = {
        'T': list(range(len(lst))),
        'N': list(df_lst[0].index),
        'F': list(df.columns)
        }
    return df.values.reshape(len(lst), -1, df.shape[1]), label_dict 

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
    
    # Decorator to infer the data type of X, load it and return it in TNF format.
    
    @wraps(func)
    def args_selector(
            self: Any, 
            X: npt.NDArray[np.float64]|str|List, 
            *args, 
            **kwargs
            ) -> Any:
        
        # Decorator to infer the data type of X, load it and return it in TNF format.
    
        def data_loader(
                self: Any, 
                X: npt.NDArray[np.float64]|str|List, 
                arr_format: str, 
                suffix_sep: str, 
                file_reader: str,
                read_file_args: dict, 
                use_suffix_as_label: bool,
                *args, 
                **kwargs
                ) -> Any:
            
            """
            Parameters
            -----------
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
            use_suffix_as_label: bool, default False
                If True, use the suffixes of the file names as labels in `label_dict`. If `arr_format` = 'TNF', the suffixes will be used as labels of timesteps, else if `arr_format` = 'NTF', they will be used as labels of entities.
                If False, a linear range of the number of files is used as labels (e.g.) range(n_files), where n_files is the number of files.
            *args: 
                any other positional arguments for fit method or function to be decorated.

            **kwargs:
                any keyword argument for fit method, or function to be decorated.
            """

            if isinstance(X, np.ndarray):
                X_arr = X 
                label_dict = {
                    'T': list(range(X_arr.shape[0])),
                    'N': list(range(X_arr.shape[1])),
                    'F': list(range(X_arr.shape[2]))
                }
            
            elif isinstance(X, list):

                if is_all_type(X, np.ndarray):
                    X_arr = np.array(X)
                    label_dict = {
                        'T': list(range(X_arr.shape[0])),
                        'N': list(range(X_arr.shape[1])),
                        'F': list(range(X_arr.shape[2]))
                    }

                elif is_all_type(X, pd.DataFrame):
                    df_lst = [df.sort_index() for df in X]
                    df = pd.concat(df_lst, axis=0, sort=False)
                    label_dict = {
                        'T': list(range(len(X))),
                        'N': list(df_lst[0].index),
                        'F': list(df.columns)
                        }
                    X_arr = df.values.reshape(len(X), -1, df.shape[1])
                    
                elif is_all_type(X, str):
                    X_arr, label_dict = read_all_files(X, file_reader, **read_file_args)

            elif isinstance(X, str):

                if os.path.isfile(X):
                    X_arr = file_readers['np_load'](X, **read_file_args)
                    label_dict = {
                        'T': list(range(X_arr.shape[0])),
                        'N': list(range(X_arr.shape[1])),
                        'F': list(range(X_arr.shape[2]))
                    }

                else:
                    file_names = get_lst_of_filenames(X)

                    file_list_sort_key = lambda filename: int("".join(filename.split(".")[0]).split(suffix_sep)[-1])

                    sorted_filenames = sorted(file_names, key=file_list_sort_key)

                    lst_of_filepaths = [os.path.join(X, f) for f in sorted_filenames]

                    X_arr, label_dict = read_all_files(lst_of_filepaths, file_reader, **read_file_args)

                    if use_suffix_as_label:
                        label_dict['T'] = sorted([str(file_list_sort_key(f)) for f in file_names])
            
            else:
                raise TypeError(f"Invalid type! Expected any of {valid_data_load_types_names}, but got '{type(X).__name__}'")
            
            X_arr, label_dict = to_tnf(X_arr, arr_format, label_dict) 

            try:
                self._label_dict_ = label_dict
            except AttributeError:
                self.append(label_dict)

            return func(self, X_arr, *args, **kwargs)
        
        arr_format = get_infer_data_wrapper_args('arr_format', kwargs)
        suffix_sep = get_infer_data_wrapper_args('suffix_sep', kwargs)
        file_reader = get_infer_data_wrapper_args('file_reader', kwargs)
        read_file_args = get_infer_data_wrapper_args('read_file_args', kwargs)
        use_suffix_as_label = get_infer_data_wrapper_args('use_suffix_as_label', kwargs)
            
        return data_loader(self, X, arr_format, suffix_sep, file_reader, read_file_args, use_suffix_as_label, *args, **kwargs)
    
    return args_selector

@infer_data
def _get_inferred_data(label_dict: dict, X: npt.NDArray[np.float64]|str|List, **kwargs) -> np.float64:
    """
    function to replace self argument
    """
    return X

def load_data(
        X: npt.NDArray[np.float64]|str|List, 
        *,
        arr_format: str = 'TNF',
        suffix_sep: str = '_',
        file_reader: str = 'infer',
        read_file_args: dict|None = None,
        use_suffix_as_label: bool = False,
        output_arr_format: str = 'TNF'
        ) -> Tuple[np.float64, dict]:
    """
    function to load data

    Parameters
    ----------
    X : numpy array, string or list
        Input time series data. If ndarray, should be a 3 dimensional array, use `arr_format` to specify its format. If str and a file name, will use numpy to load file.
        If str and a directory name, will load all the files in the directory in ascending order of the suffix of the filenames.
        Use suffix_sep as a keyword argument to indicate the suffix separator. Default is "_". So, file_0.csv will be read first before file_1.csv and so on.
        Supported files in the directory are any file that can be read using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
        If list, assumes the list is a list of files or filepaths. If file, each should be a numpy array or pandas DataFrame of data for the different time steps.
        If list of filepaths, data is read in the order in the list using any of np.load, pd.read_csv, pd.read_json, and pd.read_excel.
    arr_format : str, default='TNF'
        format of the input data. 'TNF' means the data dimension is Time x Number of observations x Features
        'NTF' means the data dimension is Number OF  observations x Time x Features
    suffix_sep : str, default='_'
        separator separating the suffix from the filename. The suffixes should be numbers and may not neccessarily need to start from 1 or have an interval of 1. So long the suffixes can be sorted and there is a consistent suffix separator, the directory can be parsed by `load_data` function.
    file_reader : str, default='infer'
        file loader to use. Can be any of np.load, pd.read_csv, pd.read_json, and pd.read_excel. If 'infer', decorator will attempt to infer the file type from the file name 
        and use the approproate loader.
    read_file_args : dict, default=None.
        parameters to be passed to the data loader. Keys of the dictionary should be parameter names as keys in str, values should be the values of the parameter keys.
    use_suffix_as_label: bool, default=False
        If True, use the suffixes of the file names as labels in `label_dict`. If `arr_format` = 'TNF', the suffixes will be used as labels of timesteps, else if `arr_format` = 'NTF', they will be used as labels of entities.
        If False, a linear range of the number of files is used as labels (e.g.) range(n_files), where n_files is the number of files.
    output_arr_format : str, default='TNF'
        The format of the output array. Can be any of {'TNF', 'NTF'}.

    Returns
    --------
    np.array 
        a numpy array of the data in 'TNF' or 'NTF' format (depending on the value of `output_arr_format`)
    dict
        a dictionary whose keys are 'T', 'N', and 'F', and whose values are lists of the labels of each key. 
    """

    label_dict = []

    if read_file_args is None:
        read_file_args = {}

    X_arr = _get_inferred_data(
        label_dict, 
        X, 
        arr_format=arr_format, 
        suffix_sep=suffix_sep,
        file_reader=file_reader,
        read_file_args=read_file_args,
        use_suffix_as_label=use_suffix_as_label
        )

    if output_arr_format == 'NTF':
        X_arr = tnf_to_ntf(X_arr)

    return X_arr, label_dict[0]

def to_dfs(
        X: npt.NDArray[np.float64],
        label_dict: dict|None = None,
        arr_format: str = 'TNF',
        output_df_format: str = 'TNF'
        ):
    
    """
    Function to convert from (numpy, label_dict) to dataframes

    Parameters
    ----------
    X : numpy array
        The array to be converted to list of dataframes. Should be a 3D array.
    label_dict dict, default=None
        a dictionary whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
        - 'T' is a list of names/labels of each time step to be used as index of each dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
        - 'N' is a list of names/labels of each entity to be used as index of the dataframe. If None, range(0, N) is used. Where N is the number of entities/observations in the fitted data 
        - 'F' is a list of names/labels of each feature to be used as column of each dataframe. If None, range(0, F) is used. Where F is the number of features in the fitted data 
        If label_dict is None, a linear range of the dimensions of the array is used.
    arr_format : str, default 'TNF'
        format of the array. 'TNF' means the data dimension is Time x Number of observations x Features
        'NTF' means the data dimension is Number OF  observations x Time x Features
    output_df_format : str, default='TNF'
        The format of the output dataframes. Can be any of {'TNF', 'NTF'}. If 'TNF', output is a list of T dataframes each of shape (N, F). If 'NTF', output is a list of N dataframes each of shape (T, F).

    Returns
    -------
    list[pd.DataFrame]
        A list of T pandas DataFrames. Where T is the number of time steps. The t-th dataframe in the list is a N x F dataframe of the values of the time series data of all entities at the t-th timestep.    
    """

    # arr_format will help us to interpret the array and convert it to a common format i.e TNF
    if arr_format == 'NTF':
        Xt = ntf_to_tnf(X)

        arr_lst = [Xt[t] for t in range(Xt.shape[0])]

    elif arr_format == 'TNF':
        arr_lst = [X[t] for t in range(X.shape[0])]

    else:
        raise ValueError(f"Expected arr_format to be any of {'TNF', 'NTF'} but got '{arr_format}'")

    dimensions = ('T', 'N', 'F')

    # get label_dict ready
    if label_dict is None:
        label_dict = {k: list(range(np.array(arr_lst).shape[i])) for i, k in enumerate(dimensions)}

    else:
        for i, k in enumerate(dimensions):
            if label_dict[k] is None:
                label_dict[k] = list(range(np.array(arr_lst).shape[i]))

    # output to required format
    if output_df_format == 'NTF':
        Xt = tnf_to_ntf(np.array(arr_lst))

        df_lst = [pd.DataFrame(Xt[i], index=label_dict['T'], columns=label_dict['F']) for i in range(Xt.shape[0])]

    elif output_df_format == 'TNF':
        df_lst = [pd.DataFrame(Xt, index=label_dict['N'], columns=label_dict['F']) for Xt in arr_lst]

    else:
        raise ValueError(f"Expected output_df_format to be any of {'TNF', 'NTF'} but got '{output_df_format}'")

    return df_lst