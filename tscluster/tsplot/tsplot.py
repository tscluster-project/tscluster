from __future__ import annotations
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib
import mpl_toolkits
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.widgets import Slider

from tscluster.preprocessing.utils import broadcast_data

def _data_validator(
        X: npt.NDArray[np.float64]|None = None, 
        cluster_centers: npt.NDArray[np.float64]|None = None, 
        labels: npt.NDArray[np.float64]|None = None        
    ) -> None:

    """function to check the shapes of the data. Raises error if there are any inconsistenties in dimension of the data"""

    data = (X, cluster_centers, labels)
    valid_shapes = [{3}, {2, 3}, {2, 1}]
    names = ('X', 'cluster_centers', 'labels') 

    for i, j, k in zip(data, valid_shapes, names):
        if i is not None:
            if i.ndim not in j:
                raise TypeError(f"Invalid ndim. Expected {k}'s dimension to be any of {j} but got {i.ndim}")

    if cluster_centers is not None and labels is not None:
        if cluster_centers.shape[-2] != len(np.unique(labels)):
            raise ValueError(f"Number of clusters in cluster_centers and labels are not the same, they are {cluster_centers.shape[-2]} and {len(np.unique(labels))} respectively")
        elif cluster_centers.ndim == 3 and labels.ndim == 2 and cluster_centers.shape[0] != labels.shape[1]:
            raise ValueError(f"Number of timesteps in cluster_centers and labels are not the same, they are {cluster_centers.shape[0]} and {labels.shape[1]} respectively")

    if cluster_centers is not None and X is not None:
        if cluster_centers.shape[-1] != X.shape[-1]:
            raise ValueError(f"Number of features in cluster_centers and input data (X) are not the same, they are {cluster_centers.shape[-1]} and {X.shape[-1]} respectively")
        elif cluster_centers.ndim == 3 and cluster_centers.shape[0] != X.shape[0]:
            raise ValueError(f"Number of timesteps in cluster_centers and input data (X) are not the same, they are {cluster_centers.shape[0]} and {X.shape[0]} respectively")
     
    if labels is not None and X is not None:
        if labels.shape[0] != X.shape[1]:
            raise ValueError(f"Number of entities in labels and input data (X) are not the same, they are {labels.shape[0]} and {X.shape[1]} respectively")
        elif labels.ndim == 2 and labels.shape[1] != X.shape[0]:
            raise ValueError(f"Number of timesteps in labels and input data (X) are not the same, they are {labels.shape[1]} and {X.shape[0]} respectively")

def _get_shape(
        X: npt.NDArray[np.float64]|None = None, 
        cluster_centers: npt.NDArray[np.float64]|None = None, 
        labels: npt.NDArray[np.float64]|None = None 
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int]]:

    """
    Function to return the shape of the input data
    """

    if X is not None:
        X_shape = X.shape
    else:
        X_shape = (0, 0, 0)

    if cluster_centers is not None:
        if cluster_centers.ndim == 3:
            cc_shape = cluster_centers.shape 
        elif cluster_centers.ndim == 2:
            cc_shape = (0, *cluster_centers.shape)
    else:
        cc_shape = (0, 0, 0)

    if labels is not None:
        if labels.ndim == 2:
            l_shape = labels.shape 
        elif labels.ndim == 1:
            l_shape = (labels.shape[0], 0)
    else:
        l_shape = (0, 0)

    return X_shape, cc_shape, l_shape

def plot(
        *,
        X: npt.NDArray[np.float64]|None = None, 
        cluster_centers: npt.NDArray[np.float64]|None = None, 
        labels: npt.NDArray[np.float64]|None = None, 
        entity_idx: List[int]|None = None,
        entities_labels: List[str]|None = None,
        label_dict: dict|None = None,
        annot_fontsize: float|int = 10,
        show_all_entities: bool = True,
        figsize: Tuple[float, float] | None = None,
        shape_of_subplot: Tuple[int, int]|None = None, 
        xlabel: str|None = 'timesteps', 
        ylabel: str|None = 'value',
        cluster_labels: List[str]|None = None,
        title_list: List[str]|None = None,
        show_all_xticklabels: bool = True, 
        x_rotation: float|int = 45,
        show_X_marker: bool = False,
        show_cluster_center_marker: bool = False,
        ) -> Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]:
    
    """
    Function to plot subplots of the time series data, cluster centers and label assignemnts. One subplot per feature. This is built on top of matplotlib.


    Parameters
    ----------
    X : numpy array, default=None
        The time series data in TNF format.
    cluster_centers : numpy array, default=None
        If numpy array, it is expected to be a 3D in TNF format. Here, N is the number of clusters. 
        If 2-D array, then it is interpreted as a K x F array where K is the number of clusters, and F is the number of features. Suitable for fixed cluster centers clustering.
    labels : numpy array, default=None
        It is expected to be a 2D array of shape (N, T) . Where N is the number of entities and T is the number of time steps. The value of the ith row at the t-th column is the label (cluster index) entity i was assigned to at time t.
        If 1-D array, it is interpreted as an array of length N. Where N is the number of entities. In such case, the i-th element is the cluster the i-th entity was assigned to across all time steps. Suitable for fixed assignment clustering.
    entity_idx : list, default=None 
        list of index of entities to display in the plot. If `show_all_entities` is True, `entity_idx` will be interpreted as the index of entities to be annonated.
    entities_labels : list, default=None
        list of labels for annotating the entities in `entity_idx`. If None, then labels of `entity_idx` in `label_dict` are used.
    label_dict dict, default=None
        a dictionary whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
        - 'T' is a list of names/labels of each time step to be used as index of each dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
        - 'N' is a list of names/labels of each entity. If None, range(0, N) is used. Where N is the number of entities/observations in the fitted data 
        - 'F' is a list of names/labels of each feature to be used as column of each dataframe. If None, range(0, F) is used. Where F is the number of features in the fitted data 
        If label_dict is not None, it is used to label timestep, entities, and features in the plot.   
    annot_fontsize : float|int, default=10
        The font size to be used for annotating entities in `entity_idx`.
    show_all_entities : bool, default=True
        If True, displays all the entities in `X` in the plot. If False, only entities in `entity_idx` are plotted.
    figsize : tuple, default=None
        The size of the figure. Should be tuple of length 2, where the first value is width of the figure, while the second value is the height of the figure.
    shape_of_subplot : tuple, default=None
        The shape of the subplots. Should be tuple of length 2, where the first value is number of rows, while the second value is the of columns in the figure.
        If None, (F, 1) is used. Where F is the number of features in `X` or `cluster_centers`.
    xlabel : str, default='timesteps'
        The label for the x-axis of each subplots
    ylabel : str, default='timesteps'
        The label for the y-axis of each subplots
    cluster_labels : list, default=None
        The labels to use for the clusters. If None, list(range(K)) is used. Where `K` is the number of clusters in `cluster_centers` or `labels`.
    title_list : list, default=None
        The title of each subplot (or feature).
    show_all_xticklabels : bool, default=True
        If True, shows labels of all time steps in the plot. If False, some may be suppressed (depending on the size of the plot)
    x_rotation : float or int, default=45
        The angle (in degrees) to rotate the timestep labels 
    show_X_marker : bool, default=False
        If True, show markers in the time series plot of X.
    show_cluster_center_marker : bool, default=False
        If True, show markers in the time series plot of the cluster centers.

    Returns
    -------
    matplotlib.figure.Figure
        the matplotlib figure object
    list
        a list of the matplotlib axes, one axes per subplot. 
    """
    
    _data_validator(X=X, cluster_centers=cluster_centers, labels=labels)

    X_shape, cc_shape, l_shape = _get_shape(X=X, cluster_centers=cluster_centers, labels=labels)

    # determine T. Should be the longest in X, cluster_centers and labels. this is to allow for variable number of input variables ie X, cluster_centers and labels
    T = max(X_shape[0], cc_shape[0], l_shape[1])
    N = max(X_shape[1], l_shape[0])
    F = max(X_shape[2], cc_shape[2])
    K = cc_shape[1]
    
    if labels is not None:
        K = max(K, len(np.unique(labels)))

    K = max(K, 1)

    label_dict_init = {'T': T, 'N': N, 'F': F}

    if label_dict is None:
        label_dict = {}
    
    for key, val in label_dict_init.items():
            _ = label_dict.setdefault(key, list(range(val)))

    cmap = pl.cm.get_cmap('rainbow')

    # broadcast cluster centers and labels if need be. This is done at this point because we need to compute T before now
    cluster_centers, labels = broadcast_data(T, cluster_centers=cluster_centers, labels=labels)

    norm = plt.Normalize(vmin=0, vmax=K-1)
    
    if cluster_labels is None:
        cluster_labels = list(map(str, range(K)))

    # determine the number of feature

    fig = plt.figure(figsize=figsize)

    if shape_of_subplot is None:
        shape_of_subplot = (F, 1)

    if title_list is None:
        title_list = ['Feature ' + str(f+1) if isinstance(f, int) else 'Feature ' + f for f in label_dict['F']]

    axes = []

    for f in range(F):
        ax = fig.add_subplot(*shape_of_subplot, f+1)

        if X is not None:
            if show_all_entities:
                idx = np.arange(X.shape[1])
            else:
                idx = entity_idx

            marker = ''
            if show_X_marker:
                marker = '.'

            # plot all data for feature f
            plt.plot(range(X.shape[0]), X[:, idx, f], c='k', ls='--', alpha=0.5, marker=marker)

            if entity_idx is not None:
                for li, i in enumerate(entity_idx):
                    if entities_labels is None:
                        e_labels = label_dict['N'][i]
                    else:
                        e_labels = entities_labels[li]

                    annot_i = np.random.choice(np.arange(len(X[:, i, f])), 1)[0]
                    annot_xy = list(enumerate(X[:, i, f]))[annot_i]

                    plt.annotate(e_labels, xy=annot_xy, xytext=(annot_xy[0]+0.5, annot_xy[1]+0.5), fontsize=annot_fontsize,
                                arrowprops=dict(facecolor='green',shrink=0))
                    
            if labels is not None:
            # scatter plot for marker for label assignment of data points. 
                for i in idx:
                    c = labels[i] 
                    plt.scatter(range(X.shape[0]), X[:, i, f], color=cmap(norm(c)), s=10)

                # label legend for cluster centers to match that of label assignment
                if cluster_centers is None:
                    for j in range(K):
                        plt.plot([], [], color=cmap(norm(j)), label=cluster_labels[j])  

        # plot of cluster centers
        if cluster_centers is not None:

            marker = ''
            if show_cluster_center_marker:
                marker = '.'

            for j in range(K):
                plt.plot(
                    range(cluster_centers.shape[0]), 
                    cluster_centers[:, j, f], 
                    color=cmap(norm(j)), 
                    label=cluster_labels[j],
                    marker=marker
                    )

        ax.set_xlabel(xlabel)

        ax.set_ylabel(ylabel)
        
        ax.set_title(title_list[f])
        
        ax.set_xticks(ticks=list(range(T))) # needed so that xticks wouldn't be float
        if show_all_xticklabels:
            ax.set_xticklabels(label_dict['T'], rotation=x_rotation)  

        if f != F-1:
            ax.set_xlabel('')
            ax.set_xticklabels('')
        
        if cluster_centers is not None or labels is not None:
            plt.legend()

        axes.append(ax)

    return fig, axes

def waterfall_plot(
        time_series: npt.NDArray[np.float64],
        label_dict: dict|None = None,
        *,
        xlabel: str = 'time-axis',
        ylabel: str = 'Features-axis',
        zlabel: str = 'Feature Values',
        title: str|None = None
        ) -> Tuple[matplotlib.figure.Figure, mpl_toolkits.mplot3d.axes3d.Axes3D]:
    
    """
    Function to plot a waterfall plot of a single time series data. This data can be a time series of a single entity or cluster center.
    To make the plot interactive, use a suitable matplotlib's magic command. E.g. `%matplotlib widget`. See this site for more: https://matplotlib.org/stable/users/explain/figure/interactive.html

    Parameters
    ----------
    time_series : numpy array
        The time series data to plot. This data can be a time series of a single entity or cluster center.
        Data should be a 2-D array of shape (T, F), where T and F are the number of timesteps and features respectively.
    label_dict : dict, default=None
        a dictionary whose keys are 'T', 'N', and 'F' (which are the number of time steps, entities, and features respectively). Value of each key is a list such that the value of key:
        - 'T' is a list of names/labels of each time step to be used as index of each dataframe. If None, range(0, T) is used. Where T is the number of time steps in the fitted data
        - 'N' (ignored) is a list of names/labels of each entity. If None, range(0, N) is used. Where N is the number of entities/observations in the fitted data 
        - 'F' is a list of names/labels of each feature to be used as column of each dataframe. If None, range(0, F) is used. Where F is the number of features in the fitted data 
        If label_dict is not None, it is used to label timestep and features in the plot. 
    xlabel : str, default='time-axis'
        The label of the x-axis (time axis)
    ylabel : str, default='Features-axis'
        The label of the y-axis (feature axis)
    zlabel : str, default='Feature Values'
        The label of the z-axis (axis for the values of the features)
    title : str, default=None
        The title of the plot

    Returns
    -------
    matplotlib.figure.Figure
        the matplotlib figure object
    mpl_toolkits.mplot3d.axes3d.Axes3D
        the matplotlib's 3-D axes object. 
    """

    x = np.arange(time_series.shape[0]) # timesteps
    y = np.arange(time_series.shape[1]) # features
    
    X, Y = np.meshgrid(x, y)

    X, Y = X.T, Y.T

    # Creating a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotting the basic 3D surface
    ax.plot_surface(X, Y, time_series, color='grey', alpha=0.9)
    
    T, F = time_series.shape
    
    for f in range(F):
        ax.plot(x, [f]*T, time_series[:, f], color='red')
    
    label_dict_init = {'T': T, 'F': F}

    if label_dict is None:
        label_dict = {}
    
    for key, val in label_dict_init.items():
            _ = label_dict.setdefault(key, list(range(1, val+1)))

    x_tick_labels = label_dict['T']
    y_tick_labels = label_dict['F']

    if title is None:
        title = ''

    # Customizing the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.set_xticks(x, x_tick_labels)
    ax.set_yticks(y, y_tick_labels)
    
    # Create sliders for elevation and azimuth angles
    ax_elev = plt.axes([0.1, 0.05, 0.8, 0.02])
    ax_azim = plt.axes([0.1, 0.02, 0.8, 0.02])
    
    def _update(val):
        ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
        fig.canvas.draw_idle()
        
    slider_elev = Slider(ax_elev, 'Elevation', 0, 360, valinit=45)
    slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=45)
    
    # Connect sliders to update function
    slider_elev.on_changed(_update)
    slider_azim.on_changed(_update)
    
    # Displaying the plot
    return fig, ax