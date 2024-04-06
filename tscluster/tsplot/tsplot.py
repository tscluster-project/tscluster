from __future__ import annotations
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.widgets import Slider

from tscluster.preprocessing.utils import broadcast_data

def _data_validator(
        X: npt.NDArray[np.float64]|None = None, 
        cluster_centers: npt.NDArray[np.float64]|None = None, 
        labels: npt.NDArray[np.float64]|None = None        
    ) -> None:

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
        annot_fontsize: float = 10,
        show_all_entities: bool = True,
        figsize: Tuple[float, float] | None = None,
        shape_of_subplot: Tuple[int, int]|None = None, 
        xlabel: str|None = 'timesteps', 
        ylabel: str|None = 'val',
        cluster_labels: List[str]|None = None,
        title_list: List[str]|None = None,
        set_xticklabels: List[str]|None = None, 
        rotation: float|int = 45, 
        # out_parent_dir, data_file, year_labels=None, 
        ) -> None:
    
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

    # Fs = np.arange(F)

    # if feature_idx is not None:
    #     Fs = feature_idx

    # entities_labels = label_dict['N']

    for f in range(F):
        ax = fig.add_subplot(*shape_of_subplot, f+1)

        if X is not None:
            if show_all_entities:
                idx = np.arange(X.shape[1])
            else:
                idx = entity_idx

            # plot all data for feature f
            plt.plot(range(X.shape[0]), X[:, idx, f], c='k', ls='--', alpha=0.5)

            if entities_labels is not None:
                for li, i in enumerate(entity_idx):
                    annot_i = np.random.choice(np.arange(len(X[:, i, f])), 1)[0]
                    annot_xy = list(enumerate(X[:, i, f]))[annot_i]
                    plt.annotate(entities_labels[li], xy=annot_xy, xytext=(annot_xy[0]+0.5, annot_xy[1]+0.5), fontsize=annot_fontsize,
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
            for j in range(K):
                plt.plot(range(cluster_centers.shape[0]), cluster_centers[:, j, f], color=cmap(norm(j)), label=cluster_labels[j])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_xticks(ticks=list(range(X.shape[0])))
        
        if title_list is not None:
            ax.set_title(title_list[f])
        
        if set_xticklabels is not None:
            ax.set_xticklabels(set_xticklabels, rotation=rotation)  

        if cluster_centers is not None or labels is not None:
            plt.legend()

# Update function for sliders

def waterfall_plot(
        time_series: npt.NDArray[np.float64],
        label_dict: dict|None = None,
        *,
        xlabel: str = 'time-axis',
        ylabel: str = 'Features-axis',
        zlabel: str = 'Feature Values',
        title: str = 'Basic 3D Surface Plot'
        ):
    
    x = np.arange(time_series.shape[0]) # timesteps
    y = np.arange(time_series.shape[1]) # features
    
    X, Y = np.meshgrid(x, y)

    # Creating a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotting the basic 3D surface
    ax.plot_surface(X, Y, time_series.T.values, rstride=y.shape[0], cstride=x.shape[0]-1, color='grey', alpha=0.9)
    
    for f in range(time_series.shape[1]):
        ax.plot(x, [f]*x.shape[0], time_series.iloc[:, f], color='red')
    
    if label_dict is None:
        x_tick_labels = time_series.index
        y_tick_labels = time_series.columns
    else:
        for k in label_dict:
            if k is not None and k == 'T':
                x_tick_labels = label_dict[k]
            elif k is not None and k == 'F':
                y_tick_labels = label_dict[k]

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