from __future__ import annotations
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.widgets import Slider

from tscluster.preprocessing.utils import broadcast_data

# def _data_validator(
#         X: npt.NDArray[np.float64]|None = None, 
#         cluster_centers: npt.NDArray[np.float64]|None = None, 
#         labels: npt.NDArray[np.float64]|None = None        
# ):
#     if cluster_centers is not None and labels is not None:
#         if cluster_centers.ndim == 3 and labels.

def plot(
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
    
    
    cmap = pl.cm.get_cmap('rainbow')
        
    # determine T. Should be the longest in X, cluster_centers and labels. this is to allow for variable timesteps for X, ad cluster centers
    T = 1 
    F = 0 

    if X is not None:

        # X, _label_dict = get_inferred_data(X)

        # if label_dict is None:
        #     label_dict = _label_dict

        # if X.shape[0] > T:
        T = X.shape[0]
        F = X.shape[2] # determine the number of features

    if cluster_centers is not None:
        if cluster_centers.shape[-1] > F:
            F = cluster_centers.shape[-1] # determine the number of features

        if cluster_centers.ndim == 3 and cluster_centers.shape[0] > T:
            T = cluster_centers.shape[0]

        if label_dict is None:
            label_dict = {}
            label_dict['T'] = list(range(T))
            label_dict['F'] = list(range(F))

    # in case cluster_centers was not passed
    if labels is not None:
        if labels.ndim == 2 and labels.shape[1] > T:
            T = labels.shape[1]

        if label_dict is None:
            label_dict = {}
            label_dict['T'] = list(range(T))

    # broadcast cluster centers and labels if need be
    cluster_centers, labels = broadcast_data(T, cluster_centers=cluster_centers, labels=labels)

    # determine number of clusters
    k = 1 # assumes all data belongs to one cluster 

    if cluster_centers is not None:
        k = cluster_centers.shape[1]

    # in case cluster_centers was not passed
    if labels is not None:
        if np.unique(labels).shape[0] > k:
            k = np.unique(labels).shape[0]

    norm = plt.Normalize(vmin=0, vmax=k-1)
    
    if cluster_labels is None:
        cluster_labels = list(map(str, range(k)))


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
                    c = labels[i] #np.argmax(out[2][-1][:, i, :], axis=-1)
                    plt.scatter(range(X.shape[0]), X[:, i, f], color=cmap(norm(c)), s=10)

                if cluster_centers is None:
                    for j in range(k):
                        plt.plot([], [], color=cmap(norm(j)), label=cluster_labels[j])  

        # plot of cluster centers

        if cluster_centers is not None:
            for j in range(k):
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