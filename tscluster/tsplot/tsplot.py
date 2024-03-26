from __future__ import annotations
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from tscluster.preprocessing.utils import broadcast_data, _get_inferred_data

class tsplot():
    def __init__(self):
        pass 

    @staticmethod
    def plot(
            X: npt.NDArray[np.float64]|str|List|None = None, 
            cluster_centers: npt.NDArray[np.float64]|None = None, 
            labels: npt.NDArray[np.float64]|None = None, 
            entity_idx: List[int]|None = None,
            entities_labels: List[int]|None = None,
            annot_fontsize: float = 10,
            show_all_entities: bool = True,
            figsize: Tuple[float, float] | None = None,
            shape_of_subplot: Tuple[int, int]|None = None, 
            xlabel: str|None = 'timesteps', 
            cluster_labels: List[str]|None = None,
            title_list: List[str]|None = None,
            set_xticklabels: List[str]|None = None, 
            rotation: float|int = 45, 
            # out_parent_dir, data_file, year_labels=None, 
            ) -> None:
        # out = []
        # out_keys = ["Es_star", "Zs_ans", "Cs_hats", "I_adds", "zis", "Z", "warm_start_labels_"]
        
        # for out_k in out_keys:
        #     out.append(np.load(out_parent_dir+"\\"+out_k+"_"+str(idx)+".npy", allow_pickle=True))


        # X = np.load(data_file)
        
        cmap = pl.cm.get_cmap('rainbow')
        
        # summary_df = pd.read_csv(out_parent_dir+"\\summary_"+str(idx)+".csv.gz")
        
        # determine T. Should be the longest in X, cluster_centers and labels. this is to allow for variable timesteps for X, ad cluster centers
        T = 1 
        F = 0 

        if X is not None:

            X = _get_inferred_data(None, X)

            # if X.shape[0] > T:
            T = X.shape[0]
            F = X.shape[2] # determine the number of features

        if cluster_centers is not None:
            if cluster_centers.shape[-1] > F:
                F = cluster_centers.shape[-1] # determine the number of features

            if cluster_centers.ndim == 3 and cluster_centers.shape[0] > T:
                T = cluster_centers.shape[0]

        # in case cluster_centers was not passed
        if labels is not None:
            if labels.ndim == 2 and labels.shape[1] > T:
                T = labels.shape[1]
    
        # broadcast cluster centers and labels if need be
        cluster_centers, labels = broadcast_data(T, cluster_centers, labels)

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
            
            # # if len(out[1][-1].shape) == 2:
            # #     for j in range(out[1][-1].shape[0]):
            # #         plt.plot(range(X.shape[0]), [out[1][-1][j, f]]*X.shape[0], color=cmap(norm(j)))
            # elif len(out[1][-1].shape) == 3:   

            # plot of cluster centers

            if cluster_centers is not None:
                for j in range(k):
                    plt.plot(range(cluster_centers.shape[0]), cluster_centers[:, j, f], color=cmap(norm(j)), label=cluster_labels[j])

            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"val")
            # ax.set_xticks(ticks=list(range(X.shape[0])))
            
            if title_list is not None:
                ax.set_title(title_list[f])
            
            if set_xticklabels is not None:
                ax.set_xticklabels(set_xticklabels, rotation=rotation)  

            if cluster_centers is not None or labels is not None:
                plt.legend()

        plt.show()