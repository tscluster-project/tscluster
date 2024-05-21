Introduction
============

Overview
--------

Temporal clustering is a popular unsupervised machine-learning task with applications to datasets including census, finance, and healthcare data that is used to group time series data into different groups according to common temporal trends.

In this tscluster open-source toolbox, we provide a range of methods for temporal clustering that include both traditional and novel methods for temporal clustering as we illustrate below:

.. image:: source/images/venn_diagram.png

As shown above, existing methods of temporal clustering in literature fall under two categories:

- **Time Series Clustering (TSC)** [*Aghabozorgi et al, 2014*]: On the above left is shown an example of TSC which involves grouping time series (as multidimensional vectors) based on similarity metrics (e.g. euclidean distance). While cluster centers change over time, the cluster labels for each entity remain constant.  For example, one can identify similar groups of stocks by clustering them on their daily price data.
- **Sequence Labelling Analysis (SLA)** [*Delmelle et al, 2016*]: On the above right is shown an example of SLA that assumes a constant (non-changing) cluster center, but which allows for the cluster labels of each entity to change over time unlike TSC.  For example, SLA could be used to identify trends in gentrification as indicated by a census tract transitioning from a low income and high unemployment cluster label to a high income and low unemployment cluster label.

While tscluster supports both TSC and SLA in a common framework, it also provides novel combinations of these methods (e.g., allowing both dynamic cluster centers and cluster labels) as we outline next.

.. image:: source/images/table_schemes.png 

In this table, we organize all clustering methods according to two choices:

1. In the rows we can choose to have either static (unchanging) cluster centers or dynamic (changing) cluster centers over time.
2. In the columns we can choose how labels are allowed to change over time: static (no label change), unbounded (unlimited label change), or bounded (an upper limit on the number of label changes allowed).  

Perhaps one of the most important novel tools in tscluster is specifically the capability to perform Bounded Fully Dynamic clustering (middle bottom), which allows us to identify the (anomalous) entities that diverge most from existing dynamic trends.  As an example use case for census analysis, we can identify census tracts that change due to external forces (e.g., significant rezoning) by using tscluster's Bounded Fully Dynamic clustering scheme to constrain the number of cluster changes.

Purpose and Benefits
--------------------
With tscluster, you can:

- Use opttscluster subpackage to cluster temporal data using any combination of static or fixed cluster labels and centers with optimality guarantees underscored by Mixed Linear Integer Programming.

- Use opttscluster subpackage to find entities that are most likely to change cluster label assignment if a total number of n label changes are allowed.

Tscluster also encompasses the two existing approaches by proving the following classes available in its tskmeans subpackage:
- TSKmeans class for TSC (builds on tslearn).
- TSGlobalKmeans class for SLA (builds on sklearn).

Tscluster implemented some utility tools in the following subpackages to help in temporal clustering tasks.
- preprocessing: This can be used to preprocess and load temporal data. Data can be loaded from either a directory, a file, a list of Pandas DataFrames, or a numpy array
file (.npy).

.. image:: source/images/tscluster_schema.png

- metric: contains useful temporal clustering evaluation metrics such as inertia and max_dist.
- tsplot: Useful for seamlessly generating 2D time series plots and 3D waterfall plots of all features within temporal data and the cluster centers. 

.. With tscluster, you can:

.. - Cluster time series data with optimality guarantees using ``opttscluster`` subpackage with any of the six shemes in the design space introduced in this paper. 
.. - Use k-means for time series clustering using the ``TSKmeans`` class (built on top of ``tslean``) in the ``tskmeans`` subpackage.
.. - Do time-series label analysis (TLA) clustering using the  ``TSGlobalKmeans`` class in the ``tskmeans`` subpackage.
.. - Preprocess time series data using its ``preprocessing`` subpackage.
.. - Evaluate clustering alogrithms using ``metrics`` subpackage.
.. - Seamlessly generate plots of the time series and their cluster using its ``tsplot`` subpackage. 

License
-------
This software is distributed under the MIT License.

Contributors
------------
- Jolomi Tosanwumi (University of Toronto, CA)
- Jiazhou Liang (University of Toronto, CA)