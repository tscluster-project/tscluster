Installation Guide
==================

Requirements
------------
We require Python 3.8+.

* numpy>=1.26 
* scipy>=1.10 
* gurobipy>=11.0 
* tslearn>=0.6.3   
* h5py>=3.10
* pandas>=2.2
* matplotlib>=3.8

Note: you will need Gurobi licence when using OptTSCluster with large model size. See `here <https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license>`_ for more about Gurobi licence

Installing via pip
-----------------
.. code-block:: shell

    pip install tscluster

Installing the Pre-Release Version via git
---------
.. code-block:: shell

    pip install git+https://github.com/tscluster-project/tscluster.git