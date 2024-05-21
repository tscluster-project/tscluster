Tutorial with Synthetic Data
============================

This is an example notebook with overview of the usage of the modules in tscluster.

.. code-block:: python

    !pip install tscluster # install tscluster 

.. code-block:: python

    ## Importing Libraries

.. code-block:: python

    # uncomment the line below if widget is enable in your environment. This is useful for making tsplot's waterfall_plot interactive
    # %matplotlib widget
    
    import os
    
    import numpy as np
    import pandas as pd
    import requests
    
    from tscluster.opttscluster import OptTSCluster
    from tscluster.tskmeans import TSKmeans, TSGlobalKmeans
    from tscluster.preprocessing import TSStandardScaler, TSMinMaxScaler
    from tscluster.preprocessing.utils import load_data, tnf_to_ntf, ntf_to_tnf, to_dfs, broadcast_data
    from tscluster.metrics import inertia, max_dist
    from tscluster.tsplot import tsplot

.. code-block:: python

    par_dir = "tscluster_sample_data"

.. code-block:: python

    # download the sample data
    !wget https://raw.githubusercontent.com/tscluster-project/tscluster/main/test/tscluster_sample_data.zip
    
    if not os.path.exists(par_dir):
            os.makedirs(par_dir)
    
    !unzip tscluster_sample_data.zip -d tscluster_sample_data


.. parsed-literal::

    --2024-04-09 00:09:18--  https://raw.githubusercontent.com/tscluster-project/tscluster/main/test/tscluster_sample_data.zip
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 11197 (11K) [application/zip]
    Saving to: ‘tscluster_sample_data.zip.4’
    
    
    tscluster   0%[                    ]       0  --.-KB/s               
    tscluster_sample_da 100%[===================>]  10.93K  --.-KB/s    in 0s      
    
    2024-04-09 00:09:19 (47.5 MB/s) - ‘tscluster_sample_data.zip.4’ saved [11197/11197]
    
    Archive:  tscluster_sample_data.zip

.. code-block:: python

    os.chdir(par_dir)

Loading Data
------------

from a npy file
~~~~~~~~~~~~~~~

If data is a numpy array stored as a ``.npy`` file, you can use the
``load_data`` function to load it.

.. code-block:: python

    X, label_dict = load_data("./sythetic_data.npy")
    X.shape




.. parsed-literal::

    (10, 15, 1)



The ``load_data`` function returns a tuple, the first value of the tuple
is the loaded data (a 3-D array in ‘TNF’ format), while the second value
of the tuple is the label_dict of the data. The ``label_dict`` is a
dictionary whose keys are ‘T’, ‘N’, and ‘F’ (which are the number of
time steps, entities, and features respectively). Value of each key is a
list such that the value of key: - ‘T’ is a list of names/labels of each
time step to be used as index of each dataframe. If None, range(0, T) is
used. Where T is the number of time steps in the fitted data - ‘N’
(ignored) is a list of names/labels of each entity. If None, range(0, N)
is used. Where N is the number of entities/observations in the fitted
data - ‘F’ is a list of names/labels of each feature to be used as
column of each dataframe. If None, range(0, F) is used. Where F is the
number of features in the fitted data

.. code-block:: python

    print(label_dict)


.. parsed-literal::

    {'T': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'N': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 'F': [0]}
    

from a list
~~~~~~~~~~~

Data can also be loaded from a list. This can be a list of 2-D numpy
arrays, or list of pandas dataframes, or list of file paths. By default,
the list is of length ``T`` (number of time steps), where each element
of the list is interpreted as a data for all entities at a particular
time step. Set the ``arr_format`` parameter to ‘NTF’ to specify that
each element of the input list is the time series data for a particular
entity for all time steps. Valid files are ``.npy``, ``.npz``,
``.json``, ``xlsx``, ``.csv`` or any file readable by
``pandas.read_csv`` function.

Reading from a list of dataframes

.. code-block:: python

    df1 = pd.DataFrame({
        'f1': np.arange(5),
        'f2': np.arange(5, 10)
    }, index=['e'+str(i+1) for i in range(5)]
                      )
    df1




.. raw:: html

    
      <div id="df-71d5e55e-670f-4979-9cfd-9172065c732c" class="colab-df-container">
        <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>f1</th>
          <th>f2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>e1</th>
          <td>0</td>
          <td>5</td>
        </tr>
        <tr>
          <th>e2</th>
          <td>1</td>
          <td>6</td>
        </tr>
        <tr>
          <th>e3</th>
          <td>2</td>
          <td>7</td>
        </tr>
        <tr>
          <th>e4</th>
          <td>3</td>
          <td>8</td>
        </tr>
        <tr>
          <th>e5</th>
          <td>4</td>
          <td>9</td>
        </tr>
      </tbody>
    </table>
    </div>
        <div class="colab-df-buttons">
    
      <div class="colab-df-container">
        <button class="colab-df-convert" onclick="convertToInteractive('df-71d5e55e-670f-4979-9cfd-9172065c732c')"
                title="Convert this dataframe to an interactive table."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
        <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
      </svg>
        </button>
    
      <style>
        .colab-df-container {
          display:flex;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        .colab-df-buttons div {
          margin-bottom: 4px;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
        <script>
          const buttonEl =
            document.querySelector('#df-71d5e55e-670f-4979-9cfd-9172065c732c button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-71d5e55e-670f-4979-9cfd-9172065c732c');
            const dataTable =
              await google.colab.kernel.invokeFunction('convertToInteractive',
                                                        [key], {});
            if (!dataTable) return;
    
            const docLinkHtml = 'Like what you see? Visit the ' +
              '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
              + ' to learn more about interactive tables.';
            element.innerHTML = '';
            dataTable['output_type'] = 'display_data';
            await google.colab.output.renderOutput(dataTable, element);
            const docLink = document.createElement('div');
            docLink.innerHTML = docLinkHtml;
            element.appendChild(docLink);
          }
        </script>
      </div>
    
    
    <div id="df-dbcce90d-9074-46ec-815a-98008f3c3b8a">
      <button class="colab-df-quickchart" onclick="quickchart('df-dbcce90d-9074-46ec-815a-98008f3c3b8a')"
                title="Suggest charts"
                style="display:none;">
    
    <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
         width="24px">
        <g>
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
        </g>
    </svg>
      </button>
    
    <style>
      .colab-df-quickchart {
          --bg-color: #E8F0FE;
          --fill-color: #1967D2;
          --hover-bg-color: #E2EBFA;
          --hover-fill-color: #174EA6;
          --disabled-fill-color: #AAA;
          --disabled-bg-color: #DDD;
      }
    
      [theme=dark] .colab-df-quickchart {
          --bg-color: #3B4455;
          --fill-color: #D2E3FC;
          --hover-bg-color: #434B5C;
          --hover-fill-color: #FFFFFF;
          --disabled-bg-color: #3B4455;
          --disabled-fill-color: #666;
      }
    
      .colab-df-quickchart {
        background-color: var(--bg-color);
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: var(--fill-color);
        height: 32px;
        padding: 0;
        width: 32px;
      }
    
      .colab-df-quickchart:hover {
        background-color: var(--hover-bg-color);
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: var(--button-hover-fill-color);
      }
    
      .colab-df-quickchart-complete:disabled,
      .colab-df-quickchart-complete:disabled:hover {
        background-color: var(--disabled-bg-color);
        fill: var(--disabled-fill-color);
        box-shadow: none;
      }
    
      .colab-df-spinner {
        border: 2px solid var(--fill-color);
        border-color: transparent;
        border-bottom-color: var(--fill-color);
        animation:
          spin 1s steps(1) infinite;
      }
    
      @keyframes spin {
        0% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
          border-left-color: var(--fill-color);
        }
        20% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        30% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
          border-right-color: var(--fill-color);
        }
        40% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        60% {
          border-color: transparent;
          border-right-color: var(--fill-color);
        }
        80% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-bottom-color: var(--fill-color);
        }
        90% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
        }
      }
    </style>
    
      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-dbcce90d-9074-46ec-815a-98008f3c3b8a button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
    
      <div id="id_8ab1e4af-6d53-4503-9233-15f70b639c0f">
        <style>
          .colab-df-generate {
            background-color: #E8F0FE;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: none;
            fill: #1967D2;
            height: 32px;
            padding: 0 0 0 0;
            width: 32px;
          }
    
          .colab-df-generate:hover {
            background-color: #E2EBFA;
            box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
            fill: #174EA6;
          }
    
          [theme=dark] .colab-df-generate {
            background-color: #3B4455;
            fill: #D2E3FC;
          }
    
          [theme=dark] .colab-df-generate:hover {
            background-color: #434B5C;
            box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
            filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
            fill: #FFFFFF;
          }
        </style>
        <button class="colab-df-generate" onclick="generateWithVariable('df1')"
                title="Generate code using this dataframe."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
           width="24px">
        <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
      </svg>
        </button>
        <script>
          (() => {
          const buttonEl =
            document.querySelector('#id_8ab1e4af-6d53-4503-9233-15f70b639c0f button.colab-df-generate');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          buttonEl.onclick = () => {
            google.colab.notebook.generateWithVariable('df1');
          }
          })();
        </script>
      </div>
    
        </div>
      </div>
    



.. code-block:: python

    df2 = pd.DataFrame({
        'f2': np.arange(105, 110),
        'f1': np.arange(100, 105)
    }, index=['e'+str(i+1) for i in range(5)]
                      )
    df2




.. raw:: html

    
      <div id="df-43c5ac55-2300-4e56-ab4a-9ec5ce0aac03" class="colab-df-container">
        <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>f2</th>
          <th>f1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>e1</th>
          <td>105</td>
          <td>100</td>
        </tr>
        <tr>
          <th>e2</th>
          <td>106</td>
          <td>101</td>
        </tr>
        <tr>
          <th>e3</th>
          <td>107</td>
          <td>102</td>
        </tr>
        <tr>
          <th>e4</th>
          <td>108</td>
          <td>103</td>
        </tr>
        <tr>
          <th>e5</th>
          <td>109</td>
          <td>104</td>
        </tr>
      </tbody>
    </table>
    </div>
        <div class="colab-df-buttons">
    
      <div class="colab-df-container">
        <button class="colab-df-convert" onclick="convertToInteractive('df-43c5ac55-2300-4e56-ab4a-9ec5ce0aac03')"
                title="Convert this dataframe to an interactive table."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
        <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
      </svg>
        </button>
    
      <style>
        .colab-df-container {
          display:flex;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        .colab-df-buttons div {
          margin-bottom: 4px;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
        <script>
          const buttonEl =
            document.querySelector('#df-43c5ac55-2300-4e56-ab4a-9ec5ce0aac03 button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-43c5ac55-2300-4e56-ab4a-9ec5ce0aac03');
            const dataTable =
              await google.colab.kernel.invokeFunction('convertToInteractive',
                                                        [key], {});
            if (!dataTable) return;
    
            const docLinkHtml = 'Like what you see? Visit the ' +
              '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
              + ' to learn more about interactive tables.';
            element.innerHTML = '';
            dataTable['output_type'] = 'display_data';
            await google.colab.output.renderOutput(dataTable, element);
            const docLink = document.createElement('div');
            docLink.innerHTML = docLinkHtml;
            element.appendChild(docLink);
          }
        </script>
      </div>
    
    
    <div id="df-48608e4f-fedd-4531-a26c-507cf8135a72">
      <button class="colab-df-quickchart" onclick="quickchart('df-48608e4f-fedd-4531-a26c-507cf8135a72')"
                title="Suggest charts"
                style="display:none;">
    
    <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
         width="24px">
        <g>
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
        </g>
    </svg>
      </button>
    
    <style>
      .colab-df-quickchart {
          --bg-color: #E8F0FE;
          --fill-color: #1967D2;
          --hover-bg-color: #E2EBFA;
          --hover-fill-color: #174EA6;
          --disabled-fill-color: #AAA;
          --disabled-bg-color: #DDD;
      }
    
      [theme=dark] .colab-df-quickchart {
          --bg-color: #3B4455;
          --fill-color: #D2E3FC;
          --hover-bg-color: #434B5C;
          --hover-fill-color: #FFFFFF;
          --disabled-bg-color: #3B4455;
          --disabled-fill-color: #666;
      }
    
      .colab-df-quickchart {
        background-color: var(--bg-color);
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: var(--fill-color);
        height: 32px;
        padding: 0;
        width: 32px;
      }
    
      .colab-df-quickchart:hover {
        background-color: var(--hover-bg-color);
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: var(--button-hover-fill-color);
      }
    
      .colab-df-quickchart-complete:disabled,
      .colab-df-quickchart-complete:disabled:hover {
        background-color: var(--disabled-bg-color);
        fill: var(--disabled-fill-color);
        box-shadow: none;
      }
    
      .colab-df-spinner {
        border: 2px solid var(--fill-color);
        border-color: transparent;
        border-bottom-color: var(--fill-color);
        animation:
          spin 1s steps(1) infinite;
      }
    
      @keyframes spin {
        0% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
          border-left-color: var(--fill-color);
        }
        20% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        30% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
          border-right-color: var(--fill-color);
        }
        40% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        60% {
          border-color: transparent;
          border-right-color: var(--fill-color);
        }
        80% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-bottom-color: var(--fill-color);
        }
        90% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
        }
      }
    </style>
    
      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-48608e4f-fedd-4531-a26c-507cf8135a72 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
    
      <div id="id_06391235-c312-497c-935b-d0731a25df8a">
        <style>
          .colab-df-generate {
            background-color: #E8F0FE;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: none;
            fill: #1967D2;
            height: 32px;
            padding: 0 0 0 0;
            width: 32px;
          }
    
          .colab-df-generate:hover {
            background-color: #E2EBFA;
            box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
            fill: #174EA6;
          }
    
          [theme=dark] .colab-df-generate {
            background-color: #3B4455;
            fill: #D2E3FC;
          }
    
          [theme=dark] .colab-df-generate:hover {
            background-color: #434B5C;
            box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
            filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
            fill: #FFFFFF;
          }
        </style>
        <button class="colab-df-generate" onclick="generateWithVariable('df2')"
                title="Generate code using this dataframe."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
           width="24px">
        <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
      </svg>
        </button>
        <script>
          (() => {
          const buttonEl =
            document.querySelector('#id_06391235-c312-497c-935b-d0731a25df8a button.colab-df-generate');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          buttonEl.onclick = () => {
            google.colab.notebook.generateWithVariable('df2');
          }
          })();
        </script>
      </div>
    
        </div>
      </div>
    



.. code-block:: python

    X_arr, label_dict = load_data([df1, df2])
    print(f"shape of X_arr is {X_arr.shape}")
    X_arr = X_arr.astype(np.float64)
    X_arr


.. parsed-literal::

    shape of X_arr is (2, 5, 2)
    



.. parsed-literal::

    array([[[  0.,   5.],
            [  1.,   6.],
            [  2.,   7.],
            [  3.,   8.],
            [  4.,   9.]],
    
           [[100., 105.],
            [101., 106.],
            [102., 107.],
            [103., 108.],
            [104., 109.]]])



.. code-block:: python

    label_dict




.. parsed-literal::

    {'T': [0, 1], 'N': ['e1', 'e2', 'e3', 'e4', 'e5'], 'F': ['f1', 'f2']}



To get the output in ‘NTF’ format, set the ``output_arr_format``
parameter to ‘NTF’

.. code-block:: python

    X_arr, label_dict = load_data([df1, df2], output_arr_format='NTF')
    print(f"shape of X_arr is {X_arr.shape}")
    X_arr


.. parsed-literal::

    shape of X_arr is (5, 2, 2)

    array([[[  0.,   5.],
            [100., 105.]],
    
           [[  1.,   6.],
            [101., 106.]],
    
           [[  2.,   7.],
            [102., 107.]],
    
           [[  3.,   8.],
            [103., 108.]],
    
           [[  4.,   9.],
            [104., 109.]]])



.. code-block:: python

    label_dict # label_dict will remain the same




.. parsed-literal::

    {'T': [0, 1], 'N': ['e1', 'e2', 'e3', 'e4', 'e5'], 'F': ['f1', 'f2']}



The same applies to list of file paths. E.g.

.. code-block:: python

    file_list = [
        "./synthetic_csv/timestep_0.csv",
        "./synthetic_csv/timestep_1.csv",
        "./synthetic_csv/timestep_2.csv",
        "./synthetic_csv/timestep_3.csv",
        "./synthetic_csv/timestep_4.csv"
    ]
    
    X_arr, label_dict = load_data(file_list)
    print(f"shape of X_arr is {X_arr.shape}")


.. parsed-literal::

    shape of X_arr is (5, 20, 2)
    

.. code-block:: python

    label_dict




.. parsed-literal::

    {'T': [0, 1, 2, 3, 4],
     'N': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
     'F': [0, 1]}



You can also pass arguments to the file reader used by using the
``read_file_args`` parameter. This parameter accepts a dictionary where
the keys are the names of the file reader parameters (in string), and
the values are the values of the file reader parameter. E.g. if file
reader is pd.read_csv (reader for csv file), you can pass ``names`` and
``skiprows`` arguments (and basically any argument you want to pass to
the file reader).

.. code-block:: python

    file_list = [
        "./synthetic_csv/timestep_0.csv",
        "./synthetic_csv/timestep_1.csv",
        "./synthetic_csv/timestep_2.csv",
        "./synthetic_csv/timestep_3.csv",
        "./synthetic_csv/timestep_4.csv"
    ]
    
    X_arr, label_dict = load_data(file_list, read_file_args={'names': ['x1', 'x2'], 'skiprows': 10})
    print(f"shape of X_arr is {X_arr.shape}")


.. parsed-literal::

    shape of X_arr is (5, 10, 2)
    

.. code-block:: python

    label_dict




.. parsed-literal::

    {'T': [0, 1, 2, 3, 4], 'N': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'F': ['x1', 'x2']}



from a directory
~~~~~~~~~~~~~~~~

You can instead pass a directory path (as a string) to the ``load_data``
function. In this case, the suffix (not file extension) of the filenames
will be used for ordering the files before loading them as different
timesteps. The suffix consists of characters after ``suffix_sep`` (not
including file extension). The default value for ``suffix_sep`` is an
undescore “\_“. E.g. if the ‘synthetic_csv’ directory contains the
following files:

-  timestep_1.csv
-  timestep_2.csv
-  timestep_3.csv
-  timestep_4.csv

We can read the files as follows:

.. code-block:: python

    X_arr, label_dict = load_data('./synthetic_csv')
    print(f"shape of X_arr is {X_arr.shape}")


.. parsed-literal::

    shape of X_arr is (5, 20, 2)
    

.. code-block:: python

    label_dict




.. parsed-literal::

    {'T': [0, 1, 2, 3, 4],
     'N': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
     'F': [0, 1]}



The suffixes of the filenames may not neccessarily start from 1 or have
an interval of 1. For example, the filenames could be:

-  year-2000.csv
-  year-2005.csv
-  year-2010.csv
-  year-2015.csv
-  year-2020.csv

So long the suffixes can be sorted and there is a consistent suffix
separator (“-” is this case), the directory can be parsed by
``load_data`` function.

.. code-block:: python

    # checking how the head of a single
    pd.read_csv('./synthetic_csv2/year-2005.csv').head()




.. raw:: html

    
      <div id="df-bdcb377c-6a54-4c0f-a83b-2e67b9a99c49" class="colab-df-container">
        <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Unnamed: 0</th>
          <th>x1</th>
          <th>x2</th>
          <th>x3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>i1</td>
          <td>1.144403</td>
          <td>1.384766</td>
          <td>-0.296697</td>
        </tr>
        <tr>
          <th>1</th>
          <td>i2</td>
          <td>-0.221455</td>
          <td>-2.379010</td>
          <td>1.616871</td>
        </tr>
        <tr>
          <th>2</th>
          <td>i3</td>
          <td>1.533177</td>
          <td>-1.650524</td>
          <td>-0.548531</td>
        </tr>
        <tr>
          <th>3</th>
          <td>i4</td>
          <td>-0.615204</td>
          <td>0.794567</td>
          <td>-0.726242</td>
        </tr>
        <tr>
          <th>4</th>
          <td>i5</td>
          <td>0.622818</td>
          <td>-0.129735</td>
          <td>-0.723215</td>
        </tr>
      </tbody>
    </table>
    </div>
        <div class="colab-df-buttons">
    
      <div class="colab-df-container">
        <button class="colab-df-convert" onclick="convertToInteractive('df-bdcb377c-6a54-4c0f-a83b-2e67b9a99c49')"
                title="Convert this dataframe to an interactive table."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
        <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
      </svg>
        </button>
    
      <style>
        .colab-df-container {
          display:flex;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        .colab-df-buttons div {
          margin-bottom: 4px;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
        <script>
          const buttonEl =
            document.querySelector('#df-bdcb377c-6a54-4c0f-a83b-2e67b9a99c49 button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-bdcb377c-6a54-4c0f-a83b-2e67b9a99c49');
            const dataTable =
              await google.colab.kernel.invokeFunction('convertToInteractive',
                                                        [key], {});
            if (!dataTable) return;
    
            const docLinkHtml = 'Like what you see? Visit the ' +
              '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
              + ' to learn more about interactive tables.';
            element.innerHTML = '';
            dataTable['output_type'] = 'display_data';
            await google.colab.output.renderOutput(dataTable, element);
            const docLink = document.createElement('div');
            docLink.innerHTML = docLinkHtml;
            element.appendChild(docLink);
          }
        </script>
      </div>
    
    
    <div id="df-8f194f28-74e0-4e3a-a6f3-454ee1d3f727">
      <button class="colab-df-quickchart" onclick="quickchart('df-8f194f28-74e0-4e3a-a6f3-454ee1d3f727')"
                title="Suggest charts"
                style="display:none;">
    
    <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
         width="24px">
        <g>
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
        </g>
    </svg>
      </button>
    
    <style>
      .colab-df-quickchart {
          --bg-color: #E8F0FE;
          --fill-color: #1967D2;
          --hover-bg-color: #E2EBFA;
          --hover-fill-color: #174EA6;
          --disabled-fill-color: #AAA;
          --disabled-bg-color: #DDD;
      }
    
      [theme=dark] .colab-df-quickchart {
          --bg-color: #3B4455;
          --fill-color: #D2E3FC;
          --hover-bg-color: #434B5C;
          --hover-fill-color: #FFFFFF;
          --disabled-bg-color: #3B4455;
          --disabled-fill-color: #666;
      }
    
      .colab-df-quickchart {
        background-color: var(--bg-color);
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: var(--fill-color);
        height: 32px;
        padding: 0;
        width: 32px;
      }
    
      .colab-df-quickchart:hover {
        background-color: var(--hover-bg-color);
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: var(--button-hover-fill-color);
      }
    
      .colab-df-quickchart-complete:disabled,
      .colab-df-quickchart-complete:disabled:hover {
        background-color: var(--disabled-bg-color);
        fill: var(--disabled-fill-color);
        box-shadow: none;
      }
    
      .colab-df-spinner {
        border: 2px solid var(--fill-color);
        border-color: transparent;
        border-bottom-color: var(--fill-color);
        animation:
          spin 1s steps(1) infinite;
      }
    
      @keyframes spin {
        0% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
          border-left-color: var(--fill-color);
        }
        20% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        30% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
          border-right-color: var(--fill-color);
        }
        40% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        60% {
          border-color: transparent;
          border-right-color: var(--fill-color);
        }
        80% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-bottom-color: var(--fill-color);
        }
        90% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
        }
      }
    </style>
    
      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-8f194f28-74e0-4e3a-a6f3-454ee1d3f727 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
    
        </div>
      </div>
    



.. code-block:: python

    # if we were to indicate to pandas that the first column is the index and the first row is the header, we would have done
    pd.read_csv('./synthetic_csv2/year-2005.csv', index_col=[0], header=0).head()




.. raw:: html

    
      <div id="df-28c57e14-8baf-4df7-9a7d-058784391b06" class="colab-df-container">
        <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>x1</th>
          <th>x2</th>
          <th>x3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>i1</th>
          <td>1.144403</td>
          <td>1.384766</td>
          <td>-0.296697</td>
        </tr>
        <tr>
          <th>i2</th>
          <td>-0.221455</td>
          <td>-2.379010</td>
          <td>1.616871</td>
        </tr>
        <tr>
          <th>i3</th>
          <td>1.533177</td>
          <td>-1.650524</td>
          <td>-0.548531</td>
        </tr>
        <tr>
          <th>i4</th>
          <td>-0.615204</td>
          <td>0.794567</td>
          <td>-0.726242</td>
        </tr>
        <tr>
          <th>i5</th>
          <td>0.622818</td>
          <td>-0.129735</td>
          <td>-0.723215</td>
        </tr>
      </tbody>
    </table>
    </div>
        <div class="colab-df-buttons">
    
      <div class="colab-df-container">
        <button class="colab-df-convert" onclick="convertToInteractive('df-28c57e14-8baf-4df7-9a7d-058784391b06')"
                title="Convert this dataframe to an interactive table."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
        <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
      </svg>
        </button>
    
      <style>
        .colab-df-container {
          display:flex;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        .colab-df-buttons div {
          margin-bottom: 4px;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
        <script>
          const buttonEl =
            document.querySelector('#df-28c57e14-8baf-4df7-9a7d-058784391b06 button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-28c57e14-8baf-4df7-9a7d-058784391b06');
            const dataTable =
              await google.colab.kernel.invokeFunction('convertToInteractive',
                                                        [key], {});
            if (!dataTable) return;
    
            const docLinkHtml = 'Like what you see? Visit the ' +
              '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
              + ' to learn more about interactive tables.';
            element.innerHTML = '';
            dataTable['output_type'] = 'display_data';
            await google.colab.output.renderOutput(dataTable, element);
            const docLink = document.createElement('div');
            docLink.innerHTML = docLinkHtml;
            element.appendChild(docLink);
          }
        </script>
      </div>
    
    
    <div id="df-b6bbce95-cf5c-4b71-8247-362ef99bfb03">
      <button class="colab-df-quickchart" onclick="quickchart('df-b6bbce95-cf5c-4b71-8247-362ef99bfb03')"
                title="Suggest charts"
                style="display:none;">
    
    <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
         width="24px">
        <g>
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
        </g>
    </svg>
      </button>
    
    <style>
      .colab-df-quickchart {
          --bg-color: #E8F0FE;
          --fill-color: #1967D2;
          --hover-bg-color: #E2EBFA;
          --hover-fill-color: #174EA6;
          --disabled-fill-color: #AAA;
          --disabled-bg-color: #DDD;
      }
    
      [theme=dark] .colab-df-quickchart {
          --bg-color: #3B4455;
          --fill-color: #D2E3FC;
          --hover-bg-color: #434B5C;
          --hover-fill-color: #FFFFFF;
          --disabled-bg-color: #3B4455;
          --disabled-fill-color: #666;
      }
    
      .colab-df-quickchart {
        background-color: var(--bg-color);
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: var(--fill-color);
        height: 32px;
        padding: 0;
        width: 32px;
      }
    
      .colab-df-quickchart:hover {
        background-color: var(--hover-bg-color);
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: var(--button-hover-fill-color);
      }
    
      .colab-df-quickchart-complete:disabled,
      .colab-df-quickchart-complete:disabled:hover {
        background-color: var(--disabled-bg-color);
        fill: var(--disabled-fill-color);
        box-shadow: none;
      }
    
      .colab-df-spinner {
        border: 2px solid var(--fill-color);
        border-color: transparent;
        border-bottom-color: var(--fill-color);
        animation:
          spin 1s steps(1) infinite;
      }
    
      @keyframes spin {
        0% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
          border-left-color: var(--fill-color);
        }
        20% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        30% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
          border-right-color: var(--fill-color);
        }
        40% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        60% {
          border-color: transparent;
          border-right-color: var(--fill-color);
        }
        80% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-bottom-color: var(--fill-color);
        }
        90% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
        }
      }
    </style>
    
      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-b6bbce95-cf5c-4b71-8247-362ef99bfb03 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
    
        </div>
      </div>
    



.. code-block:: python

    # using load_data function
    X_arr, label_dict = load_data('./synthetic_csv2',
                                  suffix_sep='-',
                                  use_suffix_as_label=True,
                                  read_file_args={'index_col': [0], 'header': 0})
    print(f"shape of X_arr is {X_arr.shape}")


.. parsed-literal::

    shape of X_arr is (5, 10, 3)
    

.. code-block:: python

    print(label_dict)


.. parsed-literal::

    {'T': ['2000', '2005', '2010', '2015', '2020'], 'N': ['i1', 'i10', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9'], 'F': ['x1', 'x2', 'x3']}
    

Data Conversion
---------------

to_dfs
~~~~~~

We can convert a 3-D array to a list of dataframes using the ``to_dfs``
function. This is basically the reverse process of ``load_dict`` in that
it takes a 3-D array and an optional label_dict, and returns a list of
dataframes. Similar to ``load_dict`` function, you can use
``arr_format`` and ``output_df_format`` to specify the format of the
input data and output data respectively.

.. code-block:: python

    dfs = to_dfs(X_arr, label_dict)
    print(f"Length of dfs is: {len(dfs)}")
    dfs[0].head() # first five rows of the first dataframe in the list


.. parsed-literal::

    Length of dfs is: 5
    



.. raw:: html

    
      <div id="df-a2e91af5-2a55-4b99-a019-edb01d352d25" class="colab-df-container">
        <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>x1</th>
          <th>x2</th>
          <th>x3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>i1</th>
          <td>0.496714</td>
          <td>-0.138264</td>
          <td>-0.291524</td>
        </tr>
        <tr>
          <th>i10</th>
          <td>0.097078</td>
          <td>0.968645</td>
          <td>0.626228</td>
        </tr>
        <tr>
          <th>i2</th>
          <td>-0.463418</td>
          <td>-0.465730</td>
          <td>-0.312976</td>
        </tr>
        <tr>
          <th>i3</th>
          <td>1.465649</td>
          <td>-0.225776</td>
          <td>0.488591</td>
        </tr>
        <tr>
          <th>i4</th>
          <td>-0.601707</td>
          <td>1.852278</td>
          <td>-0.078235</td>
        </tr>
      </tbody>
    </table>
    </div>
        <div class="colab-df-buttons">
    
      <div class="colab-df-container">
        <button class="colab-df-convert" onclick="convertToInteractive('df-a2e91af5-2a55-4b99-a019-edb01d352d25')"
                title="Convert this dataframe to an interactive table."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
        <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
      </svg>
        </button>
    
      <style>
        .colab-df-container {
          display:flex;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        .colab-df-buttons div {
          margin-bottom: 4px;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
        <script>
          const buttonEl =
            document.querySelector('#df-a2e91af5-2a55-4b99-a019-edb01d352d25 button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-a2e91af5-2a55-4b99-a019-edb01d352d25');
            const dataTable =
              await google.colab.kernel.invokeFunction('convertToInteractive',
                                                        [key], {});
            if (!dataTable) return;
    
            const docLinkHtml = 'Like what you see? Visit the ' +
              '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
              + ' to learn more about interactive tables.';
            element.innerHTML = '';
            dataTable['output_type'] = 'display_data';
            await google.colab.output.renderOutput(dataTable, element);
            const docLink = document.createElement('div');
            docLink.innerHTML = docLinkHtml;
            element.appendChild(docLink);
          }
        </script>
      </div>
    
    
    <div id="df-e00afac6-51df-46a3-a780-981754ddddb0">
      <button class="colab-df-quickchart" onclick="quickchart('df-e00afac6-51df-46a3-a780-981754ddddb0')"
                title="Suggest charts"
                style="display:none;">
    
    <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
         width="24px">
        <g>
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
        </g>
    </svg>
      </button>
    
    <style>
      .colab-df-quickchart {
          --bg-color: #E8F0FE;
          --fill-color: #1967D2;
          --hover-bg-color: #E2EBFA;
          --hover-fill-color: #174EA6;
          --disabled-fill-color: #AAA;
          --disabled-bg-color: #DDD;
      }
    
      [theme=dark] .colab-df-quickchart {
          --bg-color: #3B4455;
          --fill-color: #D2E3FC;
          --hover-bg-color: #434B5C;
          --hover-fill-color: #FFFFFF;
          --disabled-bg-color: #3B4455;
          --disabled-fill-color: #666;
      }
    
      .colab-df-quickchart {
        background-color: var(--bg-color);
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: var(--fill-color);
        height: 32px;
        padding: 0;
        width: 32px;
      }
    
      .colab-df-quickchart:hover {
        background-color: var(--hover-bg-color);
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: var(--button-hover-fill-color);
      }
    
      .colab-df-quickchart-complete:disabled,
      .colab-df-quickchart-complete:disabled:hover {
        background-color: var(--disabled-bg-color);
        fill: var(--disabled-fill-color);
        box-shadow: none;
      }
    
      .colab-df-spinner {
        border: 2px solid var(--fill-color);
        border-color: transparent;
        border-bottom-color: var(--fill-color);
        animation:
          spin 1s steps(1) infinite;
      }
    
      @keyframes spin {
        0% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
          border-left-color: var(--fill-color);
        }
        20% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        30% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
          border-right-color: var(--fill-color);
        }
        40% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        60% {
          border-color: transparent;
          border-right-color: var(--fill-color);
        }
        80% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-bottom-color: var(--fill-color);
        }
        90% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
        }
      }
    </style>
    
      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-e00afac6-51df-46a3-a780-981754ddddb0 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
    
        </div>
      </div>
    



tnf_to_ntf
~~~~~~~~~~

``tnf_to_ntf`` function can be used to convert a data from ‘TNF’ format
to ‘NTF’ format. E.g

.. code-block:: python

    print(f"Shape of X_arr in 'TNF' format is: {X_arr.shape}")
    
    X_arr_ntf = tnf_to_ntf(X_arr)
    
    print(f"Shape of X_arr in 'NTF' format is: {X_arr_ntf.shape}")


.. parsed-literal::

    Shape of X_arr in 'TNF' format is: (5, 10, 3)
    Shape of X_arr in 'NTF' format is: (10, 5, 3)
    

ntf_to_tnf
~~~~~~~~~~

Similarly, ``ntf_to_tnf`` function can be used to convert from ‘NTF’
format to ‘TNF’ format. E.g.

.. code-block:: python

    print(f"Shape of X_arr in 'NTF' format is: {X_arr_ntf.shape}")
    
    print(f"Shape of X_arr in 'TNF' format is: {ntf_to_tnf(X_arr_ntf).shape}")


.. parsed-literal::

    Shape of X_arr in 'NTF' format is: (10, 5, 3)
    Shape of X_arr in 'TNF' format is: (5, 10, 3)
    

broadcast_data
~~~~~~~~~~~~~~

If you want to broadcast a fixed cluster center along the time axis, you
can use ``broadcast_data`` function. E.g. if you have fixed cluster
centers as a 2-D array of shape (K, F), where K is the number of
clusters and F is the number of features; you can convert it to a 3-D
array such that the first axis is the time axis. This is usefule
especially when dealing with fixed center or fixed assignment because
they return (for memory efficiency) a 2-D array and a 1-D array
respectively.

.. code-block:: python

    np.random.seed(0)
    cluster_centers = np.random.randn(3, 2)
    cluster_centers




.. parsed-literal::

    array([[ 1.76405235,  0.40015721],
           [ 0.97873798,  2.2408932 ],
           [ 1.86755799, -0.97727788]])



.. code-block:: python

    T = 3 # number of time steps
    cluster_centers_broadcasted, _ = broadcast_data(T, cluster_centers=cluster_centers)
    cluster_centers_broadcasted




.. parsed-literal::

    array([[[ 1.76405235,  0.40015721],
            [ 0.97873798,  2.2408932 ],
            [ 1.86755799, -0.97727788]],
    
           [[ 1.76405235,  0.40015721],
            [ 0.97873798,  2.2408932 ],
            [ 1.86755799, -0.97727788]],
    
           [[ 1.76405235,  0.40015721],
            [ 0.97873798,  2.2408932 ],
            [ 1.86755799, -0.97727788]]])



You can also broadcast labels. E.g if the cluster labels is a 1-D numpy
array of shape (N, ).

.. code-block:: python

    np.random.seed(2)
    labels = np.random.choice([0, 1, 2], 10)
    labels




.. parsed-literal::

    array([0, 1, 0, 2, 2, 0, 2, 1, 1, 2])



.. code-block:: python

    T = 3 # number of time steps
    _, labels_broadcasted = broadcast_data(T, labels=labels)
    labels_broadcasted




.. parsed-literal::

    array([[0, 0, 0],
           [1, 1, 1],
           [0, 0, 0],
           [2, 2, 2],
           [2, 2, 2],
           [0, 0, 0],
           [2, 2, 2],
           [1, 1, 1],
           [1, 1, 1],
           [2, 2, 2]])



You can also broadcast both cluster_centers and labels at the same time

.. code-block:: python

    T = 3 # number of time steps
    cluster_centers_broadcasted, labels_broadcasted = broadcast_data(T, cluster_centers=cluster_centers, labels=labels)
    cluster_centers_broadcasted




.. parsed-literal::

    array([[[ 1.76405235,  0.40015721],
            [ 0.97873798,  2.2408932 ],
            [ 1.86755799, -0.97727788]],
    
           [[ 1.76405235,  0.40015721],
            [ 0.97873798,  2.2408932 ],
            [ 1.86755799, -0.97727788]],
    
           [[ 1.76405235,  0.40015721],
            [ 0.97873798,  2.2408932 ],
            [ 1.86755799, -0.97727788]]])



.. code-block:: python

    labels_broadcasted




.. parsed-literal::

    array([[0, 0, 0],
           [1, 1, 1],
           [0, 0, 0],
           [2, 2, 2],
           [2, 2, 2],
           [0, 0, 0],
           [2, 2, 2],
           [1, 1, 1],
           [1, 1, 1],
           [2, 2, 2]])



Preprocessing
-------------

The ``preprocessing`` module has two main scalers: ``TSStandardScaler``
and ``TSMinMaxScaler``.

TSStandardScaler
~~~~~~~~~~~~~~~~

This scaler uses sklearn’s StandardScaler to scale a time series data.
Scaling can be done per timesteps (default) or per feature

Using ``fit`` and ``transform`` methods. During ``fit``, the scaler
parameters are stored. They will be used for ``tranform`` and
``inverse-tansform`` of data.

.. code-block:: python

    scaler = TSStandardScaler(per_time=True) # initialize a time series standard scaler
    scaler.fit(X_arr) # fit
    X_scaled = scaler.fit_transform(X_arr) # transform
    print(f"X_scaled shape is {X_scaled.shape}")
    print()
    print("First five entities for the first time step are:")
    print(X_scaled[0, :5, :])


.. parsed-literal::

    X_scaled shape is (5, 10, 3)
    
    First five entities for the first time step are:
    [[ 0.53075651 -0.62117007 -0.2344527 ]
     [-0.12234591  0.79082039  0.91426627]
     [-1.03833007 -1.03889002 -0.26130345]
     [ 2.11422893 -0.73280172  0.74199078]
     [-1.26432746  1.91799644  0.03251411]]
    

``fit`` and ``transform`` can be done with a single method called
``fit_transform``. E.g.

.. code-block:: python

    scaler = TSStandardScaler(per_time=True) # initialize a time series standard scaler
    X_scaled = scaler.fit_transform(X_arr) # fit and transform at the same time
    print(f"X_scaled shape is {X_scaled.shape}")
    print()
    print("First five entities for the first time step are:")
    print(X_scaled[0, :5, :])


.. parsed-literal::

    X_scaled shape is (5, 10, 3)
    
    First five entities for the first time step are:
    [[ 0.53075651 -0.62117007 -0.2344527 ]
     [-0.12234591  0.79082039  0.91426627]
     [-1.03833007 -1.03889002 -0.26130345]
     [ 2.11422893 -0.73280172  0.74199078]
     [-1.26432746  1.91799644  0.03251411]]
    

We can use ``inverse-tranform`` method to reverse the transformation.

.. code-block:: python

    print("First five entities for the first time step of the original data are:")
    print(X_arr[0, :5, :])
    print()
    print("First five entities for the first time step of the inverse tranform of X_scaled are:")
    print(scaler.inverse_transform(X_scaled)[0, :5, :])


.. parsed-literal::

    First five entities for the first time step of the original data are:
    [[ 0.49671415 -0.1382643  -0.29152375]
     [ 0.09707755  0.96864499  0.62622751]
     [-0.46341769 -0.46572975 -0.31297574]
     [ 1.46564877 -0.2257763   0.48859067]
     [-0.60170661  1.85227818 -0.07823474]]
    
    First five entities for the first time step of the inverse tranform of X_scaled are:
    [[ 0.49671415 -0.1382643  -0.29152375]
     [ 0.09707755  0.96864499  0.62622751]
     [-0.46341769 -0.46572975 -0.31297574]
     [ 1.46564877 -0.2257763   0.48859067]
     [-0.60170661  1.85227818 -0.07823474]]
    

TSMinMaxScaler
~~~~~~~~~~~~~~

The same methods of ``TSStandardScaler`` applies to ``TSMinMaxScaler``

This scaler uses sklearn’s MinMaxScaler to scale a time series data.
Scaling can be done per timesteps (default) or per feature

Using ``fit`` and ``transform`` methods.

During ``fit``, the scaler parameters are stored. They will be used for
``tranform`` and ``inverse-tansform`` of data.

.. code-block:: python

    scaler = TSMinMaxScaler(per_time=True) # initialize a time series minmax scaler
    scaler.fit(X_arr) # fit
    X_scaled = scaler.fit_transform(X_arr) # transform
    print(f"X_scaled shape is {X_scaled.shape}")
    print()
    print("First five entities for the first time step are:")
    print(X_scaled[0, :5, :])


.. parsed-literal::

    X_scaled shape is (5, 10, 3)
    
    First five entities for the first time step are:
    [[0.53131686 0.1412702  0.40094123]
     [0.33800873 0.6187963  0.75951472]
     [0.0668917  0.         0.39255975]
     [1.         0.1035171  0.7057388 ]
     [0.         1.         0.48427512]]
    

``fit`` and ``transform`` can be done with a single method called
``fit_transform``. E.g.

.. code-block:: python

    scaler = TSMinMaxScaler(per_time=True) # initialize a time series minmax scaler
    X_scaled = scaler.fit_transform(X_arr) # fit and transform at the same time
    print(f"X_scaled shape is {X_scaled.shape}")
    print()
    print("First five entities for the first time step are:")
    print(X_scaled[0, :5, :])


.. parsed-literal::

    X_scaled shape is (5, 10, 3)
    
    First five entities for the first time step are:
    [[0.53131686 0.1412702  0.40094123]
     [0.33800873 0.6187963  0.75951472]
     [0.0668917  0.         0.39255975]
     [1.         0.1035171  0.7057388 ]
     [0.         1.         0.48427512]]
    

We can use ``inverse-tranform`` method to reverse the transformation.

.. code-block:: python

    print("First five entities for the first time step of the original data are:")
    print(X_arr[0, :5, :])
    print()
    print("First five entities for the first time step of the inverse tranform of X_scaled are:")
    print(scaler.inverse_transform(X_scaled)[0, :5, :])


.. parsed-literal::

    First five entities for the first time step of the original data are:
    [[ 0.49671415 -0.1382643  -0.29152375]
     [ 0.09707755  0.96864499  0.62622751]
     [-0.46341769 -0.46572975 -0.31297574]
     [ 1.46564877 -0.2257763   0.48859067]
     [-0.60170661  1.85227818 -0.07823474]]
    
    First five entities for the first time step of the inverse tranform of X_scaled are:
    [[ 0.49671415 -0.1382643  -0.29152375]
     [ 0.09707755  0.96864499  0.62622751]
     [-0.46341769 -0.46572975 -0.31297574]
     [ 1.46564877 -0.2257763   0.48859067]
     [-0.60170661  1.85227818 -0.07823474]]
    

Metrics
-------

There are currently two metrics in ``tscluster`` package: ``inertia``
and ``max_dist``.

The inertia is calculated as:

.. math::


       \sum_{t=1}^{T} \sum_{i=1}^{N} D(X_{ti}, Z_t)

Where - :math:`T`, :math:`N` are the number of time steps and entities
respectively - :math:`D` is a distance function (or metric e.g
:math:`L_1` distance, :math:`L_2` distance etc) - :math:`f` is the
number of features - :math:`X_{ti} \in \mathbf{R}^f` is the feature
vector of entity :math:`i` at time :math:`t` -
:math:`Z_t \in \mathbf{R}^f` is the cluster center :math:`X_{ti}` is
assigned to at time :math:`t`

The max_dist is calculated as:

.. math::


       max(D(X_{ti}, Z_t))

Where - :math:`D` is a distance function (or metric e.g
:math:`L_1`\ distance, :math:`L_2` distance etc) - :math:`f` is the
number of features - :math:`X_{ti} \in \mathbf{R}^f` is the feature
vector of entity :math:`i` at time :math:`t`, -
:math:`Z_t \in \mathbf{R}^f` is the cluster center :math:`X_{ti}` is
assigned to at time :math:`t`.

Both ``inertia`` and ``max_dist`` functions take four arguments: 1. The
data X (in TNF format) 2. cluster_centers 3. labels 4. ord (which
specifies the order of the Minkowski distance)

They can also take both 3-D and 2-D arrays for dynamic and fixed cluster
centers respectively, and 2-D and 1-D arrays for dynamic and fixed
labels respectively.

.. code-block:: python

    # using fixed cluster centers and dynamic label assignment
    np.random.seed(0)
    cluster_centers = np.random.randn(3, X_arr.shape[2]) # 2-D array (for fixed cluster)
    
    np.random.seed(2)
    labels = np.random.choice([0, 1, 2], (X_arr.shape[1], X_arr.shape[0])) # 2-D array (for dynamic labels)

.. code-block:: python

    print(f"inertia score is {inertia(X_arr, cluster_centers, labels, ord=1)}") # using l1 distance
    print(f"max_dist score is {max_dist(X_arr, cluster_centers, labels, ord=1)}") # using l1 distance


.. parsed-literal::

    inertia score is 217.22127047719061
    max_dist score is 10.202923513064336
    

.. code-block:: python

    # using dynamic cluster centers and fixed label assignment
    np.random.seed(0)
    cluster_centers = np.random.randn(X_arr.shape[0], 3, X_arr.shape[2]) # 3-D array (for dynamic cluster)
    
    np.random.seed(2)
    labels = np.random.choice([0, 1, 2], X_arr.shape[1]) # 1-D array (for fixed labels)
    labels




.. parsed-literal::

    array([0, 1, 0, 2, 2, 0, 2, 1, 1, 2])



.. code-block:: python

    print(f"inertia score is {inertia(X_arr, cluster_centers, labels, ord=2)}") # using l2 distance
    print(f"max_dist score is {max_dist(X_arr, cluster_centers, labels, ord=2)}") # using l2 distance


.. parsed-literal::

    inertia score is 138.29240897541072
    max_dist score is 7.3157146070128745
    

TSPlot
------

plot
~~~~

``plot`` function is used to plot a time series plots of the different
features in a time series data

.. code-block:: python

    fig, ax = tsplot.plot(X=X_arr)



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_89_0.png


We can add label assignment to the plot

.. code-block:: python

    fig, ax = tsplot.plot(X=X_arr, labels=labels)



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_91_0.png


We can plot only cluster centers

.. code-block:: python

    fig, ax = tsplot.plot(cluster_centers=cluster_centers)



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_93_0.png


We can plot all of data X, cluster centers and label assignment in the
same plot

.. code-block:: python

    fig, ax = tsplot.plot(X=X_arr, cluster_centers=cluster_centers, labels=labels)
    # note that the cluster centers are not meaningfull since they were randomly generated



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_95_0.png


We can also annotate only specific entities by passing their index to
the ``entity_idx`` parameter

.. code-block:: python

    fig, ax = tsplot.plot(X=X_arr, cluster_centers=cluster_centers, labels=labels, entity_idx=[0, 4, 9])



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_97_0.png


We can show only the entities in ``entity_idx`` by setting
``show_all_entities`` to False

.. code-block:: python

    fig, ax = tsplot.plot(X=X_arr, cluster_centers=cluster_centers, labels=labels, entity_idx=[0, 4, 9], show_all_entities=False)



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_99_0.png


We can use the labels in label_dict to label the entities in
``entity_idx`` by passing ``label_dict``

.. code-block:: python

    # recall our label dict
    label_dict




.. parsed-literal::

    {'T': ['2000', '2005', '2010', '2015', '2020'],
     'N': ['i1', 'i10', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9'],
     'F': ['x1', 'x2', 'x3']}



.. code-block:: python

    fig, ax = tsplot.plot(
        X=X_arr,
        cluster_centers=cluster_centers,
        labels=labels,
        entity_idx=[0, 4, 9],
        show_all_entities=False,
        label_dict=label_dict
    )



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_102_0.png


We can pass custom labels to the labels in ``entity_idx`` using the
``entities_labels`` parameter.

.. code-block:: python

    fig, ax = tsplot.plot(
        X=X_arr,
        cluster_centers=cluster_centers,
        labels=labels,
        entity_idx=[0, 4, 9],
        entities_labels=['e0', 'e4', 'e9'],
        show_all_entities=False
    )



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_104_0.png


We can also pass custom labels for the cluster centers using the
``cluster_labels`` parameter

.. code-block:: python

    fig, ax = tsplot.plot(
        X=X_arr,
        cluster_centers=cluster_centers,
        labels=labels,
        entity_idx=[0, 4, 9],
        entities_labels=['e0', 'e4', 'e9'],
        show_all_entities=False,
        label_dict=label_dict,
        cluster_labels=['C1', 'C2', 'C3']
    )



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_106_0.png


waterfall_plot
~~~~~~~~~~~~~~

``waterfall_plot`` can be used to generate a 3-D time series plot of a
particular entity or cluster center.

To make the plot interactive, use a suitable matplotlib’s magic command.
E.g. ``%matplotlib widget``. See this site for more:
https://matplotlib.org/stable/users/explain/figure/interactive.html

.. code-block:: python

    # waterfall plot of a single entity
    idx = 0
    fig, ax = tsplot.waterfall_plot(X_arr[:, idx, :])



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_109_0.png


.. code-block:: python

    # waterfall plot of a single cluster center
    idx = 0
    fig, ax = tsplot.waterfall_plot(cluster_centers[:, idx, :])



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_110_0.png


Temporal Clustering Models
--------------------------

All temporal clustering modules implements a ``fit`` method (in which on
executing, compute the cluster centers and label assignments).

We can use the ``cluster_centers_`` and ``labels_`` attributes to
retreive the cluster centers and label assignments respectively. Here we
used sklearn’s convention of using trailing underscores for attributes
whose values are known only after fitting.

OptTSCluster
~~~~~~~~~~~~

**fixed centers, dynamic assignment**

.. code-block:: python

    # initialize the model
    opt_ts = OptTSCluster(
        k=3,
        scheme='z0c1', # fixed centers, dynamic assignment
        n_allow_assignment_change=None # number of changes to allow, None means allow as many changes as possible
        # warm_start=True # warm start with kmeans
    )

.. code-block:: python

    model_size = opt_ts.get_model_size(X_arr)
    print(f"model has {model_size[0]} variables and {model_size[1]} constraints")


.. parsed-literal::

    Restricted license - for non-production use only - expires 2025-11-24
    model has 610 variables and 950 constraints
    

.. code-block:: python

    label_dict




.. parsed-literal::

    {'T': ['2000', '2005', '2010', '2015', '2020'],
     'N': ['i1', 'i10', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9'],
     'F': ['x1', 'x2', 'x3']}



.. code-block:: python

    # fit the model
    opt_ts.fit(X_arr, label_dict=label_dict); # we can optionally pass the label dict to the model during fit


.. parsed-literal::

    Warm starting...
    Done with warm start after 0.03secs
    
    Obj val: [3.77787002]
    
    Total time is 0.8secs
    
    

.. code-block:: python

    # checking the label dict
    opt_ts.label_dict_




.. parsed-literal::

    {'T': ['2000', '2005', '2010', '2015', '2020'],
     'N': ['i1', 'i10', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9'],
     'F': ['x1', 'x2', 'x3']}



We can get the cluster centers as a dataframe with the labels in
``label_dict``

.. code-block:: python

    cluster_centers_lst = opt_ts.get_named_cluster_centers()
    cluster_centers_lst[0] # first cluster




.. raw:: html

    
      <div id="df-7bc18d36-bd2a-4cd4-af29-606522765807" class="colab-df-container">
        <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>x1</th>
          <th>x2</th>
          <th>x3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2000</th>
          <td>-2.028199</td>
          <td>-2.377959</td>
          <td>-0.353205</td>
        </tr>
        <tr>
          <th>2005</th>
          <td>-2.028199</td>
          <td>-2.377959</td>
          <td>-0.353205</td>
        </tr>
        <tr>
          <th>2010</th>
          <td>-2.028199</td>
          <td>-2.377959</td>
          <td>-0.353205</td>
        </tr>
        <tr>
          <th>2015</th>
          <td>-2.028199</td>
          <td>-2.377959</td>
          <td>-0.353205</td>
        </tr>
        <tr>
          <th>2020</th>
          <td>-2.028199</td>
          <td>-2.377959</td>
          <td>-0.353205</td>
        </tr>
      </tbody>
    </table>
    </div>
        <div class="colab-df-buttons">
    
      <div class="colab-df-container">
        <button class="colab-df-convert" onclick="convertToInteractive('df-7bc18d36-bd2a-4cd4-af29-606522765807')"
                title="Convert this dataframe to an interactive table."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
        <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
      </svg>
        </button>
    
      <style>
        .colab-df-container {
          display:flex;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        .colab-df-buttons div {
          margin-bottom: 4px;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
        <script>
          const buttonEl =
            document.querySelector('#df-7bc18d36-bd2a-4cd4-af29-606522765807 button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-7bc18d36-bd2a-4cd4-af29-606522765807');
            const dataTable =
              await google.colab.kernel.invokeFunction('convertToInteractive',
                                                        [key], {});
            if (!dataTable) return;
    
            const docLinkHtml = 'Like what you see? Visit the ' +
              '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
              + ' to learn more about interactive tables.';
            element.innerHTML = '';
            dataTable['output_type'] = 'display_data';
            await google.colab.output.renderOutput(dataTable, element);
            const docLink = document.createElement('div');
            docLink.innerHTML = docLinkHtml;
            element.appendChild(docLink);
          }
        </script>
      </div>
    
    
    <div id="df-45f0d440-daaf-425d-99bd-9db20b46222b">
      <button class="colab-df-quickchart" onclick="quickchart('df-45f0d440-daaf-425d-99bd-9db20b46222b')"
                title="Suggest charts"
                style="display:none;">
    
    <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
         width="24px">
        <g>
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
        </g>
    </svg>
      </button>
    
    <style>
      .colab-df-quickchart {
          --bg-color: #E8F0FE;
          --fill-color: #1967D2;
          --hover-bg-color: #E2EBFA;
          --hover-fill-color: #174EA6;
          --disabled-fill-color: #AAA;
          --disabled-bg-color: #DDD;
      }
    
      [theme=dark] .colab-df-quickchart {
          --bg-color: #3B4455;
          --fill-color: #D2E3FC;
          --hover-bg-color: #434B5C;
          --hover-fill-color: #FFFFFF;
          --disabled-bg-color: #3B4455;
          --disabled-fill-color: #666;
      }
    
      .colab-df-quickchart {
        background-color: var(--bg-color);
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: var(--fill-color);
        height: 32px;
        padding: 0;
        width: 32px;
      }
    
      .colab-df-quickchart:hover {
        background-color: var(--hover-bg-color);
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: var(--button-hover-fill-color);
      }
    
      .colab-df-quickchart-complete:disabled,
      .colab-df-quickchart-complete:disabled:hover {
        background-color: var(--disabled-bg-color);
        fill: var(--disabled-fill-color);
        box-shadow: none;
      }
    
      .colab-df-spinner {
        border: 2px solid var(--fill-color);
        border-color: transparent;
        border-bottom-color: var(--fill-color);
        animation:
          spin 1s steps(1) infinite;
      }
    
      @keyframes spin {
        0% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
          border-left-color: var(--fill-color);
        }
        20% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        30% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
          border-right-color: var(--fill-color);
        }
        40% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        60% {
          border-color: transparent;
          border-right-color: var(--fill-color);
        }
        80% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-bottom-color: var(--fill-color);
        }
        90% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
        }
      }
    </style>
    
      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-45f0d440-daaf-425d-99bd-9db20b46222b button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
    
        </div>
      </div>
    



We can also get the labels as a dataframe indexed with labels in
``label_dict``

.. code-block:: python

    opt_ts.get_named_labels()




.. raw:: html

    
      <div id="df-e6c7e4a8-9188-4b89-8895-4e143ee38f44" class="colab-df-container">
        <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>2000</th>
          <th>2005</th>
          <th>2010</th>
          <th>2015</th>
          <th>2020</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>i1</th>
          <td>2</td>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>i10</th>
          <td>2</td>
          <td>2</td>
          <td>0</td>
          <td>2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>i2</th>
          <td>2</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>i3</th>
          <td>2</td>
          <td>2</td>
          <td>2</td>
          <td>1</td>
          <td>2</td>
        </tr>
        <tr>
          <th>i4</th>
          <td>2</td>
          <td>1</td>
          <td>2</td>
          <td>2</td>
          <td>2</td>
        </tr>
        <tr>
          <th>i5</th>
          <td>1</td>
          <td>2</td>
          <td>0</td>
          <td>1</td>
          <td>2</td>
        </tr>
        <tr>
          <th>i6</th>
          <td>2</td>
          <td>2</td>
          <td>2</td>
          <td>2</td>
          <td>2</td>
        </tr>
        <tr>
          <th>i7</th>
          <td>2</td>
          <td>0</td>
          <td>2</td>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>i8</th>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>2</td>
        </tr>
        <tr>
          <th>i9</th>
          <td>2</td>
          <td>2</td>
          <td>2</td>
          <td>2</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>
        <div class="colab-df-buttons">
    
      <div class="colab-df-container">
        <button class="colab-df-convert" onclick="convertToInteractive('df-e6c7e4a8-9188-4b89-8895-4e143ee38f44')"
                title="Convert this dataframe to an interactive table."
                style="display:none;">
    
      <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
        <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
      </svg>
        </button>
    
      <style>
        .colab-df-container {
          display:flex;
          gap: 12px;
        }
    
        .colab-df-convert {
          background-color: #E8F0FE;
          border: none;
          border-radius: 50%;
          cursor: pointer;
          display: none;
          fill: #1967D2;
          height: 32px;
          padding: 0 0 0 0;
          width: 32px;
        }
    
        .colab-df-convert:hover {
          background-color: #E2EBFA;
          box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
          fill: #174EA6;
        }
    
        .colab-df-buttons div {
          margin-bottom: 4px;
        }
    
        [theme=dark] .colab-df-convert {
          background-color: #3B4455;
          fill: #D2E3FC;
        }
    
        [theme=dark] .colab-df-convert:hover {
          background-color: #434B5C;
          box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
          filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
          fill: #FFFFFF;
        }
      </style>
    
        <script>
          const buttonEl =
            document.querySelector('#df-e6c7e4a8-9188-4b89-8895-4e143ee38f44 button.colab-df-convert');
          buttonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
    
          async function convertToInteractive(key) {
            const element = document.querySelector('#df-e6c7e4a8-9188-4b89-8895-4e143ee38f44');
            const dataTable =
              await google.colab.kernel.invokeFunction('convertToInteractive',
                                                        [key], {});
            if (!dataTable) return;
    
            const docLinkHtml = 'Like what you see? Visit the ' +
              '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
              + ' to learn more about interactive tables.';
            element.innerHTML = '';
            dataTable['output_type'] = 'display_data';
            await google.colab.output.renderOutput(dataTable, element);
            const docLink = document.createElement('div');
            docLink.innerHTML = docLinkHtml;
            element.appendChild(docLink);
          }
        </script>
      </div>
    
    
    <div id="df-2ba85a72-293a-4bcc-9dfb-7fc0cc514ffa">
      <button class="colab-df-quickchart" onclick="quickchart('df-2ba85a72-293a-4bcc-9dfb-7fc0cc514ffa')"
                title="Suggest charts"
                style="display:none;">
    
    <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
         width="24px">
        <g>
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
        </g>
    </svg>
      </button>
    
    <style>
      .colab-df-quickchart {
          --bg-color: #E8F0FE;
          --fill-color: #1967D2;
          --hover-bg-color: #E2EBFA;
          --hover-fill-color: #174EA6;
          --disabled-fill-color: #AAA;
          --disabled-bg-color: #DDD;
      }
    
      [theme=dark] .colab-df-quickchart {
          --bg-color: #3B4455;
          --fill-color: #D2E3FC;
          --hover-bg-color: #434B5C;
          --hover-fill-color: #FFFFFF;
          --disabled-bg-color: #3B4455;
          --disabled-fill-color: #666;
      }
    
      .colab-df-quickchart {
        background-color: var(--bg-color);
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: var(--fill-color);
        height: 32px;
        padding: 0;
        width: 32px;
      }
    
      .colab-df-quickchart:hover {
        background-color: var(--hover-bg-color);
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: var(--button-hover-fill-color);
      }
    
      .colab-df-quickchart-complete:disabled,
      .colab-df-quickchart-complete:disabled:hover {
        background-color: var(--disabled-bg-color);
        fill: var(--disabled-fill-color);
        box-shadow: none;
      }
    
      .colab-df-spinner {
        border: 2px solid var(--fill-color);
        border-color: transparent;
        border-bottom-color: var(--fill-color);
        animation:
          spin 1s steps(1) infinite;
      }
    
      @keyframes spin {
        0% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
          border-left-color: var(--fill-color);
        }
        20% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        30% {
          border-color: transparent;
          border-left-color: var(--fill-color);
          border-top-color: var(--fill-color);
          border-right-color: var(--fill-color);
        }
        40% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-top-color: var(--fill-color);
        }
        60% {
          border-color: transparent;
          border-right-color: var(--fill-color);
        }
        80% {
          border-color: transparent;
          border-right-color: var(--fill-color);
          border-bottom-color: var(--fill-color);
        }
        90% {
          border-color: transparent;
          border-bottom-color: var(--fill-color);
        }
      }
    </style>
    
      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-2ba85a72-293a-4bcc-9dfb-7fc0cc514ffa button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>
    
        </div>
      </div>
    



Checking most dynamic entities

.. code-block:: python

    print(f"total number of cluster changes is: {opt_ts.n_changes_}")
    opt_ts.get_dynamic_entities() # dynamic entities and their number of cluster changes


.. parsed-literal::

    total number of cluster changes is: 19
    



.. parsed-literal::

    (['i5', 'i7', 'i10', 'i8', 'i4', 'i3', 'i9', 'i2', 'i1'],
     [4, 3, 3, 2, 2, 2, 1, 1, 1])



.. code-block:: python

    # retrieve the cluster centers and labels
    cc_opt_ts = opt_ts.cluster_centers_
    labels_opt_ts = opt_ts.labels_
    labels_opt_ts




.. parsed-literal::

    array([[2, 2, 1, 1, 1],
           [2, 2, 0, 2, 0],
           [2, 0, 0, 0, 0],
           [2, 2, 2, 1, 2],
           [2, 1, 2, 2, 2],
           [1, 2, 0, 1, 2],
           [2, 2, 2, 2, 2],
           [2, 0, 2, 1, 1],
           [2, 1, 1, 1, 2],
           [2, 2, 2, 2, 1]])



.. code-block:: python

    # plot model results
    fig, ax = tsplot.plot(X=X_arr, cluster_centers=cc_opt_ts, labels=labels_opt_ts, label_dict=opt_ts.label_dict_)



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_127_0.png


.. code-block:: python

    # waterfall plot of a particular cluster center
    cc_idx = 0 # index of cluster center to plot
    cc = broadcast_data(X_arr.shape[0], cluster_centers=cc_opt_ts)[0][:, cc_idx, :] # broadcasting the cluster center
    fig, ax = tsplot.waterfall_plot(cc, label_dict=opt_ts.label_dict_)
    fig.suptitle(f"Water fall plot of cluster center {cc_idx}");



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_128_0.png


.. code-block:: python

    # waterfall plot of most dynamic entity
    most_dynamic_entity_idx = np.where(opt_ts.get_named_labels().index == opt_ts.get_dynamic_entities()[0][0])[0][0]
    fig, ax = tsplot.waterfall_plot(X_arr[:, most_dynamic_entity_idx, :], label_dict=opt_ts.label_dict_)
    fig.suptitle("Water fall plot of most dynamic entity");



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_129_0.png


.. code-block:: python

    # scoring the model
    print(f"inertia score is {inertia(X_arr, cc_opt_ts, labels_opt_ts, ord=1)}") # using l1 distance
    print(f"max_dist score is {max_dist(X_arr, cc_opt_ts, labels_opt_ts, ord=1)}") # using l1 distance


.. parsed-literal::

    inertia score is 138.84033246055895
    max_dist score is 3.777870015440997
    

We can also set the label_dict after fitting

.. code-block:: python

    old_label_dict = opt_ts.label_dict_
    old_label_dict




.. parsed-literal::

    {'T': ['2000', '2005', '2010', '2015', '2020'],
     'N': ['i1', 'i10', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9'],
     'F': ['x1', 'x2', 'x3']}



.. code-block:: python

    new_label_dict = {k: v for k, v in old_label_dict.items()}
    new_label_dict['F'] = ['A', 'B', 'C']
    
    opt_ts.set_label_dict(new_label_dict)

.. code-block:: python

    opt_ts.label_dict_




.. parsed-literal::

    {'T': ['2000', '2005', '2010', '2015', '2020'],
     'N': ['i1', 'i10', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9'],
     'F': ['A', 'B', 'C']}



**dynamic centers, fixed assignment**

.. code-block:: python

    # loading the data
    X_arr2, _ = load_data("./sythetic_data.npy")
    X_arr2.shape




.. parsed-literal::

    (10, 15, 1)



.. code-block:: python

    # visualizing the data
    fig, ax = tsplot.plot(X=X_arr2)



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_137_0.png


.. code-block:: python

    # initialize the model
    opt_ts = OptTSCluster(
        k=3,
        scheme='z1c1', # dynamic centers, dynamic assignment. Scheme needs to be a dynamic label scheme when using constrained cluster change
                       # you can also use 'z1c0' scheme here
        n_allow_assignment_change=0, # number of changes to allow, 0 means allow as no changes are allowed.
        warm_start=True # warm start with kmeans
    )

.. code-block:: python

    # checking the size of the model
    model_size = opt_ts.get_model_size(X_arr2)
    print(f"model has {model_size[0]} variables and {model_size[1]} constraints")


.. parsed-literal::

    model has 1066 variables and 1051 constraints
    

.. code-block:: python

    # fit the model
    opt_ts.fit(X_arr2);


.. parsed-literal::

    Warm starting...
    Done with warm start after 0.1secs
    
    Obj val: [1.51774178]
    
    Total time is 0.29secs
    
    

.. code-block:: python

    print(f"total number of cluster changes is: {opt_ts.n_changes_}")
    opt_ts.get_dynamic_entities() # indexes of dynamic entities and their number of cluster changes


.. parsed-literal::

    total number of cluster changes is: 0
    



.. parsed-literal::

    ([], [])



.. code-block:: python

    # retrieve the cluster centers and labels
    cc_opt_ts = opt_ts.cluster_centers_
    labels_opt_ts = opt_ts.labels_

.. code-block:: python

    labels_opt_ts




.. parsed-literal::

    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])



.. code-block:: python

    # plot of model results
    fig, ax = tsplot.plot(X=X_arr2, cluster_centers=cc_opt_ts, labels=labels_opt_ts)



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_144_0.png


.. code-block:: python

    # scoring the model
    print(f"inertia score is {inertia(X_arr2, cc_opt_ts, labels_opt_ts, ord=1)}") # using l1 distance
    print(f"max_dist score is {max_dist(X_arr2, cc_opt_ts, labels_opt_ts, ord=1)}") # using l1 distance


.. parsed-literal::

    inertia score is 117.50053747638934
    max_dist score is 1.5177417770731711
    

Lagrangian constrained changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Using lagrangian multiplier to penalize cluster changes**

.. code-block:: python

    # initialize the model
    opt_ts = OptTSCluster(
        k=3,
        scheme='z1c1', # dynamic centers, dynamic assignment. Scheme needs to be a dynamic label scheme when using constrained cluster change
        lagrangian_multiplier=0.1, # penalty for cluster changes.
        warm_start=True # warm start with kmeans
    )

.. code-block:: python

    # fit the model
    opt_ts.fit(X_arr2);


.. parsed-literal::

    Warm starting...
    Done with warm start after 0.07secs
    
    Obj val: [1.17290776]
    
    Total time is 3.59secs
    
    

.. code-block:: python

    opt_ts.get_model_size(X_arr2)




.. parsed-literal::

    (1066, 1050)



.. code-block:: python

    print(f"total number of cluster changes is: {opt_ts.n_changes_}")
    opt_ts.get_dynamic_entities() # indexes of dynamic entities and their number of cluster changes


.. parsed-literal::

    total number of cluster changes is: 4
    



.. parsed-literal::

    ([9, 12, 7], [2, 1, 1])



.. code-block:: python

    # retrieve the cluster centers and labels
    cc_opt_ts = opt_ts.cluster_centers_
    labels_opt_ts = opt_ts.labels_
    labels_opt_ts[opt_ts.get_dynamic_entities()[0]]




.. parsed-literal::

    array([[1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
           [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])



.. code-block:: python

    # plot of model results
    fig, ax = tsplot.plot(
        X=X_arr2,
        cluster_centers=cc_opt_ts,
        labels=labels_opt_ts,
        entity_idx=opt_ts.get_dynamic_entities()[0],
        show_all_entities=False
    )



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_153_0.png


.. code-block:: python

    # scoring the results
    print(f"inertia score is {inertia(X_arr2, cc_opt_ts, labels_opt_ts, ord=1)}") # using l1 distance
    print(f"max_dist score is {max_dist(X_arr2, cc_opt_ts, labels_opt_ts, ord=1)}") # using l1 distance


.. parsed-literal::

    inertia score is 106.79350571578108
    max_dist score is 1.172907757906211
    

Hard constrained changes
^^^^^^^^^^^^^^^^^^^^^^^^

**Creating dynamic entities**

.. code-block:: python

    dynamic_X1 = np.concatenate([X_arr2[:3, 0, :], X_arr2[3:, 2, :]], axis=0)[:, np.newaxis, :]
    dynamic_X2 = np.concatenate([X_arr2[:6, 6, :], X_arr2[6:, 4, :]], axis=0)[:, np.newaxis, :]

.. code-block:: python

    X_arr3 = np.concatenate([X_arr2, dynamic_X1, dynamic_X2], axis=1)
    X_arr3.shape




.. parsed-literal::

    (10, 17, 1)



.. code-block:: python

    # plotting the synthetically created dynamic entities
    fig, ax = tsplot.plot(X=X_arr3, entity_idx=np.arange(X_arr2.shape[1], X_arr3.shape[1]), show_all_entities=False)



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_159_0.png


.. code-block:: python

    # initialize the model
    opt_ts = OptTSCluster(
        k=3,
        scheme='z1c1', # dynamic centers, dynamic assignment. Scheme needs to be a dynamic label scheme when using constrained cluster change
        n_allow_assignment_change=2, # number of changes to allow, None means allow as many changes as possible
        warm_start=True # warm start with kmeans
    )

.. code-block:: python

    # fit the model
    opt_ts.fit(X_arr3);


.. parsed-literal::

    Warm starting...
    Done with warm start after 0.04secs
    
    Obj val: [1.51774178]
    
    Total time is 13.17secs
    
    

.. code-block:: python

    print(f"total number of cluster changes is: {opt_ts.n_changes_}")
    opt_ts.get_dynamic_entities() # indexes of dynamic entities and their number of cluster changes


.. parsed-literal::

    total number of cluster changes is: 2
    



.. parsed-literal::

    ([16, 15], [1, 1])



.. code-block:: python

    # retrieve the cluster centers and labels
    cc_opt_ts = opt_ts.cluster_centers_
    labels_opt_ts = opt_ts.labels_
    
    # labels of dynamic entities
    labels_opt_ts[opt_ts.get_dynamic_entities()[0]]




.. parsed-literal::

    array([[2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
           [2, 2, 2, 0, 0, 0, 0, 0, 0, 0]])



.. code-block:: python

    # plot of model results
    fig, ax = tsplot.plot(
        X=X_arr3,
        cluster_centers=cc_opt_ts,
        labels=labels_opt_ts,
        entity_idx=opt_ts.get_dynamic_entities()[0],
        show_all_entities=False
    )



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_164_0.png


.. code-block:: python

    # scoring the results
    print(f"inertia score is {inertia(X_arr3, cc_opt_ts, labels_opt_ts, ord=1)}") # using l1 distance
    print(f"max_dist score is {max_dist(X_arr3, cc_opt_ts, labels_opt_ts, ord=1)}") # using l1 distance


.. parsed-literal::

    inertia score is 140.53425058177024
    max_dist score is 1.5177417770731711
    

.. code-block:: python

    # checking the default label_dict (since we did not set the label dict or pass any during fit)
    print(opt_ts.label_dict_)


.. parsed-literal::

    {'T': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'N': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 'F': [0]}
    

TSGlobalKmeans
~~~~~~~~~~~~~~

This module applies sklearn’s k-mean clustering to the data resulting
from concatenating along the time axis.

.. code-block:: python

    # initialize the model
    g_ts_km = TSGlobalKmeans(n_clusters=3)

.. code-block:: python

    # fit the model
    g_ts_km.fit(X_arr3);


.. parsed-literal::

    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    

.. code-block:: python

    print(f"total number of cluster changes is: {g_ts_km.n_changes_}")
    g_ts_km.get_dynamic_entities() # indexes of dynamic entities and their number of cluster changes


.. parsed-literal::

    total number of cluster changes is: 53
    



.. parsed-literal::

    ([9, 16, 10, 1, 6, 8, 11, 13, 14, 0, 15, 5, 3, 2, 12, 7, 4],
     [6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1])



.. code-block:: python

    # retrieve the cluster centers and labels
    cc_g_ts_km = g_ts_km.cluster_centers_
    labels_g_ts_km = g_ts_km.labels_
    
    # labels of dynamic entities
    labels_g_ts_km[g_ts_km.get_dynamic_entities()[0]]




.. parsed-literal::

    array([[1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
           [2, 1, 0, 0, 0, 0, 1, 1, 1, 2],
           [2, 1, 0, 0, 0, 0, 0, 0, 1, 2],
           [2, 1, 1, 0, 0, 0, 0, 0, 1, 2],
           [2, 1, 0, 0, 0, 0, 0, 0, 1, 2],
           [2, 1, 1, 0, 0, 0, 0, 0, 1, 2],
           [2, 1, 1, 0, 0, 0, 0, 1, 1, 2],
           [2, 1, 1, 0, 0, 0, 0, 1, 1, 2],
           [2, 1, 0, 0, 0, 0, 0, 0, 1, 2],
           [2, 1, 1, 0, 0, 0, 0, 0, 1, 2],
           [2, 1, 1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
           [0, 1, 1, 1, 1, 1, 2, 2, 2, 2],
           [0, 1, 1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 2]], dtype=int32)



.. code-block:: python

    # plot of model results
    fig, ax = tsplot.plot(
        X=X_arr3,
        cluster_centers=cc_g_ts_km,
        labels=labels_g_ts_km,
        entity_idx=g_ts_km.get_dynamic_entities()[0],
        show_all_entities=False
    )



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_173_0.png


.. code-block:: python

    # scoring the results
    print(f"inertia score is {inertia(X_arr3, cc_g_ts_km, labels_g_ts_km, ord=1)}") # using l1 distance
    print(f"max_dist score is {max_dist(X_arr3, cc_g_ts_km, labels_g_ts_km, ord=1)}") # using l1 distance


.. parsed-literal::

    inertia score is 165.60528870396382
    max_dist score is 3.2183145554572867
    

TSKmeans
~~~~~~~~

This module applies tslearn’s time series k-mean clustering to the data.

.. code-block:: python

    # initialize the model
    ts_km = TSKmeans(n_clusters=3)

.. code-block:: python

    # fit the model
    ts_km.fit(X_arr3);

.. code-block:: python

    print(f"total number of cluster changes is: {ts_km.n_changes_}")
    ts_km.get_dynamic_entities() # indexes of dynamic entities and their number of cluster changes


.. parsed-literal::

    total number of cluster changes is: 0
    



.. parsed-literal::

    ([], [])



.. code-block:: python

    # retrieve the cluster centers and labels
    cc_ts_km = ts_km.cluster_centers_
    labels_ts_km = ts_km.labels_
    
    # labels of dynamic entities
    labels_ts_km[ts_km.get_dynamic_entities()[0]]




.. parsed-literal::

    array([], dtype=int64)



.. code-block:: python

    # plot of model results
    fig, ax = tsplot.plot(
        X=X_arr3,
        cluster_centers=cc_ts_km,
        labels=labels_ts_km
    )



.. image:: tscluster_tutorial_files/with_synthetic_data/images/tscluster_tutorial_181_0.png


.. code-block:: python

    # scoring the results
    print(f"inertia score is {inertia(X_arr3, cc_ts_km, labels_ts_km, ord=1)}") # using l1 distance
    print(f"max_dist score is {max_dist(X_arr3, cc_ts_km, labels_ts_km, ord=1)}") # using l1 distance


.. parsed-literal::

    inertia score is 113.5200041020405
    max_dist score is 5.437567193350128
    