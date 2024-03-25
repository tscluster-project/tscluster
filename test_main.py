import numpy as np
from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tscluster.opttscluster import OptTSCluster
from tscluster.tskmeans import TSKmeans, TSGlobalKmeans
from tscluster.preprocessing import utils as preprocess_utils, TSStandardScaler, TSMinMaxScaler
from tscluster.metrics import inertia, max_dist

X_mini = np.load("./sythetic_data.npy")

kc_opt_model = OptTSCluster(3, random_state=42, use_full_constraints=1, warm_start=1, 
                          scheme='z1c1', n_allow_assignment_change=None,
                             use_MILP_centroid=True, is_tight_constraints=False, is_Z_positive=True,
                             init_with_prev=False, use_sum_distance=False)

kc_opt_model.fit(X_mini, verbose=True)

print(f"inertia score is {inertia(X_mini, kc_opt_model.cluster_centers_, kc_opt_model.labels_, ord=1)}")
print(f"max_dist score is {max_dist(X_mini, kc_opt_model.cluster_centers_, kc_opt_model.labels_, ord=1)}")
print(f"shape of labels_ is {kc_opt_model.labels_.shape}")
print("head of labels_ is:")
print(kc_opt_model.labels_[:5, :])

print(f"shape of cluster_centers_ is {kc_opt_model.cluster_centers_.shape}")
print("head of cluster_centers_ is:")

if len(kc_opt_model.cluster_centers_.shape) == 3:
    print(kc_opt_model.cluster_centers_[:5, :, :])
else:
    print(kc_opt_model.cluster_centers_)

X = random_walks(n_ts=50, sz=32, d=2, random_state=42)
Xt = preprocess_utils.NTF_to_TNF(X)

km = TSKmeans(n_clusters=3, metric="euclidean", max_iter=5, random_state=0)
km.fit(X, arr_format="NTF")
print(km.Xt.shape)
print(km.labels_)
print(len(km.labels_))
print(km.cluster_centers_.shape)
print(km.cluster_centers_[0, :5, :])
print()

km2 = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, random_state=0).fit(X)
print(km2.labels_)
print(len(km2.labels_))
print(km2.cluster_centers_[:5, 0, :])

km3 = TSGlobalKmeans(n_clusters=3)
km3.fit(Xt)
print(km3.labels_)
print(km3.labels_.shape)

print("Z score")

print("per_time =True")

scaler = TSStandardScaler(per_time=True)
scaler.fit(X, arr_format='NTF')
X_scaled = scaler.transform(X, arr_format='NTF',) 
print(f"Zscore X_scaled shape is {X_scaled.shape}")
print(X_scaled[0, :5, :])
print(f"Original data is:")
print(Xt[0, :5, :])
print("Inverse is: ")
print(scaler.inverse_transform(X_scaled)[0, :5, :])
print()

sk_scaler = StandardScaler()
print(sk_scaler.fit_transform(Xt[0])[:5])

print("per_time = False")

scaler = TSStandardScaler(per_time=False)
X_scaled = scaler.fit_transform(Xt)
print(f"Zscore X_scaled shape is {X_scaled.shape}")
print(X_scaled[0, :5, :])
print(f"Original data is:")
print(Xt[0, :5, :])
print("Inverse is: ")
print(scaler.inverse_transform(X_scaled)[0, :5, :])
print()

sk_scaler = StandardScaler()
print(sk_scaler.fit_transform(np.vstack(Xt))[:5])


print("MinMax")

print("per_time =True")

scaler = TSMinMaxScaler(per_time=True)
X_scaled = scaler.fit_transform(Xt)
print(f"MinMax X_scaled shape is {X_scaled.shape}")
print(X_scaled[0, :5, :])
print(f"Original data is:")
print(Xt[0, :5, :])
print("Inverse is: ")
print(scaler.inverse_transform(X_scaled)[0, :5, :])
print()

sk_scaler = MinMaxScaler()
print(sk_scaler.fit_transform(Xt[0])[:5])

print("per_time = False")

scaler = TSMinMaxScaler(per_time=False)
X_scaled = scaler.fit_transform(Xt)
print(f"MinMax X_scaled shape is {X_scaled.shape}")
print(X_scaled[0, :5, :])
print(f"Original data is:")
print(Xt[0, :5, :])
print("Inverse is: ")
print(scaler.inverse_transform(X_scaled)[0, :5, :])
print()

sk_scaler = MinMaxScaler()
print(sk_scaler.fit_transform(np.vstack(Xt))[:5])

scaler = TSMinMaxScaler(per_time=False)
print()
print("trying fit(str) for csv")
Xt = scaler.fit_transform("../synthetic_csv")
print(Xt[0, :5, :])
print("Inverse tranform for csv")
print(scaler.inverse_transform(Xt)[0, :5, :])

print()
print("trying fit(str) for json")
Xt = scaler.fit_transform("../synthetic_json")
print(Xt[0, :5, :])
print("Inverse tranform for json")
print(scaler.inverse_transform(Xt)[0, :5, :])

print()
print("trying fit(str) for npy")
Xt = scaler.fit_transform("../synthetic_npy")
print(Xt[0, :5, :])
print("Inverse tranform for npy")
print(scaler.inverse_transform(Xt)[0, :5, :])

print()
print("trying fit(lst) for csv")
file_list = [
    "../synthetic_csv/timestep_0.csv",
    "../synthetic_csv/timestep_1.csv",
    "../synthetic_csv/timestep_2.csv",
    "../synthetic_csv/timestep_3.csv",
    "../synthetic_csv/timestep_4.csv"
]

scaler.fit(file_list, read_file_args={'header': 0, 'sep': ","})
Xt = scaler.transform(file_list, read_file_args={'header': 0, 'sep': ","})
print(Xt[0, :5, :])
print("Inverse tranform for csv")
print(scaler.inverse_transform(Xt)[0, :5, :])

print()
print("fit_transform")
# scaler.fit(file_list, read_file_args={'header': None, 'sep': ","})
Xt = scaler.fit_transform(file_list, read_file_args={'header': 0, 'sep': ","})
print(Xt[0, :5, :])
print("Inverse tranform for csv")
print(scaler.inverse_transform(Xt)[0, :5, :])
