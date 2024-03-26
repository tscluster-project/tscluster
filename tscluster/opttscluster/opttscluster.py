from __future__ import annotations
from typing import TextIO, Tuple, List
import sys
import re
from time import time
import copy

import numpy as np
import numpy.typing as npt
from scipy.cluster.vq import kmeans, vq
import gurobipy as gp

from tscluster.interface import TSClusterInterface
from tscluster.base import TSCluster
from tscluster.opttscluster import optcluster
from tscluster.preprocessing.utils import infer_data

class OptTSCluster(TSCluster, TSClusterInterface):
    def __init__(
            self, 
            k: int, 
            scheme: str = 'z1c0', 
            n_allow_assignment_change: None|int = None, 
            is_Z_positive: bool = True, 
            is_tight_constraints: bool = True, 
            lagrangian_multiplier: int = 0, 
            use_sum_distance: bool = False,
            warm_start: bool = True, 
            normalise_assignment_penalty: bool = True, 
            strictly_n_allow_assignment_change: bool = False, 
            use_MILP_centroid: bool = True,
            use_full_constraints: bool = True, 
            IFrac: float = 0.2, 
            top_n_percentile: float = 0.0, 
            max_iter: int = 10, 
            random_state: None|int = None, 
            add_constraint_per_cluster: bool = True,
            init_with_prev: bool = True
            ) -> None:

        """
        Class for optimal time-series clustering. Throughout this doc and code, 'z' refers to cluster centers, while 'c' to label assignment.
        This creates an OptTSCluster object

        k: int
            number of clusters
        scheme: {'z0c0', 'z0c1', 'z1c0', 'z1c1'}, default='z1c0'
            the scheme to use for tsclustering
        n_allow_assignment_change: int or None, default=None
            total number of label changes to allow
        is_Z_positive: bool, default=True
            True means assume non-negativity constraint for z. The scale of the final results are not affected since OptTSCluster will return solutions in the original scale of the input data.
            Setting this to True often leads to speedup. See ... for more details.  
        is_tight_constraints: bool, default=True
            Indicate if to use use tight bounds (the bounding box of the data) for z and the objective value
        lagrangian_multiplier: int, default=0
            The penalty term for constrained label changes. Value should be in range [0, np.inf], the higher the value, the less
            the number of label changes allowed
        use_sum_distance: bool, default=False
            Indicate if to use sum of distance to cluster as the objective. This is the sum of the distances between points in a time series
            and their centroids. 
        warm_start: bool, default=True
            Indicates if to use k-means to initialize the centroids (Z) and their assignments (C).
        normalise_assignment_penalty: bool, default=True
            Indicate if to normalize the penalty term when using lagrangian_multiplier
        strictly_n_allow_assignment_change: bool, default=False
            If True, indicate if to use n_allow_assignment_change constraint as an active constraint.
        use_MILP_centroid: bool, default=True
            If True, cluster_centers_ atrribute will be cluster centers obtained from MILP solution, else the average of the 
            datapoints per timestep
        use_full_constraints: bool, default=True
            If True, use all the samples as constraints. If False, use constraint generation. When using constraint generation, 
            subsamples are being used as constraints with samples added to the constraints until optimal solution is found.
        IFrac: float, default=0.2
            fraction of samples to use as initial constraints when using constraint generation.
        top_n_percentile: foat, default=0.0
            percentile of most violated constraints to add to constraints when using constraint generation. 
            Value should in range [0, 1]. Note that a value of 0 means to use the most violated constraint. 
        max_iter: int, default=10
            Maximum number of iterations to use when using constraint violation.
        random_state: int, default=None
            Set the random seed used when initializing with k-means or when initializing samples when using constraint generation.
        add_constraint_per_cluster: bool, default=True
            If True, top_n_percentile constraints are being from each cluster when using constraint generation. 
        init_with_prev: bool, default=True
            If True, use the solution from previous iteration as initial solution for the next iteration when using constraint generation        
        """

        self.k = k
        self.scheme = scheme.lower()
        self.n_allow_assignment_change = n_allow_assignment_change
        self.is_Z_positive = is_Z_positive
        self.is_tight_constraints = is_tight_constraints 
        self.lagrangian_multiplier = lagrangian_multiplier
        self.use_sum_distance = use_sum_distance
        self.warm_start = warm_start
        self.normalise_assignment_penalty = normalise_assignment_penalty
        self.strictly_n_allow_assignment_change = strictly_n_allow_assignment_change
        self.use_MILP_centroid = use_MILP_centroid
        self.use_full_constraints = use_full_constraints
        self.IFrac = IFrac
        self.top_n_percentile = top_n_percentile 
        self.max_iter = max_iter
        self.random_state = random_state
        self.add_constraint_per_cluster = add_constraint_per_cluster
        self.init_with_prev = init_with_prev 

        self._use_closest_center_for_all = False

                # inital random X
        self.rand_gen = np.random.RandomState(self.random_state)

        if self.use_sum_distance:
            self.use_full_constraints = True 

        self.solver_schemes = {'z0c0', 'z1c0', 'z0c1', 'z1c1'}

        if self.scheme not in self.solver_schemes:
            raise ValueError(f"Invalid value for scheme. Expected any of {self.solver_schemes}, but got '{self.scheme}'")

        if re.search("c0", self.scheme, flags=re.IGNORECASE):
            self.n_allow_assignment_change = 0

        self.constant_assigment_constraint_violation_scheme = 'allow_v_violations_in_total'

        self.x_min = 0

    @infer_data
    def fit(
            self, 
            X: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
            y: npt.NDArray[np.float64] | npt.NDArray[np.int64] | None = None, 
            verbose: bool = True, 
            print_to: TextIO = sys.stdout, 
            ) -> "OptTSCluster": 

        """
        Method for solving the MILP model.

        X: np.ndarray of shape (T, N, F)
            T is the number of timesteps, N is the number of timeseries entities, F is the number of features. The input data to cluster
        y: ignored, not used. Only present as a convention for fit methods of most models.
        verbose: bool, default=True
            If True, some model training information will be printed out. Set to False to surpress printouts
        print_to: TextIO, default=sys.stdout
            An object with a write method to write model's printout information during training. Default is standard output.

        Return: self, fitted OptTSCluster object. 
        """
        epsilon = 1e-4
        verbose_flush = True 

        self.verbose = verbose
        self.T_, self.N_, self.F_ = X.shape
        self.X_ = copy.deepcopy(X)

        X = copy.deepcopy(X)

        n = len(X)

        if np.min(X) < 0 and self.is_Z_positive:
            self.x_min = np.min(X) - epsilon

        X -= self.x_min

        X_2d = X.reshape(-1, X.shape[2])

        x_std = np.std(X_2d, axis=0)

        z_min = np.min(X_2d, axis=0) - 0.5*epsilon # to ensure Z is not negative
        z_max = np.max(X_2d, axis=0) + x_std

        e_max = np.abs(z_max - z_min)
        E_max = np.linalg.norm(e_max, ord=1)

        if not self.is_Z_positive:
            z_min = -gp.GRB.INFINITY
            z_max = gp.GRB.INFINITY

        if not self.is_tight_constraints:
            z_min = 0
            if not self.is_Z_positive:
                z_min = -gp.GRB.INFINITY

            z_max = gp.GRB.INFINITY
            e_max = gp.GRB.INFINITY

        if self.warm_start:
            init_Zs = []
            init_Cs = []

            if self.verbose:    
                print("Warm starting...", file=print_to, flush=verbose_flush)
                t0 = time()

            for t in range(X.shape[0]):
                # warm start clusters
                if len(np.unique(X[t, :, :])) == 1:
                    init_Zs.append(np.array([np.unique(X[t, :, :])[0]]*(self.k*X.shape[2])).reshape(self.k, X.shape[2]))

                    c_ = np.zeros((X.shape[1], self.k))
                    ran_c = self.rand_gen.choice(np.arange(self.k), X.shape[1])

                    c_[np.arange(len(ran_c)), ran_c] = 1
                    init_Cs.append(c_)

                    continue

                km_Z, _ = kmeans(X[t, :, :], self.k)
                km_labels, _ = vq(X[t, :, :], km_Z)

                # km_Z = km.cluster_centers_
                km_Z_to_Z_mapper = km_Z[:, 0].argsort()

                Z = km_Z[km_Z_to_Z_mapper, :]

                C = np.zeros((X.shape[1], self.k))

                km_Z_to_Z_mapper_key_val_pairs = np.array(list(enumerate(km_Z_to_Z_mapper)))

                for i in range(C.shape[0]):
                    j = km_Z_to_Z_mapper_key_val_pairs[:, 0][km_Z_to_Z_mapper_key_val_pairs[:, 1] == km_labels[i]] 
                    C[i, j] = 1      

                init_Zs.append(Z)
                init_Cs.append(C)  

            if self.verbose:    
                print(f"Done with warm start after {np.round(time() - t0, 2)}secs", file=print_to, flush=verbose_flush)        
                print(file=print_to, flush=verbose_flush)

            init_Zs = np.array(init_Zs)
            init_Cs = np.array(init_Cs)

            if re.search("z0", self.scheme, flags=re.IGNORECASE):
                init_Zs = np.mean(init_Zs, axis=0)
                init_Cs = OptTSCluster.assign_cluster(X, init_Zs)
            #     if self.scheme in ['z0c0', 'z0c0_1']:
            #         init_Cs = init_Cs[0, :, :]

            # elif re.search("z1c0", self.scheme, flags=re.IGNORECASE):
            #     init_Cs = OptTSCluster.assign_cluster(X, init_Zs[0, :, :])
            #     if self.scheme in ['z1c0', 'z1c0_1']:
            #         init_Cs = init_Cs[0, :, :]
        
        else:
            init_Zs = None
            init_Cs = None  

        # initialize X
        if self.use_full_constraints:
            I = X 
            I_idx = list(range(X.shape[1]))
        else:
            # if self.scheme in ['z1c1', 'z1c1_1']:
            #     n = X.shape[1]
            #     I = []
            #     I_idx_t = []
            #     for t in range(X.shape[0]):
            #         I_idx = list(self.rand_gen.choice(list(range(n)), size=np.int64(np.round(self.IFrac*n)), replace=False))
            #         I_idx_t.append(I_idx)
            #         I.append(X[t, I_idx, :])
            #     I = np.array(I)

            # elif self.scheme == 'z0c1_2':
            #     n = X.shape[0]*X.shape[1]

            #     I_idx = list(self.rand_gen.randint(0, n, np.int64(np.round(self.IFrac*n))))

            #     I = X.reshape(n, X.shape[2])[I_idx, :]

            # else:
            n = X.shape[1]
            I_idx = list(self.rand_gen.choice(list(range(n)), size=np.int64(np.round(self.IFrac*n)), replace=False))
            I = X[:, I_idx, :]
        
        E_star = np.array([0])
        E_hat = np.array([np.Inf])
        count = 1
        
        epsilon = epsilon 
        
        Es_star = []
        Zs_ans = []
        Cs_hats = []
        I_adds = []
        zis = []

        solver_times = []
        percent_data_used = []

        t0 = time()

        has_any_new_constraint = True

        if self.use_full_constraints:
            self.max_iter = 1

        while ((E_hat > E_star+epsilon).sum() > 0 or has_any_new_constraint) and count < self.max_iter+1:

            z_fixed = False
            if re.search("z0", self.scheme, flags=re.IGNORECASE):
                z_fixed = True

            solver_time_0 = time()
            E_star, Z_ans, C_ans = OptTSCluster.solve_ts_MILP(I, self.k, 
                                                                    n_allow_assignment_change=self.n_allow_assignment_change,
                                                                    z_fixed=z_fixed,
                            constant_assigment_constraint_violation_scheme=self.constant_assigment_constraint_violation_scheme,
                                                                    init_Z=init_Zs,
                                                                    init_C=init_Cs,
                                                strictly_n_allow_assignment_change=self.strictly_n_allow_assignment_change,
                                                z_max=z_max,
                                                z_min=z_min,
                                                e_max=e_max,
                                                E_max=E_max,
                                                is_Z_positive=self.is_Z_positive,
                                                is_tight_constraints=self.is_tight_constraints,
                                                lagrangian_multiplier=self.lagrangian_multiplier,
                                                normalise_assignment_penalty=self.normalise_assignment_penalty,
                                                use_sum_distance=self.use_sum_distance
                                                                    )
            
            solver_times.append(np.round(time() - solver_time_0, 2))

            C_hat = OptTSCluster.assign_cluster(X, Z_ans) # C_hat contains assignment for all datapoints, while
                                                        # C_ans contains only for those used for solving the MILP.

            if not self._use_closest_center_for_all: #and self.scheme in ['z1c0']:
                C_hat[:, I_idx, :] = C_ans

            return_E_hat_per_t = False
            # if self.scheme in ['z1c1', 'z1c1_1']:
            #     return_E_hat_per_t = True  

            percent_data_used.append(np.round(100*I.size/X.size, 2))

            if self.verbose:
                print(file=print_to, flush=verbose_flush)
                print(f"{percent_data_used[-1]}% of data used", file=print_to, flush=verbose_flush)

            if self.use_sum_distance:
                E_hat, I_add, zi = E_star, [], []
            else:
                E_hat, I_add, zi = OptTSCluster.add_constraints(X, Z_ans, C_hat, 
                                                                        n_allow_assignment_change=self.n_allow_assignment_change,
                    constant_assigment_constraint_violation_scheme=self.constant_assigment_constraint_violation_scheme,
                                                                        per_cluster=self.add_constraint_per_cluster,
                                                                        top_n_percentile=self.top_n_percentile,
                                                                        return_E_hat_per_t=return_E_hat_per_t,
                                                                        verbose=self.verbose,
                                                                        print_to=print_to)
            # if self.scheme in ['z1c1', 'z1c1_1']:
            #     I = []
            #     for t in range(X.shape[0]):
            #         I_idx_t[t].extend(I_add[t])
            #         I.append(X[t, I_idx_t[t], :])
            #     I = np.array(I)
                
            # else:
            # print("checking for new constraints", file=print_to, flush=verbose_flush) # remeber to remove
            I_idx = set(I_idx) 

            I_idx_ = copy.deepcopy(I_idx)
            I_idx.update(I_add)

            if I_idx_ == I_idx:
                has_any_new_constraint = False  
        
            I_idx = list(I_idx)       

                # if self.scheme == 'z0c1_2':
                #     Xt = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
                #     I = Xt[I_idx, :]
                # else:
            I = X[:, I_idx, :]

            if self.verbose:    
                print(f"Done with {count} of {self.max_iter}. Ehat: {E_hat}, Estar: {E_star}", file=print_to, flush=verbose_flush)
                print(file=print_to, flush=verbose_flush)
        
                count += 1

            if self.init_with_prev:
                init_Zs = Z_ans 
                init_Cs = C_hat  
                
            Es_star.append(E_star)
            Zs_ans.append(Z_ans + self.x_min)
            Cs_hats.append(C_hat)
            I_adds.append(I_add)
            zis.append(np.array(zi) + self.x_min)

        self.Es_star_ = Es_star
        self.Zs_ans_ = Zs_ans
        self.Cs_hats_ = Cs_hats
        self.I_adds_ = I_adds
        self.zis_ = zis

        self.C_ans_ = C_ans

        self.solver_times_ = solver_times
        self.n_iter_ = count - 1
        self.percent_data_used_ = percent_data_used 
        self.E_hat_ = E_hat 

        if self.warm_start:
            self.warm_start_Z_ = Z + self.x_min
            self.warm_start_labels_ = km_labels

        if self.verbose:
            print(f"Total time is {np.round(time() - t0, 2)}secs", file=print_to, flush=verbose_flush)  
            print(file=print_to, flush=verbose_flush)

        return self 
        
    @property
    def cluster_centers_(self):
        if self.use_MILP_centroid:
            return self.Zs_ans_[-1]
        
        labels_ = self.labels_

        Zs_ans_ = np.zeros((self.T_, self.k, self.F_))

        for t in range(self.T_):
            for j in range(self.k):
                idx = labels_[:, t] == j
                Zs_ans_[t, j, :] = np.mean(self.X_[t, idx, :], axis=0)

        return Zs_ans_

    @property
    def labels_(self):
        return np.argmax(self.Cs_hats_[-1], axis=-1).T

    @property
    def fitted_data_shape_(self) -> Tuple[int, int, int]:
        """
        returns a tuple of the shape of the fitted data in TNF format. E.g (T, N, F) where T, N, and F are the number of timesteps,
        observations, and features respectively. 
        """
        return self.T_, self.N_, self.F_

    @staticmethod
    def solve_ts_MILP(
        I: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
        k: int, 
        init_Z: npt.NDArray[np.float64] | npt.NDArray[np.int64] | None = None, 
        init_C: npt.NDArray[np.float64] | npt.NDArray[np.int64] | None = None, 
        **kwargs
        ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        # MILP
        m = gp.Model()

        z_fixed = False
        if 'z_fixed' in kwargs:
            z_fixed = kwargs['z_fixed']

        v = kwargs['n_allow_assignment_change']
        lambda_val = kwargs['lagrangian_multiplier']
        normalise_assignment_penalty = kwargs['normalise_assignment_penalty']

        use_sum_distance = kwargs['use_sum_distance']

        T = I.shape[0]
        N = I.shape[1]
        F = I.shape[2]

        z_min = kwargs['z_min']
        z_max = kwargs['z_max']
        E_max = kwargs['E_max']
        E_max_ = kwargs['E_max']
        e_max = kwargs['e_max']

        if type(e_max) == type(gp.GRB.INFINITY): # to avoid error from 2D array e_max
            if e_max == gp.GRB.INFINITY:
                E_max = gp.GRB.INFINITY

        if kwargs['is_tight_constraints']:
            if kwargs['is_Z_positive']:
                z_min = np.vstack([z_min for _ in range(k)])
                z_max = np.vstack([z_max for _ in range(k)])

            e_max = np.vstack(
                [np.vstack(
                    [np.vstack([e_max for _ in range(k)]
                            ) for _ in range(N)
                    ]
                        ) for _ in range(T)
                ]
            )

        # creating Vars
        if use_sum_distance:
            E = m.addVars(T, N, name='E', lb=0, ub=E_max, vtype=gp.GRB.CONTINUOUS)
        else:
            E = m.addVar(name='E', lb=0, ub=E_max, vtype=gp.GRB.CONTINUOUS)

        if z_fixed:
            Z_vars = m.addVars(k, F, lb=z_min, ub=z_max, vtype=gp.GRB.CONTINUOUS)
            if init_Z is not None:
                for zj in range(k):
                    for zf in range(F):
                        Z_vars[zj, zf].start = init_Z[zj, zf]

        else:

            if kwargs['is_Z_positive'] and kwargs['is_tight_constraints']:
                z_min = np.vstack([z_min for _ in range(T)])
                z_max = np.vstack([z_max for _ in range(T)])

            Z_vars = m.addVars(T, k, F, lb=z_min, ub=z_max, vtype=gp.GRB.CONTINUOUS)
            if init_Z is not None:
                for zt in range(T):
                    for zj in range(k):
                        for zf in range(F):
                            Z_vars[zt, zj, zf].start = init_Z[zt, zj, zf]
                
        C_vars = m.addVars(T, N, k, vtype=gp.GRB.BINARY)
        if init_C is not None:
            for ct in range(T):
                for ci in range(N):
                    for cj in range(k):
                        C_vars[ct, ci, cj].start = init_C[ct, ci, cj]

        if v is not None or lambda_val > 0:
            y_vars = m.addVars(T-1, N, vtype=gp.GRB.BINARY)
                
        e = m.addVars(T, N, k, F, lb=0, ub=e_max, vtype=gp.GRB.CONTINUOUS) # to track absolute difference

        if z_fixed:
            # creating Contraints
            for t in range(T):
                for i in range(N):
                    for j in range(k):
                        for f in range(F):
            #                 e[i, j, f] = m.addVar(name='e_'+str(i)+'_'+str(j)+'_'+str(f))
                        
                            m.addConstr(e[t, i, j, f] >= I[t, i, f] - Z_vars[j, f])
                            m.addConstr(e[t, i, j, f] >= -(I[t, i, f] - Z_vars[j, f]))
        else:
            # creating Contraints
            for t in range(T):
                for i in range(N):
                    for j in range(k):
                        for f in range(F):
            #                 e[i, j, f] = m.addVar(name='e_'+str(i)+'_'+str(j)+'_'+str(f))
                        
                            m.addConstr(e[t, i, j, f] >= I[t, i, f] - Z_vars[t, j, f])
                            m.addConstr(e[t, i, j, f] >= -(I[t, i, f] - Z_vars[t, j, f]))
                    
        for t in range(T):
            for i in range(N):
                if use_sum_distance:
                    for j in range(k):
                        m.addConstr((C_vars[t, i, j] == 1) >> (E[t, i] >= gp.quicksum(e[t, i, j, f] for f in range(F))))

                else:
                    for j in range(k):
                        m.addConstr((C_vars[t, i, j] == 1) >> (E >= gp.quicksum(e[t, i, j, f] for f in range(F))))

                m.addConstr(gp.quicksum(C_vars[t, i, j] for j in range(k)) == 1)

                if t != T-1 and (v is not None or lambda_val > 0):
                    m.addConstr((y_vars[t, i] == 1) >> (gp.quicksum((jj+1)*C_vars[t, i, jj] for jj in range(k)) == 
                                    gp.quicksum((jj+1)*C_vars[t+1, i, jj] for jj in range(k))))

        if kwargs['constant_assigment_constraint_violation_scheme'] == 'allow_v_violations_in_total' and v is not None and lambda_val == 0:
            if kwargs['strictly_n_allow_assignment_change']:
                m.addConstr(gp.quicksum(y_vars[tt, ii] for tt in range(T-1) for ii in range(N)) == (T-1)*N-v)

            else:
                m.addConstr(gp.quicksum(y_vars[tt, ii] for tt in range(T-1) for ii in range(N)) >= (T-1)*N-v)
                

        elif kwargs['constant_assigment_constraint_violation_scheme'] == 'allow_v_violations_per_i' and v is not None and lambda_val == 0:
            for i in range(N):
                if kwargs['strictly_n_allow_assignment_change']:
                    m.addConstr(gp.quicksum(y_vars[tt, i] for tt in range(T-1)) == T-1-v)
                
                else:
                    m.addConstr(gp.quicksum(y_vars[tt, i] for tt in range(T-1)) >= T-1-v)

        # eps = 1e-3
        # if z_fixed:
        #         for j in range(k - 1):
        #             m.addConstr(Z_vars[j, 0] + eps <= Z_vars[j+1, 0])
        # else:

        #     for j in range(k - 1):
        #         m.addConstr(Z_vars[0, j, 0] + eps <= Z_vars[0, j+1, 0])
        
        if normalise_assignment_penalty:
            normaliser = E_max_ / ((T-1)*N)
        else:
            normaliser = 1

        if lambda_val > 0:
            if use_sum_distance:
                m.setObjective(gp.quicksum(E[tt, ii] for tt in range(T) for ii in range(N)) + 
                               lambda_val * normaliser * (1- gp.quicksum(y_vars[tt, ii] for tt in range(T-1) for ii in range(N))), 
                               gp.GRB.MINIMIZE)
            else:
                m.setObjective(E + lambda_val * normaliser * (1- gp.quicksum(y_vars[tt, ii] for tt in range(T-1) for ii in range(N))), 
                               gp.GRB.MINIMIZE)
        
        else:
            if use_sum_distance:
                m.setObjective(gp.quicksum(E[tt, ii] for tt in range(T) for ii in range(N)), gp.GRB.MINIMIZE)
            else:
                m.setObjective(E, gp.GRB.MINIMIZE)

        m.setParam('OutputFlag', 0)
        
        m.optimize() 
        
        if z_fixed:
            Z_ans = np.zeros((k, F))
        else:
            Z_ans = np.zeros((T, k, F))
        
        C_ans = np.zeros((T, N, k))

        for ji in range(k):
            for jj in range(F):
                if z_fixed:
                    Z_ans[ji, jj] = Z_vars[ji, jj].X
                else:
                    for jt in range(T):
                        Z_ans[jt, ji, jj] = Z_vars[jt, ji, jj].X

            for t in range(T):
                for ii in range(N):
                    C_ans[t, ii, ji] = C_vars[t, ii, ji].X

        if use_sum_distance:
            E_x = E.sum().getValue()      
        else: 
            E_x =E.X

        return np.array([E_x]), Z_ans, C_ans

    @staticmethod
    def assign_cluster(
        X: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
        Z: npt.NDArray[np.float64], 
        # _use_closest_center_for_all=True, 
        **kwargs
        ) -> npt.NDArray[np.float64] | npt.NDArray[np.int64]:
        # Z_cluster = []
        C_hat = []

        T = X.shape[0]
        N = X.shape[1]
        F = X.shape[2]
        k = Z.shape[0]

        for t in range(T):
            if len(Z.shape) == 3:
                C_hat.append(optcluster.MILPClustering.assign_cluster(X[t], Z[t], **kwargs))
            else:
                C_hat.append(optcluster.MILPClustering.assign_cluster(X[t], Z, **kwargs))
            
        return np.array(C_hat)
    

    @staticmethod
    def add_constraints(
        X: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
        Z: npt.NDArray[np.float64], 
        C_hat: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
        per_cluster: bool = False, 
        top_n_percentile: float = 0.0, 
        return_E_hat_per_t: bool = False, 
        **kwargs
        ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        E_hat_t = []

        if return_E_hat_per_t:
            I_add_t = []
        else:
            I_add_t = set()

        zi_t = []

        T = X.shape[0]
        N = X.shape[1]
        F = X.shape[2]
        k = Z.shape[0]

        for t in range(T):
            if len(Z.shape) == 3:
                E_hat, I_add, zi = optcluster.MILPClustering.add_constraints(X[t], Z[t], C_hat[t], 
                                                                 per_cluster=per_cluster, top_n_percentile=top_n_percentile)
            else:
                E_hat, I_add, zi = optcluster.MILPClustering.add_constraints(X[t], Z, C_hat[t], 
                                                                 per_cluster=per_cluster, top_n_percentile=top_n_percentile)
            
            E_hat_t.append(E_hat)

            if return_E_hat_per_t:
                I_add_t.append(I_add)
            else:
                I_add_t.update(I_add)

            zi_t.append(zi)
        
        if kwargs['n_allow_assignment_change'] is not None and kwargs['constant_assigment_constraint_violation_scheme'] is not None:

            I_add_t.update(OptTSCluster.constant_assigment_constraint_violators(C_hat, 
                                                                    v=kwargs['n_allow_assignment_change'],
                                                                    scheme=kwargs['constant_assigment_constraint_violation_scheme'],
                                                                    verbose=kwargs['verbose'],
                                                                    print_to=kwargs['print_to']
                                                                    )
            )
            
        if return_E_hat_per_t:
            E_hat_t = np.array(E_hat_t)
        else:
            E_hat_t = np.array([max(E_hat_t)])
            I_add_t = list(I_add_t)

        return E_hat_t, I_add_t, zi_t
    
    # @staticmethod
    # def add_constraints_Z_change_C_fixed(X, Z, C_hat, n_allow_assignment_change, constant_assigment_constraint_violation_scheme,
    #                                         per_cluster=False, top_n_percentile=False, return_E_hat_per_t=False, **kwargs):
    #     return OptTSCluster.add_constraints(X, Z, C_hat, n_allow_assignment_change=n_allow_assignment_change,
    #                                         constant_assigment_constraint_violation_scheme=constant_assigment_constraint_violation_scheme,
    #                                         per_cluster=False, top_n_percentile=False, return_E_hat_per_t=False, **kwargs)
        

    @staticmethod
    def constant_assigment_constraint_violators(
        C_hat: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
        v: None|int, 
        scheme: str, 
        **kwargs
        ) -> List[int]:

        I_add = []
        # I_add_all = []

        verbose = kwargs['verbose']

        if scheme == "allow_v_violations_in_total":
            n_violations = 0
            for i in range(C_hat.shape[1]):
                y = np.argmax(C_hat[:, i, :], axis=1)
                if (y[:-1] != y[1:]).sum() > 0:
                    if verbose:
                        print(f"i that changed: {i}", file=kwargs['print_to'])
                    I_add.append(i)
                    # I_add_all.append(i)
                    n_violations += (y[:-1] != y[1:]).sum()

            # if n_violations <= v:
            #     I_add = []
            
        elif scheme == 'allow_v_violations_per_i':
            for i in range(C_hat.shape[1]):
                y = np.argmax(C_hat[:, i, :], axis=1)
                if (y[:-1] != y[1:]).sum() > v:
                    if verbose:
                        print(f"i that changed: {i}", file=kwargs['print_to'])
                    I_add.append(i)
                    # I_add_all.append(i)

        return I_add#, I_add_all

if __name__ == "__main__":
    pass