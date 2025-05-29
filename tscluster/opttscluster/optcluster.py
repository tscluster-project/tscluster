from __future__ import annotations
import copy
from time import time
from typing import Tuple, List

import numpy as np
import numpy.typing as npt
from scipy.cluster.vq import kmeans, vq
import gurobipy as gp

class MILPClustering():
    def __init__(
            self, 
            k: int, 
            IFrac: float = 0.2, 
            use_full_constraints: bool = False, 
            max_iter: int = 10, 
            add_constaint_per_cluster: bool = True, 
            random_state: None|int = None, 
            warm_start: bool = True, 
            verbose: bool = True, 
            top_n_percentile: float = 0.0
            ) -> None:
        
        self.k = k
        self.IFrac = IFrac
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.use_full_constraints = use_full_constraints
        self.max_iter = max_iter
        self.add_constaint_per_cluster = add_constaint_per_cluster
        self.top_n_percentile = top_n_percentile 

        self.Es_star = None
        self.Zs_ans = None
        self.Cs_hats = None
        self.I_adds = None
        self.zis = None
        self.Z = None
        self.warm_start_labels_ = None

        self.x_min = 0
    
    def fit(
            self, 
            X: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
            y: npt.NDArray[np.float64] | npt.NDArray[np.int64] | None = None
            ) -> "MILPClustering":
        
        X = copy.deepcopy(X)

        n = len(X)

        if np.min(X) < 0:
            self.x_min = np.min(X) - 1
            X -= self.x_min
            
        # inital random X
        if self.random_state is not None:
            np.random.seed(self.random_state)

        I_idx = list(np.random.choice(list(range(n)), size=np.int64(np.round(self.IFrac*n)), replace=False))
        
        if self.use_full_constraints:
            I = X 
        else:
            I = X[I_idx, :]
    
        if self.warm_start:
            # warm start clusters
            km_Z, _ = kmeans(X, self.k)

            km_labels, _ = vq(X, km_Z)

            km_Z_to_Z_mapper = km_Z[:, 0].argsort()

            Z = km_Z[km_Z_to_Z_mapper, :]

            C = np.zeros((I.shape[0], self.k))

            km_Z_to_Z_mapper_key_val_pairs = np.array(list(enumerate(km_Z_to_Z_mapper)))

            for i in range(C.shape[0]):
                j = km_Z_to_Z_mapper_key_val_pairs[:, 0][km_Z_to_Z_mapper_key_val_pairs[:, 1] == km_labels[i]] 
                C[i, j] = 1        
                
        E_star = 0
        E_hat = np.inf
        count = 1
        
        epsilon = 1e-4
        
        Es_star = []
        Zs_ans = []
        Cs_hats = []
        I_adds = []
        zis = []
        
        t0 = time() 

        while E_hat > E_star+epsilon and count < self.max_iter+1:
            # solve MILP
            if self.warm_start:
                E_star, Z_ans, C_ans = MILPClustering.solve_MILP(I, self.k, Z, C)
            else:
                E_star, Z_ans, C_ans = MILPClustering.solve_MILP(I, self.k)
                
            C_hat = MILPClustering.assign_cluster(X, Z_ans)

            E_hat, I_add, zi = MILPClustering.add_constraints(X, Z_ans, C_hat, per_cluster=self.add_constaint_per_cluster,
                                                              top_n_percentile=self.top_n_percentile)
                    
            I_idx = set(I_idx) 
            I_idx.update(I_add)
            
            I_idx = list(I_idx)
            
            I = X[I_idx, :]
            
            if self.warm_start:
                C = np.zeros((I.shape[0], self.k))
                for i in range(C.shape[0]):
                    j = km_Z_to_Z_mapper_key_val_pairs[:, 0][km_Z_to_Z_mapper_key_val_pairs[:, 1] == km_labels[i]] 
                    C[i, j] = 1
            
            if self.verbose:    
                print(f"Done with {count} of {self.max_iter}. Ehat: {E_hat}, Estar: {E_star}")
        
                count += 1
            
            Es_star.append(E_star)
                        
            Zs_ans.append(Z_ans + self.x_min)
            Cs_hats.append(C_hat)
            I_adds.append(I_add)
            zis.append(np.array(zi) + self.x_min)

        self.Es_star = Es_star
        self.Zs_ans = Zs_ans
        self.Cs_hats = Cs_hats
        self.I_adds = I_adds
        self.zis = zis
        
        if self.warm_start:
            self.Z = Z + self.x_min
            self.warm_start_labels_ = km_labels

        if self.verbose:
            print(f"Total time is {np.round(time() - t0, 2)}secs")  
            print()

        return self

    def predict(
            self, 
            X: npt.NDArray[np.float64] | npt.NDArray[np.int64]
            ) -> npt.NDArray[np.float64] | npt.NDArray[np.int64]:
        
        return np.argmax(MILPClustering.assign_cluster(X, self.Z_ans), axis=1)

    @staticmethod
    def solve_MILP(
        I: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
        k: int, 
        init_Z: npt.NDArray[np.float64] | npt.NDArray[np.int64] | None = None, 
        init_C: npt.NDArray[np.float64] | npt.NDArray[np.int64] | None = None
        ) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        # init_Z is warm start
        
        # MILP
        m = gp.Model()
        # creating Vars
        E = m.addVar(name='E')
        
        Z_vars = m.addVars(k, I.shape[1], lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
        if init_Z is not None:
            for zi in range(k):
                for zj in range(I.shape[1]):
                    Z_vars[zi, zj].start = init_Z[zi, zj]
                
        C_vars = m.addVars(I.shape[0], k, vtype=gp.GRB.BINARY)
        if init_C is not None:
            for ci in range(I.shape[0]):
                for cj in range(k):
                    C_vars[ci, cj].start = init_C[ci, cj]
                
        e = m.addVars(I.shape[0], k, I.shape[1]) # to track absolute difference

        # creating Contraints
        for i in range(I.shape[0]):
            for j in range(k):
                for f in range(I.shape[1]):
    #                 e[i, j, f] = m.addVar(name='e_'+str(i)+'_'+str(j)+'_'+str(f))
                
                    m.addConstr(e[i, j, f] >= I[i, f] - Z_vars[j, f])
                    m.addConstr(e[i, j, f] >= -(I[i, f] - Z_vars[j, f]))
                    
        for i in range(I.shape[0]):
            for j in range(k):
                m.addConstr((C_vars[i, j] == 1) >> (E >= gp.quicksum(e[i, j, f] for f in range(I.shape[1]))))
                
            m.addConstr(gp.quicksum(C_vars[i, j] for j in range(k)) == 1)
            
        eps = 1e-3
        for j in range(k - 1):
            m.addConstr(Z_vars[j, 0] + eps <= Z_vars[j+1, 0])
        
        m.setObjective(E, gp.GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        
        m.optimize() 
        
        Z_ans = np.zeros((k, I.shape[1]))
        C_ans = np.zeros((I.shape[0], k))

        for ji in range(k):
            for jj in range(I.shape[1]):
                Z_ans[ji, jj] = Z_vars[ji, jj].X

            for ii in range(I.shape[0]):
                C_ans[ii, ji] = C_vars[ii, ji].X
                
        return E.X, Z_ans, C_ans

    @staticmethod
    def assign_cluster(
        X: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
        Z: npt.NDArray[np.float64] | npt.NDArray[np.int64]
        ) -> npt.NDArray[np.float64] | npt.NDArray[np.int64]:
    
        Z_cluster = []
            
        for z in range(Z.shape[0]):
            Z_cluster.append(np.linalg.norm(X - Z[z, :], ord=1, axis=1))
            
        C_hat = np.zeros((X.shape[0], Z.shape[0]))
        
        C_hat[list(range(X.shape[0])), np.argmin(np.array(Z_cluster), axis=0)] = 1
        
        return C_hat

    @staticmethod
    def add_constraints(
        X: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
        Z: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
        C_hat: npt.NDArray[np.float64] | npt.NDArray[np.int64], 
        per_cluster: bool = False, 
        top_n_percentile: float = 0.0
        ) -> Tuple[float, List, List]:

        E_hat = 0.0

        I_add = []
        zi = []

        if top_n_percentile:
            top_n_percentile_cut = np.int64(np.round(top_n_percentile*X.shape[0]))
        else:
            top_n_percentile_cut = 1

        # solve for E_hat = max violation
        for z in range(Z.shape[0]):
            distance = np.linalg.norm(X - Z[z, :], ord=1, axis=1) * C_hat[:, z]
            
            max_distance = np.max(distance)
            
            # optimize way of finding max Ehat during iteration
            if E_hat < max_distance:
                E_hat = max_distance

                if not per_cluster:
                    I_add = np.argsort(distance)[::-1][:top_n_percentile_cut]  #[np.argmax(distance) ]
                    zi = [z] 

            if per_cluster:
                I_add.extend(np.argsort(distance)[::-1][:top_n_percentile_cut])  #.append(np.argmax(distance))
                zi.append(z) 

        return E_hat, I_add, zi

if __name__ == "__main__":
    pass
