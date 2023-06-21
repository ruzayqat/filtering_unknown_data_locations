import os
import time
import numpy as np
from scipy import linalg
from scipy.sparse import diags
import h5py
from data_tools import get_data



def kalman_filter(params):
    Kalman_Filter(params).run()

def enkf(params):
    if params["dy"] > params["M_enkf"]:
        #good for larger data points: dy >> 1
        EnKF_SMW(params).run()
    else:
        #faster for larger ensembles but with low dy
        EnKF(params).run()

def etkf(params):
    ETKF(params).run()

def estkf(params):
    ESTKF(params).run()


def fwd_slash(mat1, mat2):
    """ equivalent to mat1/mat2 in MATLAB. That is to solve for x in: x * mat2 = mat1
        This is equivalent to: mat2.T * x.T = mat1.T """
    return np.linalg.solve(mat2.T, mat1.T).T


def symmetric(matrix):
    """Symmetric matrix"""
    return np.triu(matrix) + np.triu(matrix, 1).T


class Kalman_Filter():
    def __init__(self, params):
        self.KF = np.zeros((params["dx"],params["T"]+1))
        self.Pa_diags = np.zeros((params["dx"],params["T"]))
        self.nstep = 0
        self.KF[:,0] = params["x_star"]
        self.C = np.zeros((params["dy"], params["dx"]))
        for i in range(params["dy"]):
            self.C[i,(i+1)*params["s_freq"] - 1] = 1
        self.R1 = params["sig_x"]**2 * np.eye(params["dx"])
        self.R2 = params["sig_y"]**2 * np.eye(params["dy"])
        self.params = params

    def run(self):
        Pa = np.zeros((self.params["dx"],self.params["dx"]))
        for n in range(self.params["T"]):
            print("KF: n = ", n)
            if np.isscalar(self.params["A"]): 
                Xf = self.params["A"] * self.KF[:,n]
                Pf = self.params["A"]**2 * Pa + self.R1
            else:
                Xf = self.params["A"] @ self.KF[:,n]
                Pf = self.params["A"] @ Pa @ self.params["A"].T + self.R1
            if (n + 1) % self.params['t_freq'] == 0:
                Yn = get_data(self.params, n, "data")
                Vari = self.C @ Pf @ self.C.T + self.R2
                K = Pf @ fwd_slash(self.C.T, Vari)
                ###########
                self.KF[:,n+1] = Xf + K @ (Yn - self.C @ Xf)
                Pa = (np.eye(self.params["dx"]) - K @ self.C) @ Pf
            else:
                self.KF[:,n+1] = Xf 
                Pa = Pf

            self.Pa_diags[:,n] = np.diag(Pa)

        self._dump_results("kalman_filter", self.KF)
        self._dump_results("kalman_filter_cov", self.Pa_diags)


    def _dump_results(self, file, data):
        if (file+"_file") not in self.params:
            filename = "%s/%s.h5" 
            self.params[file+"_file"] = filename %(self.params[file+"_dir"],file)
        dir_ = os.path.dirname(self.params[file+"_file"])
        os.makedirs(dir_, exist_ok=True)

        if os.path.isfile(self.params[file+"_file"]):
            os.remove(self.params[file+"_file"])

        with h5py.File(self.params[file+"_file"], "w") as fout:
            fout.create_dataset(name=file, data=data)


class EnKF():
    def __init__(self, params):
        self.EnKF = np.zeros((params["dx"],params["T"]+1))
        self.Pa_diags = np.zeros((params["dx"],params["T"]))
        self.nstep = 0
        self.EnKF[:,0] = params["x_star"]
        self.C = np.zeros((params["dy"], params["dx"]))
        for i in range(params["dy"]):
            self.C[i,(i+1)*params["s_freq"] - 1] = 1
        self.R2 = params["sig_y"]**2 * np.eye(params["dy"])
        self.params = params

    def run(self):
        x_a = np.transpose([self.params["x_star"]] * self.params["M_enkf"]) #of shape dx, M_enkf
        for n in range(self.params["T"]):
            if np.isscalar(self.params["A"]): 
                x_f = self.params["A"] * x_a 
                if self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_enkf"]))
            else:
                x_f = self.params["A"] @ x_a
                if self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_enkf"]))

            if (n + 1) % self.params['t_freq'] == 0:
                if not self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_enkf"]))
                Yn = get_data(self.params, n, "data")
                diff = x_f - np.mean(x_f, axis=1).reshape(-1, 1)
                temp = np.matmul(diff.T, self.C.T)
                temp1 = np.matmul(diff, temp) / (self.params['M_enkf']-1)
                temp = np.matmul(self.C, temp1) 
                #temp = fwd_slash(np.eye(self.params["dy"]), temp + self.R2)
                #temp = np.linalg.inv(temp + self.R2)
                kappa = fwd_slash(temp1, temp + self.R2) 
                
                temp = Yn.reshape(-1, 1) + self.params['sig_y'] * np.random.normal(size=(self.params["dy"],self.params["M_enkf"])) 
                temp -= np.matmul(self.C, x_f)
                
                x_a = x_f + np.matmul(kappa, temp)
            else:
                x_a = np.copy(x_f)

            self.EnKF[:,n+1] = np.mean(x_a, axis=1) 
            #self.Pa_diags[:,n] = np.diag(Pa)

        self._dump_results("enkf", self.EnKF)

    
    def _dump_results(self, file, data):
        if (file+"_file") not in self.params:
            filename = "%s/%s.h5" 
            self.params[file+"_file"] = filename %(self.params[file+"_dir"],file)
        dir_ = os.path.dirname(self.params[file+"_file"])
        os.makedirs(dir_, exist_ok=True)

        if os.path.isfile(self.params[file+"_file"]):
            os.remove(self.params[file+"_file"])

        with h5py.File(self.params[file+"_file"], "w") as fout:
            fout.create_dataset(name=file, data=data)


class EnKF_SMW():
    '''
    EnKF using  Sherman-Morrison-Woodbury formula: 
        (R + UV.T)^−1 = R^−1 − R^−1 U (I + V.T R^−1 U)^−1 V.T R^−1
        Then use Cholesky for (I + V.T R^−1 U)
    See: Efficient Implementation of the Ensemble Kalman Filter by Jan Mandel
    '''
    def __init__(self, params):
        self.EnKF = np.zeros((params["dx"],params["T"]+1))
        self.nstep = 0
        self.EnKF[:,0] = params["x_star"]
        self.C = np.zeros((params["dy"], params["dx"]))
        for i in range(params["dy"]):
            self.C[i,(i+1)*params["s_freq"] - 1] = 1
        #self.R2 = params["sig_y"]**2 * np.eye(params["dy"])
        #self.R2_inv = 1/params["sig_y"]**2 * np.eye(params["dy"])
        #self.S2 = params["sig_y"] * np.eye(params["dy"]) #Cholesky factor S_2 such that S_2 @ S_2.T = R_2 
        #self.S2_inv = 1/params["sig_y"] * np.eye(params["dy"]) #which here is also = S2_inv.T
        self.params = params

    def run(self):
        x_a = np.transpose([self.params["x_star"]] * self.params["M_enkf"]) #of shape dx, M
        for n in range(self.params["T"]):
            print("EnKF", n )
            if np.isscalar(self.params["A"]): 
                x_f = self.params["A"] * x_a 
                if self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_enkf"]))
            else:
                x_f = self.params["A"] @ x_a
                if self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_enkf"]))

            if (n + 1) % self.params['t_freq'] == 0:
                if not self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_enkf"]))
                CXf = np.matmul(self.C, x_f) #dy x M
                Yn = get_data(self.params, n, "data")
                Y = Yn.reshape(-1, 1) + self.params['sig_y'] * np.random.normal(size=(self.params["dy"],self.params["M_enkf"])) 
                Y -= CXf #Y = Y_n + noise - C @ X_f # dy x M
                Diff = x_f - np.mean(x_f, axis=1).reshape(-1, 1)
                CDiff = self.C @ Diff # dy x M

                B = 1/self.params["sig_y"] * CDiff #B = self.S2_inv @ CDiff #dy x M
                Q = np.eye(self.params["M_enkf"]) + 1/(self.params["M_enkf"]-1) * (B.T @ B) #M x m
                L = linalg.cholesky(Q, lower = True) # M x M
                # Z = CDiff.T @ self.R2_inv @ Y # (M x dy) x (dy x dy) x (dy x M) = (M x M)
                Z = 1/self.params["sig_y"]**2 * CDiff.T @ Y 
                G = linalg.solve(L, Z, lower=True)
                W = linalg.solve(L, G, lower=True, transposed=True)
                # V = (self.R2_inv @ Y - 1/(self.params["M_enkf"]-1) * self.S2_inv @ (B @ W))
                V = (1/self.params["sig_y"]**2 * Y - 1/(self.params["M_enkf"]-1) * 1/self.params["sig_y"] * (B @ W)) 
                Z = CDiff.T @ V
                x_a = x_f + 1/(self.params["M_enkf"]-1) * (Diff @ Z)
            else:
                x_a = np.copy(x_f)

            self.EnKF[:,n+1] = np.mean(x_a, axis=1) 

        self._dump_results("enkf", self.EnKF)

    
    def _dump_results(self, file, data):
        if (file+"_file") not in self.params:
            filename = "%s/%s.h5" 
            self.params[file+"_file"] = filename %(self.params[file+"_dir"],file)
        dir_ = os.path.dirname(self.params[file+"_file"])
        os.makedirs(dir_, exist_ok=True)

        if os.path.isfile(self.params[file+"_file"]):
            os.remove(self.params[file+"_file"])

        with h5py.File(self.params[file+"_file"], "w") as fout:
            fout.create_dataset(name=file, data=data)



class ETKF():
    '''
    ETKF: ensemble-transform-Kalman filter
    '''
    def __init__(self, params):
        self.ETKF = np.zeros((params["dx"],params["T"]+1))
        self.nstep = 0
        self.ETKF[:,0] = params["x_star"]
        self.C = np.zeros((params["dy"], params["dx"]))
        for i in range(params["dy"]):
            self.C[i,(i+1)*params["s_freq"] - 1] = 1
        #self.R1 = params["sig_x"]**2 * np.eye(params["dx"])
        #self.R2 = params["sig_y"]**2 * np.eye(params["dy"])
        #self.R2_inv = 1/params["sig_y"]**2 * np.eye(params["dy"])
        self.params = params


    def run(self):
        x_a = np.transpose([self.params["x_star"]] * self.params["M_etkf"]) #of shape dx, M
        for n in range(self.params["T"]):
            print("ETKF", n )
            if np.isscalar(self.params["A"]): 
                x_f = self.params["A"] * x_a 
                if self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_etkf"]))
            else:
                x_f = self.params["A"] @ x_a
                if self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_etkf"]))

            if (n + 1) % self.params['t_freq'] == 0:
                if not self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_etkf"]))
  
                mX = np.mean(x_f, axis=1)
                Xfp = x_f-mX[:,None]
                CXf = np.matmul(self.C, x_f)
                #Mean and perturbations for model values in observation space
                mY = np.mean(CXf, axis=1)
                CXp = CXf-mY[:,None]

                T = (1/self.params["sig_y"]**2) * CXp
                A2 = (self.params["M_etkf"]-1)*np.eye(self.params["M_etkf"]) + np.matmul(CXp.T, T)

                #eigenvalue decomposition of A2, A2 is symmetric
                eigs, ev = np.linalg.eigh(A2) 
                eigs_inv = 1/eigs
                #compute perturbations
                Wp1 = np.matmul(np.diag(np.sqrt(eigs_inv)), ev.T)
                Wp = np.matmul(ev, Wp1 * np.sqrt(self.params["M_etkf"]-1))

                #differing from pseudocode
                Yn = get_data(self.params, n, "data")
                d = Yn - mY
                D2 = np.matmul(CXp.T, (1/self.params["sig_y"]**2) * d)
                wm = ev @ np.diag(eigs_inv) @ ev.T @ D2 
                W = Wp + wm[:,None]

                x_a = mX + np.matmul(Xfp, W)

                # CXf = np.matmul(self.C, x_f) #dy x M
                # Yn = get_data(self.params, n, "data")
                # m_f = np.mean(x_f, axis=1).reshape(-1, 1)
                # sfm = 1 / np.sqrt(self.params["M_etkf"] - 1) * (x_f - m_f)
                # mean = np.sum(CXf, axis=1).reshape(-1, 1) / self.params["M_etkf"]
                # phi_k = 1 / np.sqrt(self.params["M_etkf"] - 1) * (CXf - mean).T / self.params["sig_y"]
                # temp = np.matmul(phi_k.T, phi_k) #dy x dy
                # eta_k = temp + np.eye(self.params["dy"])
                # kappa = np.matmul(sfm, fwd_slash(phi_k, eta_k))
                # m_a = m_f + np.matmul(kappa, (Yn.reshape(-1, 1) - mean)/self.params["sig_y"])
                # unit, diag = linalg.svd(np.matmul(phi_k, phi_k.T))[0:2] #(phi_k @ phi_k.T) is of shape M x M
                # diag = diags(diag)
                # sfm = np.matmul(sfm, fwd_slash(unit, np.sqrt(diag + np.eye(self.params["M_etkf"]))))
                # x_a = np.sqrt(self.params["M_etkf"] - 1) * sfm + m_a
            else:
                x_a = np.copy(x_f)

            self.ETKF[:,n+1] = np.mean(x_a, axis=1) 

        self._dump_results("etkf", self.ETKF)

    
    def _dump_results(self, file, data):
        if (file+"_file") not in self.params:
            filename = "%s/%s.h5" 
            self.params[file+"_file"] = filename %(self.params[file+"_dir"],file)
        dir_ = os.path.dirname(self.params[file+"_file"])
        os.makedirs(dir_, exist_ok=True)

        if os.path.isfile(self.params[file+"_file"]):
            os.remove(self.params[file+"_file"])

        with h5py.File(self.params[file+"_file"], "w") as fout:
            fout.create_dataset(name=file, data=data)




class ESTKF():
    '''
    ESTKF: error-subspace-transform-Kalman filter
    '''
    def __init__(self, params):
        self.estkf = np.zeros((params["dx"],params["T"]+1))
        self.nstep = 0
        self.estkf[:,0] = params["x_star"]
        self.C = np.zeros((params["dy"], params["dx"]))
        for i in range(params["dy"]):
            self.C[i,(i+1)*params["s_freq"] - 1] = 1
        #self.R1 = params["sig_x"]**2 * np.eye(params["dx"])
        #self.R2 = params["sig_y"]**2 * np.eye(params["dy"])
        #self.R2_inv = 1/params["sig_y"]**2 * np.eye(params["dy"])
        self.params = params


    def run(self):
        x_a = np.transpose([self.params["x_star"]] * self.params["M_estkf"]) #of shape dx, M
        for n in range(self.params["T"]):
            print("ESTKF", n )
            if np.isscalar(self.params["A"]): 
                x_f = self.params["A"] * x_a 
                if self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_estkf"]))
            else:
                x_f = self.params["A"] @ x_a
                if self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_estkf"]))

            if (n + 1) % self.params['t_freq'] == 0:
                if not self.params["add_noise_sig_every_dt"]:
                    x_f += self.params["sig_x"] * np.random.normal(size=(self.params["dx"],self.params["M_estkf"]))
                
                #Mean of prior ensemble for each state vector variable 
                mX = np.mean(x_f, axis=1)
                #Perturbations from ensemble mean
                Xfp = x_f-mX[:,None]
                
                #Mean of model values in observation space
                CXf = np.matmul(self.C, x_f)
                mY = np.mean(CXf, axis=1)
                Yn = get_data(self.params, n, "data")
                d = Yn - mY

                """
                Create projection matrix:
                - create matrix of shape M_estkf x M_estkf-1 filled with off diagonal values
                - fill diagonal with diagonal values
                - replace values of last row
                """
                sqr_M = -1/np.sqrt(self.params["M_estkf"])
                off_diag = -1/(self.params["M_estkf"]*(-sqr_M+1))
                diag = 1 + off_diag

                L = np.ones((self.params["M_estkf"], self.params["M_estkf"]-1)) * off_diag
                np.fill_diagonal(L, diag)
                L[-1,:] = sqr_M

                HL = np.matmul(CXf, L)
                B1 = (1/self.params["sig_y"]**2) * HL
                C2 = (self.params["M_estkf"]-1)*np.eye(self.params["M_estkf"]-1) + np.matmul(HL.T, B1)
                
                #EVD of C2, assumed symmetric
                eigs,U = np.linalg.eigh(C2)
                
                d2 = U.T @ (B1.T @ d)
                d3 = d2/eigs
                T = U @ np.diag(1/np.sqrt(eigs)) @ U.T
                
                #mean weight
                wm = np.matmul(U, d3)
                Wp = np.matmul(T, L.T*np.sqrt((self.params["M_estkf"]-1)))
                #total weight matrix + projection matrix transform
                W = wm[:,None] + Wp
                Wa = np.matmul(L, W)
                x_a = mX + np.matmul(Xfp, Wa)
            else:
                x_a = np.copy(x_f)

            self.estkf[:,n+1] = np.mean(x_a, axis=1) 

        self._dump_results("estkf", self.estkf)

    
    def _dump_results(self, file, data):
        if (file+"_file") not in self.params:
            filename = "%s/%s.h5" 
            self.params[file+"_file"] = filename %(self.params[file+"_dir"],file)
        dir_ = os.path.dirname(self.params[file+"_file"])
        os.makedirs(dir_, exist_ok=True)

        if os.path.isfile(self.params[file+"_file"]):
            os.remove(self.params[file+"_file"])

        with h5py.File(self.params[file+"_file"], "w") as fout:
            fout.create_dataset(name=file, data=data)