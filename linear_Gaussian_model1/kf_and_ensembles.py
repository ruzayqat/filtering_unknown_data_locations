import os
import time
import numpy as np
from scipy import linalg
from scipy.sparse import diags
import h5py
from data_tools import get_data
from joblib import Parallel, delayed


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

def enkf_local(params):
    EnKF_Local(params).run()

def etkf_local(params):
    ETKF_Local(params).run()


def fwd_slash(mat1, mat2):
    """ equivalent to mat1/mat2 in MATLAB. That is to solve for x in: x * mat2 = mat1
        This is equivalent to: mat2.T * x.T = mat1.T """
    return np.linalg.solve(mat2.T, mat1.T).T


def symmetric(matrix):
    """Symmetric matrix"""
    return np.triu(matrix) + np.triu(matrix, 1).T

def _dump_results(params, file, data):
    if (file+"_file") not in params:
        filename = "%s/%s.h5" 
        params[file+"_file"] = filename %(params[file+"_dir"],file)
    dir_ = os.path.dirname(params[file+"_file"])
    os.makedirs(dir_, exist_ok=True)

    if os.path.isfile(params[file+"_file"]):
        os.remove(params[file+"_file"])

    with h5py.File(params[file+"_file"], "w") as fout:
        fout.create_dataset(name=file, data=data)



class Kalman_Filter():
    def __init__(self, params):
        self.KF = np.zeros((params["dx"],params["T"]+1))
        self.Pa_diags = np.zeros((params["dx"],params["T"]))
        self.nstep = 0
        self.KF[:,0] = params["x_star"]
        self.C = np.zeros((params["dy"], params["dx"]))
        for i in range(params["dy"]):
            self.C[i,(i+1)*params["s_freq"] - 1] = 0.5
        self.R1 = params["sig_x"]**2 * np.eye(params["dx"])
        self.R2 = params["sig_y"]**2 * np.eye(params["dy"])
        self.params = params

    def run(self):
        #Pa = np.zeros((self.params["dx"],self.params["dx"]))
        Pa = np.eye(self.params["dx"])
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

        _dump_results(self.params, "kalman_filter", self.KF)
        _dump_results(self.params, "kalman_filter_cov", self.Pa_diags)


class EnKF():
    def __init__(self, params):
        self.EnKF = np.zeros((params["dx"],params["T"]+1))
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
            print("EnKF: n = ", n)
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
                CXf = np.matmul(self.C, x_f)
                xf_bar = np.mean(x_f, axis=1).reshape(-1, 1)
                CXf_bar =  np.matmul(self.C, xf_bar)
                diff = (CXf - CXf_bar).T ## of shape M x dy
                temp1 = np.matmul(x_f - xf_bar, diff) / (self.params['M_enkf']-1)
                temp = np.matmul(diff.T, diff) / (self.params['M_enkf']-1)
                kappa = fwd_slash(temp1, temp + self.R2) 
                
                temp = Yn.reshape(-1, 1) + self.params['sig_y'] * np.random.normal(size=(self.params["dy"],self.params["M_enkf"])) 
                temp -= CXf
                
                x_a = x_f + np.matmul(kappa, temp)
            else:
                x_a = np.copy(x_f)

            self.EnKF[:, n+1] = np.mean(x_a, axis=1) 

        _dump_results(self.params, "enkf", self.EnKF)


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
            print("EnKF_SMW = ", n )
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

        _dump_results(self.params, "enkf", self.EnKF)

    

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
                x_a = mX[:,None] + np.matmul(Xfp, W)

            else:
                x_a = np.copy(x_f)

            self.ETKF[:,n+1] = np.mean(x_a, axis=1) 

        _dump_results(self.params, "etkf", self.ETKF)



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
                x_a = mX[:,None] + np.matmul(Xfp, Wa)
            else:
                x_a = np.copy(x_f)

            self.estkf[:,n+1] = np.mean(x_a, axis=1) 

        _dump_results(self.params, "estkf", self.estkf)


class EnKF_Local():
    # Localized version of the Ensemble Kalman Filter
    # Here we will assume that 
    #   dx = Nx * Nx , i.e dx is a complere square
    def __init__(self, params):
        self.EnKF_loc = np.zeros((params["dx"],params["T"]+1))
        self.nstep = 0
        self.EnKF_loc[:,0] = params["x_star"]
        self.C = np.zeros((params["dy"], params["dx"]))
        for i in range(params["dy"]):
            self.C[i,(i+1)*params["s_freq"] - 1] = 1
        self.R2 = params["sig_y"]**2 * np.eye(params["dy"])
        self.diagR2 = np.diag(self.R2)
        self.params = params
        subdom_elements = params["subdom_elements"]
        if params["dx"]%subdom_elements != 0:
            for i in range(int(subdom_elements/2) - 4, subdom_elements+10):
                if params["dx"]%subdom_elements == 0:
                    subdom_elements = i
                    print("subdom_elements =", subdom_elements)
                    break

        self.numb_subdom = int(params["dx"]/subdom_elements) #dx must be divisble by subdom_elements
        self.part = np.zeros(params["dx"], dtype=int)

        for i in range(self.numb_subdom):
            self.part[i*subdom_elements : (i+1)*subdom_elements] = i * np.ones(subdom_elements)

        L = 1 #some length-scale
        #in our example dx = dy, both Nx = Ny.
        self.Nx = int(np.sqrt(params["dx"]))
        self.Ny = int(np.sqrt(params["dy"]))

        x_ = np.arange(0, self.Nx) * L
        y_ = np.arange(0, self.Nx) * L 
        x_,y_ = np.meshgrid(x_, y_, indexing = 'ij')
        x_ = x_.flatten()
        y_ = y_.flatten()

        x_Obs = np.arange(0, self.Ny) * L
        y_Obs = np.arange(0, self.Ny) * L 
        x_Obs,y_Obs = np.meshgrid(x_Obs, y_Obs, indexing = 'ij')
        x_Obs = x_Obs.flatten()
        y_Obs = y_Obs.flatten()

        # this array is outside of run() because for each time n, the observations at time n are covering the whole grid (in this example).
        # if at each time step n the locations of the opbservations are different, then you need to define their locations x_obs, y_obs 
        # and selectObs inside the function run() 
        self.selectObs = np.zeros((self.params["dx"], self.params["dy"]))

        for i in range(self.params["dx"]):
            self.selectObs[i] = np.exp(- ((x_[i] - x_Obs[:])**2 + (y_[i] - y_Obs[:])**2)/L**2 )
            #self.compact_locfun(np.sqrt((x_[i] - x_Obs[:])**2 + (y_[i] - y_Obs[:])**2)/L) #Not working for some reason the filter close to zero

        
    def run(self):
        x_a = np.transpose([self.params["x_star"]] * self.params["M_enkf"]) #of shape dx, M_enkf
        
        for n in range(self.params["T"]):
            print("EnKF_Local", n )
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
                
                #run local EnKF to return x_a
                x_a = self.run_loc(x_f,Yn)
                
            else:
                x_a = np.copy(x_f)

            self.EnKF_loc[:, n+1] = np.mean(x_a, axis=1) 

        _dump_results(self.params, "enkf_loc", self.EnKF_loc)


    def compact_locfun(self, r):
        # r is of shape Ny x 1
        f = np.zeros_like(r)
        f = ((((-r/4. + 1/2) * r + 5/8) * r - 5/3) * r**2 + 1) * (r <= 1) + \
                (((((r/12. - 1/2) * r + 5/8) * r + 5/3) * r - 5) * r + 4 - 2/(3*(r+1.e-15))) *((r > 1) & (r <= 2))

        return f # a vector

    def run_loc(self,x_f,Yn,minweight = 1e-8):

        # unique element of partition vector
        p = np.unique(self.part)
        x_a = np.zeros_like(x_f)

        CXf = np.matmul(self.C, x_f)
        xf_bar = np.mean(x_f, axis=1).reshape(-1, 1)
        CXf_bar =  np.matmul(self.C, xf_bar)

        if self.params["do_parallel_local_updates"]:
            SEL, X_a_Local = zip(*Parallel(n_jobs=self.params["ncores"])(delayed(self.parallel_local)(p, minweight, x_f, xf_bar, CXf, CXf_bar, Yn, i) \
                                            for i in range(len(p)))) 
            SEL = np.array(SEL)
            X_a_Local = np.array(X_a_Local)

            for i in range(len(p)):
                x_a[SEL[i], :] = X_a_Local[i]
        else:
            for i in range(len(p)):
                sel = np.where(self.part == p[i])[0]
                weight = self.selectObs[sel[0]]
                # restrict to local observations where weight exceeds minweight
                loc = np.where(weight > minweight)[0]
                CXfloc = CXf[loc,:]
                CXf_bar_loc = CXf_bar[loc]
                R2loc = np.diag(self.diagR2[loc] / weight[loc]) #R-localization
                Ynloc = Yn[loc]
                x_a[sel, :] = self.run_locenkf(x_f[sel,:], xf_bar[sel], CXfloc, CXf_bar_loc, Ynloc, R2loc)

        

        return x_a

    def parallel_local(self, p, minweight, x_f, xf_bar, CXf, CXf_bar, Yn, i):
        sel = np.where(self.part == p[i])[0]

        weight = self.selectObs[sel[0]]
        # restrict to local observations where weight exceeds minweight
        loc = np.where(weight > minweight)[0]
        CXfloc = CXf[loc,:]
        CXf_bar_loc = CXf_bar[loc]
        R2loc = np.diag(self.diagR2[loc] / weight[loc]) #R-localization
        Ynloc = Yn[loc]

        x_a_local = self.run_locenkf(x_f[sel,:], xf_bar[sel], CXfloc, CXf_bar_loc, Ynloc, R2loc)
        return sel, x_a_local


    def run_locenkf(self, x_f, xf_bar, CXf, CXf_bar, Yn, R2):
        M = np.shape(x_f)[1]
        dy0 = np.shape(Yn)[0]
        diff = (CXf - CXf_bar).T 
        
        temp1 = np.matmul(x_f - xf_bar, diff) / (M-1)
        temp = np.matmul(diff.T, diff) / (M-1)
        kappa = fwd_slash(temp1, temp + R2) 
        
        temp = Yn.reshape(-1, 1) + self.params['sig_y'] * np.random.normal(size=(dy0,M)) 
        temp -= CXf
        
        #return x_a
        return( x_f + np.matmul(kappa, temp) )


class ETKF_Local():
    # Localized version of the Ensemble Transofm Kalman Filter
    # Here we will assume that 
    #   dx = Nx * Nx , i.e dx is a complere square
    def __init__(self, params):
        self.ETKF_loc = np.zeros((params["dx"],params["T"]+1))
        self.nstep = 0
        self.ETKF_loc[:,0] = params["x_star"]
        self.C = np.zeros((params["dy"], params["dx"]))
        for i in range(params["dy"]):
            self.C[i,(i+1)*params["s_freq"] - 1] = 1
        self.R2 = params["sig_y"]**2 * np.eye(params["dy"])
        self.diagR2 = np.diag(self.R2)
        self.params = params
        subdom_elements = 20
        self.numb_subdom = int(params["dx"]/subdom_elements) #dx must be divisble by subdom_elements
        self.part = np.zeros(params["dx"], dtype=int)

        for i in range(self.numb_subdom):
            self.part[i*subdom_elements : (i+1)*subdom_elements] = i * np.ones(subdom_elements)

        L = 1 #some length-scale
        #in our example dx = dy, both Nx = Ny.
        self.Nx = int(np.sqrt(params["dx"]))
        self.Ny = int(np.sqrt(params["dy"]))

        x_ = np.arange(0, self.Nx) * L
        y_ = np.arange(0, self.Nx) * L 
        x_,y_ = np.meshgrid(x_, y_, indexing = 'ij')
        x_ = x_.flatten()
        y_ = y_.flatten()

        x_Obs = np.arange(0, self.Ny) * L
        y_Obs = np.arange(0, self.Ny) * L 
        x_Obs,y_Obs = np.meshgrid(x_Obs, y_Obs, indexing = 'ij')
        x_Obs = x_Obs.flatten()
        y_Obs = y_Obs.flatten()

        # this array is outside of run() because for each time n, the observations at time n are covering the whole grid (in this example).
        # if at each time step n the locations of the opbservations are different, then you need to define their locations x_obs, y_obs 
        # and selectObs inside the function run() 
        self.selectObs = np.zeros((self.params["dx"], self.params["dy"]))

        for i in range(self.params["dx"]):
            self.selectObs[i] = np.exp(- ((x_[i] - x_Obs[:])**2 + (y_[i] - y_Obs[:])**2)/L**2 )
            #self.compact_locfun(np.sqrt((x_[i] - x_Obs[:])**2 + (y_[i] - y_Obs[:])**2)/L)


    def run(self):
        x_a = np.transpose([self.params["x_star"]] * self.params["M_enkf"]) #of shape dx, M_enkf
        
        for n in range(self.params["T"]):
            print("ETKF_Local", n )
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
                
                #run local EnKF to return x_a
                x_a = self.run_loc(x_f,Yn)
                
            else:
                x_a = np.copy(x_f)

            self.ETKF_loc[:,n+1] = np.mean(x_a, axis=1) 

        _dump_results(self.params, "etkf_loc", self.ETKF_loc)


    def compact_locfun(self, r):
        # r is of shape Ny x 1
        f = np.zeros_like(r)
        f = ((((-r/4. + 1/2) * r + 5/8) * r - 5/3) * r**2 + 1) * (r <= 1) + \
                (((((r/12. - 1/2) * r + 5/8) * r + 5/3) * r - 5) * r + 4 - 2/(3*(r+1.e-15))) *((r > 1) & (r <= 2))

        return f # a vector

    def run_loc(self,x_f,Yn,minweight = 1e-8):

        # unique element of partition vector
        p = np.unique(self.part)
        x_a = np.zeros_like(x_f)

        CXf = np.matmul(self.C, x_f)
        xf_bar = np.mean(x_f, axis=1).reshape(-1, 1)
        CXf_bar =  np.matmul(self.C, xf_bar)

        for i in range(len(p)):
            sel = np.where(self.part == p[i])[0]

            weight = self.selectObs[sel[0]]
            # restrict to local observations where weight exceeds minweight
            loc = np.where(weight > minweight)[0]
            CXfloc = CXf[loc,:]
            CXf_bar_loc = CXf_bar[loc]
            R2loc = np.diag(self.diagR2[loc] / weight[loc])
            Ynloc = Yn[loc]

            x_a[sel, :] = self.run_locetkf(x_f[sel,:], xf_bar[sel], CXfloc, CXf_bar_loc, Ynloc, R2loc)

        return x_a


    def run_locetkf(self, x_f, xf_bar, CXf, CXf_bar, Yn, R2):
        M = np.shape(x_f)[1]
        dy0 = np.shape(Yn)[0]

        mX = np.mean(x_f, axis=1)
        Xfp = x_f-mX[:,None]

        #Mean and perturbations for model values in observation space
        mY = np.mean(CXf, axis=1)
        CXp = CXf-mY[:,None]

        T = (1/self.params["sig_y"]**2) * CXp
        A2 = (M-1)*np.eye(M) + np.matmul(CXp.T, T)

        #eigenvalue decomposition of A2, A2 is symmetric
        eigs, ev = np.linalg.eigh(A2) 
        eigs_inv = 1/eigs
        #compute perturbations
        Wp1 = np.matmul(np.diag(np.sqrt(eigs_inv)), ev.T)
        Wp = np.matmul(ev, Wp1 * np.sqrt(M-1))

        #differing from pseudocode
        d = Yn - mY
        D2 = np.matmul(CXp.T, (1/self.params["sig_y"]**2) * d)
        wm = ev @ np.diag(eigs_inv) @ ev.T @ D2 
        W = Wp + wm[:,None]

        x_a = mX[:,None] + np.matmul(Xfp, W)

        return x_a




    