import os
import time
import numpy as np
import h5py
from solver1 import solve
from data_tools import (get_data, get_floater_info)
from scipy import linalg as scl 
from scipy.interpolate import CubicSpline 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def mcmc_filter(params, isim):
    """main function of MCMC filtering"""
    np.random.seed(isim)
    mcmc_f = MCMC_Filter(isim, params)
    mcmc_f.run()


def get_drifters_data(params):
    hyp = np.sqrt(params["dx"]**2 + params["dy"]**2)
    data = np.zeros((24*params["simu_days"],2*params["N_f"])) #stack u of size(time, N_f) and v of size(time, N_f)
    x_ind = np.arange(0, params["dgx"])
    y_ind = np.arange(0, params["dgy"])
    x_mesh = x_ind * params["dx"]
    y_mesh = y_ind * params["dy"]
    height = params["x_star"][0:params["dim2"]].reshape(params["dgx"],params["dgy"])
    floater_xy = np.zeros((24*params["simu_days"], 2 * params["N_f"]))
    floater_uv = np.zeros((24*params["simu_days"], 2 * params["N_f"]))
    floater_uv_error = np.zeros((24*params["simu_days"], 2 * params["N_f"]))
    if params["floaters_init_data_provided"]:
        floater_xy_uv = np.zeros((params["N_f"], 8))
        #read them
        for floater in range(params["N_f"]):
            for i,key in enumerate(["x","y","ve","vn","ve_error","vn_error"]):
                filename = "data/drifters/floater_%s%05d.npy" %(key,floater)
                if i == 0:
                    floater_xy[:,2*floater] = np.load(filename) #lon
                elif i == 1:
                    floater_xy[:,2*floater+1] = np.load(filename) #lat
                elif i == 2:
                    floater_uv[:,2*floater] = np.load(filename) #u
                    data[:,floater] = np.load(filename)
                elif i == 3:
                    floater_uv[:,2*floater+1] = np.load(filename) #v
                    data[:, params["N_f"] + floater] = np.load(filename)
                elif i == 4:
                    floater_uv_error[:,2*floater] = np.load(filename) #u-error
                elif i == 5:
                    floater_uv_error[:,2*floater+1] = np.load(filename) #v-error

 
                        
            floater_xy[:,2*floater] = params["m_per_deg_lon"] * np.abs(floater_xy[:,2*floater] - params["long"][0])
            floater_xy[:,2*floater+1] = params["m_per_deg_lat"] * np.abs(floater_xy[:,2*floater+1] - params["lat"][0])


        t_ = np.arange(0, 24*params['simu_days'])
        data = CubicSpline(t_, data, axis = 0)

        ##compute the height of the water at the drifter locations only at t = 0
        # h_float = np.zeros(params["N_f"]) 
        # surr_pts_ind = np.zeros((2, 4 * params["N_f"]), dtype = int)
        # surr_pts_xy = np.zeros((2, 4 * params["N_f"]))
        # w = np.zeros((params["N_f"], 4))
        # for i in range(params["N_f"]):
        #     c1 = (x_mesh >= floater_xy[0,2*i])
        #     right = x_ind[c1]
        #     c2 = (x_mesh <= floater_xy[0,2*i])
        #     left  = x_ind[c2]
        #     c3 = (y_mesh >= floater_xy[0,2*i+1])
        #     above = y_ind[c3]
        #     c4 = (y_mesh <= floater_xy[0,2*i+1])
        #     below = y_ind[c4]

        #     #print("floater_xy = ", floater_xy[2*i:2*(i+1)], "y_mesh[-1]", y_mesh[-1])
            
        #     surr_pts_ind[0, 4*i : 4*(i+1)]  = np.array([right[0],right[0], left[-1], left[-1]])
        #     surr_pts_ind[1, 4*i : 4*(i+1)]  = np.array([above[0],below[-1],below[-1],above[0]])
        #     surr_pts_xy[0, 4*i : 4*(i+1)] = x_mesh[surr_pts_ind[0, 4*i : 4*(i+1)]]
        #     surr_pts_xy[1, 4*i : 4*(i+1)] = y_mesh[surr_pts_ind[1, 4*i : 4*(i+1)]]

        #     for j in range(4):
        #         dist_from_float  = np.sqrt((surr_pts_xy[0,4*i+j] - floater_xy[0,2*i])**2\
        #                                 + (surr_pts_xy[1,4*i+j] - floater_xy[0,2*i+1])**2)
        #         w[i,j] = np.maximum(0.0, hyp - dist_from_float)
        #     w[i,:] = w[i,:] / np.sum(w[i,:])

        #     for j in range(4):
        #         h_float[i] += height[surr_pts_ind[0, 4*i+j], surr_pts_ind[1, 4*i+j]] *  w[i,j]
    else:
        raise Exception("Should provide the initial positions of the drifters!")


    return floater_xy[0], floater_uv, floater_uv_error,  data #,h_float

def block_diag_einsum(arr, num):
    rows, cols = arr.shape
    result = np.zeros((num, rows, num, cols), dtype=arr.dtype)
    diag = np.einsum('ijik->ijk', result)
    diag[:] = arr
    return result.reshape(rows * num, cols * num)


class MCMC_Filter():
    """Filtering Using MCMC main class
    pi_n(x_n|y_{1:n}) propto prod_{k=1}^n g(y_k|x_k) f(x_k|x_{k-1})
                        =   g(y_n|x_n) f(x_n|x_{n-1}) pi_{n-1}(x_{n-1})
    Approximate pi_{n-1} by an empirical distribution
    pi_{n-1}^N(x_{n-1}) = (1/N) sum_{j=1}^N delta_{X_{n-1}^{(j)}} (x_{n-1})
    Then we can approximate pi_n(x_n|y_{1:n}) by
    pi_n(x_n|y_{1:n})^N propto g(y_n|x_n) sum_{j=1}^N f(x_n|X_{n-1}^{(j)})
    Sample a uniform integer j from {1,...,N}.
    We will use MCMC to sample from the following:
    tilde{pi}_n(x_n;j|y_{1:n})^N propto g(y_n|x_n) f(x_n|X_{n-1}^{(j)})
    """
    def __init__(self, isim, params):
        self.mcmc_samples = np.zeros((params["dimx"],params["T"]+1))
        self.accep_rate = np.zeros(params["T"])
        self.mcmc_filt_iter = params["mcmc_N"] + params["burn_in"]
        self.samples = np.zeros((params["t_freq"],  params['dimx'], params["mcmc_N"]))
        self.floater_xy = np.zeros((params["T"]+1, 2 * params["N_f"]))
        self.floater_height = np.zeros((params["T"]+1, params["N_f"]))
        self.obs_indices = np.zeros(params["N_f"], dtype = int)
        self.obs_xy = np.zeros(2 * params["N_f"])
        params["isim"] = isim
        self.hyp = np.sqrt(params["dx"]**2 + params["dy"]**2)
        self.x_ind = np.arange(0, params["dgx"])
        self.y_ind = np.arange(0, params["dgy"])
        self.x_mesh = self.x_ind * params["dx"]
        self.y_mesh = self.y_ind * params["dy"]
        self.t_simul = 0
        self.tsteps = np.zeros(params["T"])
        self.nstep = 0
        # self.floater_xy[0,:], self.floater_uv, self.floater_uv_error,\
        #         self.data,self.floater_height[0,:] = get_drifters_data(params)
        
        #since we dont need floater_height (maybe only for plotting)
        self.floater_xy[0,:], self.floater_uv, self.floater_uv_error,\
                      self.data = get_drifters_data(params)

        self.J = np.arange(0,params["noise_modes_num"])

        self.A = np.sin(2 * np.pi/self.x_mesh[-1] * self.x_mesh[:,None] * self.J) #of shape (dgx, noise_modes_num )
        #self.A[:,0] = np.ones(self.A.shape[0]) #first mode is random variable *  1
        self.B = np.sin(2 * np.pi/self.y_mesh[-1] * self.y_mesh[:,None] * self.J) #of shape (dgy, noise_modes_num )
        #self.B[:,0] = np.ones(self.B.shape[0])  #first mode is random variable *  1

    
        #self.var_sqrt = params['sig_x']/np.sqrt(self.J+1) #of shape (noise_modes_num,)
        self.var_sqrt = np.zeros((params["noise_modes_num"],params["noise_modes_num"]))
        for i in range(params["noise_modes_num"]):
            for j in range(params["noise_modes_num"]):
                self.var_sqrt[i,j] = params['sig_x']/np.sqrt(np.maximum(i,j)+1)

        AA = self.A.T @ self.A
        Q = np.zeros((params["dgy"],params["dgy"]))

        for r in range(params["dgy"]):
            for s in range(params["dgy"]):
                S = 0.
                for i in range(params["noise_modes_num"]):
                    for j in range(params["noise_modes_num"]):
                        S += self.B[r,i] * self.B[s,i] * AA[j,j] /(np.maximum(i,j)+1)
                S *= params['sig_x']**2
                Q[r,s] = S

        
        Q_R = scl.cholesky(Q + 1e-12 * np.eye(params["dgy"])) #of shape (dgy , dgy)
        # noise_cov_R = scl.cholesky(noise_cov + 1e-12 * np.eye(params["dim2"])) #of shape ((dgx*dgy) , (dgx*dgy))
        # self.noise_cov_inv_R = self.fwd_slash(np.eye(params["dim2"]), noise_cov_R)
        self.Q_R_inv = self.fwd_slash(np.eye(params["dgy"]), Q_R)
        # logDetSigma = 3 * 2 *np.sum(np.log(np.diag(noise_cov_R))) #3 for h, u, v. This is 3 * log(det(noise_cov))
        logDetSigma = 3 * params["dgx"] * 2 *np.sum(np.log(np.diag(Q_R))) 
        #3 for h, u, v. This is 3 * log(det(noise_cov)). params["dgx"] is the number of Q on the diagonal block matrix
        self.log_nc = - 0.5 * (logDetSigma + params["dimx"] * np.log(2*np.pi))
        

        params["time"] = 0.0

        self.params = params
        self._setup()

    def _setup(self):
        files = ["mcmc_filter_file"]
        for _, file in enumerate(files):
            filename = os.path.basename(self.params[file])
            filename = "%s_%05d.h5" %(filename.split(".h5")[0], self.params["isim"])
            filename = os.path.join(os.path.dirname(self.params[file]), filename)
            self.params[file] = filename


    def fwd_slash(self, mat1, mat2):
        """ equivalent to mat1/mat2 in MATLAB. That is to solve for x in: x * mat2 = mat1
        This is equivalent to: mat2.T * x.T = mat1.T """
        return np.linalg.solve(mat2.T, mat1.T).T

    def logmvnpdf_g(self, X_minus_mu, sigd): 
        """evaluate the Multivariate Normal Distribution at X with mean mu and a
        a cov that is a diagonal matrix. 
        Inputs: d           - dimension
                X_minus_mu  - d x N vector (mean)
                sigd        - 1x1 value sqrt(const) when sigma = const * np.ones(d) ## dx1 vector (sqrt of diag of Sigma)  
        """
        xSigSqrtinv = X_minus_mu/sigd
        logSqrtDetSigma = np.log(sigd) * self.params["dimo"] #np.sum(np.log(sigd))
        quadform = np.sum(xSigSqrtinv**2, axis=0)
        ly = -0.5 * quadform - logSqrtDetSigma - 0.5 * self.params["dimo"] * np.log(2*np.pi)
        return ly

    def logmvnpdf_f(self,X_minus_mu, N): 
        """evaluate the Multivariate Normal Distribution at X with mean mu and a
        a cov that is a diagonal matrix. 
        Inputs: d           - dimension
                X_minus_mu  - dxN vector (mean)
                noise_cov_inv     - (dgx*dgy, dgx*dgy)   
        """
        A = np.matmul((X_minus_mu[0:self.params["dim2"]].T).reshape(N,self.params["dgx"],
                     self.params["dgy"]), self.Q_R_inv).reshape(N,-1) #N , dgx * dgy
        B = np.matmul((X_minus_mu[self.params["dim2"]:2*self.params["dim2"]].T).reshape(N,self.params["dgx"],
                     self.params["dgy"]), self.Q_R_inv).reshape(N,-1) #N , dgx * dgy
        C = np.matmul((X_minus_mu[2*self.params["dim2"]:3*self.params["dim2"]].T).reshape(N,self.params["dgx"],
                     self.params["dgy"]), self.Q_R_inv).reshape(N,-1) #N , dgx * dgy
        xSigSqrtinv = np.hstack((A,B,C)).T #shape = 3 dgx dgy , N

        quadform = np.sum(xSigSqrtinv**2, axis=0)
        ly = -0.5 * quadform + self.log_nc
        return ly


    def compute_pi(self, yn, zn, Czn, qznm1j):
        """ Compute 
        pi_n^N propto gn(y_n|loc,z_n) (1/N) sum_{j=1}^N fn(z_n|z_{n-1}^{(j)})"""
        lg = self.logmvnpdf_g(yn - Czn, self.params['sig_y']) #of shape (dimo,)
        lf = self.logmvnpdf_f(zn - qznm1j, 1) #of shape (dimx , N)
        return (lg + lf)

    def run(self):
        """Main function"""
        print("Doing MCMC Filtering - Simul = %08d...." %self.params["isim"])
        starttime = time.time()
        
        
        self.n1_sampling()

        #self.plot_for_testing(self.nstep, self.params, self.mcmc_samples)

        for self.nstep in range(1, self.params["T"]):
            self.n_larger_than1_sampling()

            #self.plot_for_testing(self.nstep, self.params, self.mcmc_samples)

        print('MCMC-Filtering: Writing Simulation %d Results to File' %self.params["isim"])
        self._dump_restart() #only dump one time at the end

        endtime = time.time() - starttime
        print('MCMC-Filtering: Simul = %d, dimx = %d, T = %d, Elapsed = %.3f'
            % (self.params["isim"], self.params["dimx"], self.params["T"], endtime ))

    def _dump_restart(self):
        filename = "%s/mcmc_restart_%08d.h5" 
        filename = filename %(self.params["mcmc_restart_dir"],
                                                            self.params["isim"])

        with h5py.File(filename, "w") as fout:
            fout.create_dataset(name="t_simul", data=self.t_simul)
            fout.create_dataset(name="mcmc_samples", data=self.mcmc_samples)
            fout.attrs["ite"] = self.nstep

    def n1_sampling(self):
        """nstep=0 (t = 1)"""
        
        #expectation of phi(x)= x_n given y_{1:n} using MCMC
        self.mcmc_samples[:,0] = self.params["x_star"]
        

        self.tsteps[0], sol = solve(1, self.params["x_star"], self.params)
        self.t_simul += self.tsteps[0]
        self.params["time"] = self.t_simul


        if (self.nstep+1) % self.params["t_freq"] == 0:
            y1 = self.data(self.t_simul/3600)
            self.floater_loc(1)

            noise = self.compute_noise(1) #only for one sample
            zn = sol + noise


            #MCMC at time 1 to sample from \pi_1(x_1|y_1)  = g(y_1|loc,x_1)  f(x_1|x_0)
            old_pi = self.logmvnpdf_g(y1 - zn[self.obs_indices],self.params['sig_y'] ) \
                    + self.logmvnpdf_f(noise, 1) #noise is zn - sol where sol is the mean of f(x1|x0)
                                
            #print("old_pi", old_pi)
            count = 0           
            for i in range (self.mcmc_filt_iter): 
                # noise = self.params["sig_mcmc_filt"] * \
                #                      np.random.normal(size=(self.params['dimx']))
                noise = self.compute_noise(1, const=self.params["sig_mcmc_filt"]/self.params["sig_x"] )
                znp = zn + noise
                new_pi = self.logmvnpdf_g(y1 - znp[self.obs_indices],self.params['sig_y'] ) \
                        + self.logmvnpdf_f(znp - sol, 1)
                alpha = new_pi - old_pi
                #print("new_pi", new_pi, alpha)
                if np.log(np.random.uniform()) <= alpha:
                    zn = znp
                    old_pi = new_pi
                    count += 1
                    if i >= self.params["burn_in"]:
                        self.samples[0,:,i-self.params["burn_in"]] = znp     
                else:
                    if i >= self.params["burn_in"]:
                        self.samples[0,:,i-self.params["burn_in"]] = zn 

            self.accep_rate[0] = count / self.mcmc_filt_iter
            print('isim %d: Accep_rate at time step 0 is %0.3f, sig_mcmc = %.3e' % (self.params["isim"], self.accep_rate[0],
                         self.params["sig_mcmc_filt"]))

            if self.accep_rate[0] > 0.25000:
                self.params["sig_mcmc_filt"] *= 1.2
            elif self.accep_rate[0] < 0.20000: 
                self.params["sig_mcmc_filt"] *= 0.85

            self.mcmc_samples[:, 1] = np.sum(self.samples[0,:,:],axis=1) / self.params["mcmc_N"]
            self.obs_indices = np.zeros(self.params["N_f"], dtype = int)
        else:
            if self.params["add_noise_sig_every_dt"]:
                self.samples[0,:,:] = np.transpose([sol] * self.params["mcmc_N"])
                noise = self.compute_noise(self.params["mcmc_N"])
                self.samples[0,:,:] += noise
                self.mcmc_samples[:, 1] = np.mean(self.samples[0,:,:], axis=1)
            else:
                self.samples[0,:,:] = np.transpose([sol] * self.params["mcmc_N"])
                self.mcmc_samples[:, 1] = sol


    def n_larger_than1_sampling(self):
        """ nstep >= 1 (t >= 2) """
        if (self.nstep+1) % self.params["t_freq"] == 0:
            
            # self.samples.shape[0] = self.params["t_freq"]. Last element is self.params["t_freq"]-1
            # the element before the last is self.params["t_freq"]-2
            # self.samples[from 0 to self.params["t_freq"]-2] already computed in the other part of the if statement
            # when self.params["t_freq"] = 1, t0 = -1. But self.samples.shape[0] = 1, therefore self.samples[-1,:,:] is the only 
            # element in self.samples and this element will get updated below because 
            # self.samples[t0+1,:,:] = self.samples[0] and that is the same as self.samples[-1] 

            j = np.random.choice(self.params["mcmc_N"]) # initialize the chain at some random particle j
            t0 = self.params["t_freq"]-2 
            znm1j = self.samples[t0,:,j]
            self.tsteps[self.nstep], sol = solve(1, znm1j, self.params) 
            self.t_simul += self.tsteps[self.nstep]
            self.params["time"] = self.t_simul
            yn = self.data(self.t_simul/3600) 


            #noise = self.compute_noise(self.params["mcmc_N"])
            #sol += noise

            
            #### The following line added after commenting noise and sol += noise above
            noise = self.compute_noise(1)
            self.floater_loc(j) #will return the average location of all floaters taken over the N particles
            zn = sol + noise
            old_pi = self.compute_pi(yn, zn, zn[self.obs_indices], sol)
            count = 0
            #print("old_pi", old_pi)
            for i in range (self.mcmc_filt_iter): 
                # noise = self.params["sig_mcmc_filt"] * \
                #                      np.random.normal(size=(self.params['dimx']))
                noise = self.compute_noise(1, const=self.params["sig_mcmc_filt"]/self.params["sig_x"] )

                znp = zn + 0.1 * noise
                new_pi = self.compute_pi(yn, znp, znp[self.obs_indices], sol)
                alpha = new_pi - old_pi

                lg_u = np.log(np.random.uniform())
                #print("alpha", alpha, "log u", lg_u, lg_u <= alpha)
                
                if lg_u <= alpha:
                    zn = znp
                    old_pi = new_pi
                    count += 1
                    if i >= self.params["burn_in"]:
                        self.samples[t0+1,:,i-self.params["burn_in"]] = zn
                    
                else:
                    if i >= self.params["burn_in"]:
                    	self.samples[t0+1,:,i-self.params["burn_in"]] = zn 

            self.accep_rate[self.nstep] = count / self.mcmc_filt_iter
            print('isim %d: Accep_rate at time step %d is %0.3f, sig_mcmc = %.3e'
                     % (self.params["isim"], self.nstep, self.accep_rate[self.nstep],
                         self.params["sig_mcmc_filt"]))

            if self.accep_rate[self.nstep] > 0.29000:
                self.params["sig_mcmc_filt"] *= 1.15
            elif self.accep_rate[self.nstep] < 0.18000: 
                self.params["sig_mcmc_filt"] *= 0.8

            self.mcmc_samples[:, self.nstep + 1] = np.sum(self.samples[t0+1,:,:],axis=1) / self.params["mcmc_N"]
            self.obs_indices = np.zeros(self.params["N_f"], dtype = int)
        else:
            t1 = self.nstep % self.params["t_freq"] 
            self.tsteps[self.nstep], self.samples[t1,:,:] = solve(self.params["mcmc_N"], self.samples[t1-1,:,:], self.params) 
            self.t_simul += self.tsteps[self.nstep]
            self.params["time"] = self.t_simul
            if self.params["add_noise_sig_every_dt"]:
                noise = self.compute_noise(self.params["mcmc_N"])
                self.samples[t1,:,:] += noise

            self.mcmc_samples[:, self.nstep + 1] = np.mean(self.samples[t1,:,:], axis=1)

      
    def compute_noise(self, N_samples, const=1):
        if N_samples == 1:
            noise = np.zeros(self.params["dimx"])
        else:
            noise = np.zeros((self.params["dimx"], N_samples)) #dimx = dgy*dgx

        for i in range(3):
            if N_samples == 1:
                # a =  const * self.var_sqrt[:,None] * np.random.normal(size=(self.params['noise_modes_num'], 1)) #sample ~ N(0, sig_x^2/j^2)
                # b = const * self.var_sqrt * np.random.normal(size=(1, self.params['noise_modes_num']))   #sample ~ N(0, sig_x^2/j^2)
                # rand = (self.B @ a) @ (b @ self.A.T)

                epsil = const * self.var_sqrt * np.random.normal(size=(self.params['noise_modes_num'], self.params['noise_modes_num']))
                Xi = self.B @ epsil @ self.A.T
                noise[i*self.params["dim2"]:(i+1)*self.params["dim2"]] = Xi.reshape(-1)
            else:
                # a =  const * self.var_sqrt[:,None] * np.random.normal(size=(N_samples,self.params['noise_modes_num'], 1)) #sample ~ N(0, sig_x^2/j^2)
                # b =  const * self.var_sqrt * np.random.normal(size=(N_samples, 1, self.params['noise_modes_num'])) #sample ~ N(0, sig_x^2/j^2)
                # c = a@b #of shape (N_samples, noise_modes_num, noise_modes_num)
                # rand = self.B @ c @ self.A.T #of shape (N, dgy, dgx)
                # rand = rand.reshape(rand.shape[0], rand.shape[1]*rand.shape[2]) #of shape (N, dgx*dgy)
                # noise[i*self.params["dim2"]:(i+1)*self.params["dim2"],:] =  rand.T #of shape (dim2 , N)

                epsil = const * self.var_sqrt * np.random.normal(size=(N_samples, self.params['noise_modes_num'], self.params['noise_modes_num']))
                #epsil of shape (N, J, J)
                Xi = self.B @ epsil @ self.A.T #of shape (N, dgy, dgx)
                Xi = Xi.reshape(Xi.shape[0], Xi.shape[1]*Xi.shape[2]) #of shape (N, dgy*dgx)
                noise[i*self.params["dim2"]:(i+1)*self.params["dim2"],:] =  Xi.T #of shape (dgy*dgx , N)


        #print("max_noise= ", np.amax(noise))
        #print(noise[0:self.params["dim2"],0].reshape(self.params["dgx"], self.params["dgy"]))
        return noise #of shape (3*dgx*dgy, N)

                            
    def floater_loc(self, pj):
        #run this only when (nstep+1) % params["t_freq"] == 0 given a segment of the path of particle pj
        #update floaters positions using the path segment from last time of observing to the current 
        #time of observing
    
        # self.floater_xy at step = self.nstep+1 - self.params["t_freq"] is available and its the average over all particles

        for step in range(self.nstep+1 - self.params["t_freq"], self.nstep+1):
            t0 = step % self.params["t_freq"]
            signal = self.samples[t0,:,pj]
            HEIGHT = signal[0:self.params["dim2"]].reshape(self.params["dgx"],self.params["dgy"])
            UVELOCITY = signal[self.params["dim2"]:2*self.params["dim2"]].reshape(self.params["dgx"],self.params["dgy"])
            VVELOCITY = signal[2*self.params["dim2"]:3*self.params["dim2"]].reshape(self.params["dgx"],self.params["dgy"])
            surr_pts_ind = np.zeros((2, 4 * self.params["N_f"]), dtype = int)
            surr_pts_xy = np.zeros((2, 4 * self.params["N_f"]))
            w = np.zeros((self.params["N_f"], 4))
            for i in range(self.params["N_f"]):
                c1 = (self.x_mesh >= self.floater_xy[step, 2*i])
                right = self.x_ind[c1]
                c2 = (self.x_mesh <= self.floater_xy[step, 2*i])
                left  = self.x_ind[c2]
                c3 = (self.y_mesh >= self.floater_xy[step, 2*i+1])
                above = self.y_ind[c3]
                c4 = (self.y_mesh <= self.floater_xy[step, 2*i+1])
                below = self.y_ind[c4]

                surr_pts_ind[0, 4*i : 4*(i+1)]  = np.array([right[0],right[0], left[-1], left[-1]])
                surr_pts_ind[1, 4*i : 4*(i+1)]  = np.array([above[0],below[-1],below[-1],above[0]])
                surr_pts_xy[0, 4*i : 4*(i+1)] = self.x_mesh[surr_pts_ind[0, 4*i : 4*(i+1)]]
                surr_pts_xy[1, 4*i : 4*(i+1)] = self.y_mesh[surr_pts_ind[1, 4*i : 4*(i+1)]]

                for j in range(4):
                    dist_from_float  = np.sqrt((surr_pts_xy[0,4*i+j] - self.floater_xy[step, 2*i])**2\
                                            + (surr_pts_xy[1,4*i+j] - self.floater_xy[step, 2*i+1])**2)
                    w[i,j] = np.maximum(0.0, self.hyp - dist_from_float)
                w[i,:] = w[i,:] / np.sum(w[i,:])

                u_float = np.zeros(self.params["N_f"])
                v_float = np.zeros(self.params["N_f"])
                ### IMPORTANT NOTE: 1d Array [a0,a1,a2,a3] in Python when reshaped to 2d array, it is 
                ### [[a0, a1]
                ###  [a2, a3]] Reshaped row-wise. Different from Matlab!
                for j in range(4):
                    u_float[i] += UVELOCITY[surr_pts_ind[0, 4*i+j], surr_pts_ind[1, 4*i+j]] * w[i,j]
                    v_float[i] += VVELOCITY[surr_pts_ind[0, 4*i+j], surr_pts_ind[1, 4*i+j]] * w[i,j]

                self.floater_xy[step+1, 2*i : 2*(i+1)] = self.floater_xy[step, 2*i : 2*(i+1)]\
                                     + self.tsteps[step] * np.array([u_float[i], v_float[i]])


            for i in range(self.params["N_f"]):
                if self.floater_xy[step+1, 2*i] < self.x_mesh[0] or self.floater_xy[step+1, 2*i] > self.x_mesh[-1] or\
                    self.floater_xy[step+1, 2*i+1] < self.y_mesh[0] or self.floater_xy[step+1, 2*i+1] > self.y_mesh[-1]:
                    self.params["N_f"] -= 1
                    self.floater_xy = np.delete(self.floater_xy, [2*i, 2*i+1], axis = 1)
                    # self.floater_height = np.delete(self.floater_height, i, axis =1 )
                    self.obs_indices = np.zeros(self.params["N_f"], dtype = int)
                    self.obs_xy = np.zeros(2 * self.params["N_f"])
                    break

            ##floaters height
            # h_float = np.zeros(self.params["N_f"])
            # w = np.zeros((self.params["N_f"], 4))
            # surr_pts_ind = np.zeros((2, 4 * self.params["N_f"]), dtype = int)
            # surr_pts_xy = np.zeros((2, 4 * self.params["N_f"]))
            # for i in range(self.params["N_f"]):
            #     c1 = (self.x_mesh >= self.floater_xy[step+1, 2*i])
            #     right = self.x_ind[c1]
            #     c2 = (self.x_mesh < self.floater_xy[step+1, 2*i])
            #     left  = self.x_ind[c2]
            #     c3 = (self.y_mesh >= self.floater_xy[step+1, 2*i+1])
            #     above = self.y_ind[c3]
            #     c4 = (self.y_mesh < self.floater_xy[step+1, 2*i+1])
            #     below = self.y_ind[c4]

            #     surr_pts_ind[0, 4*i : 4*(i+1)]  = np.array([right[0],right[0], left[-1], left[-1]])
            #     surr_pts_ind[1, 4*i : 4*(i+1)]  = np.array([above[0],below[-1],below[-1],above[0]])
            #     surr_pts_xy[0, 4*i : 4*(i+1)] = self.x_mesh[surr_pts_ind[0, 4*i : 4*(i+1)]]
            #     surr_pts_xy[1, 4*i : 4*(i+1)] = self.y_mesh[surr_pts_ind[1, 4*i : 4*(i+1)]]

            #     for j in range(4):
            #         dist_from_float = np.sqrt((surr_pts_xy[0,4*i+j] - self.floater_xy[step+1, 2*i])**2\
            #                                 + (surr_pts_xy[1,4*i+j] - self.floater_xy[step+1, 2*i+1])**2)
            #         w[i,j] = np.maximum(0.0, self.hyp - dist_from_float)
            #     w[i,:] = w[i,:] / np.sum(w[i,:])
                

            #     for j in range(4):
            #         ### IMPORTANT NOTE: 1d Array [a0,a1,a2,a3] in Python when reshaped to 2d array, it is 
            #         ### [[a0, a1]
            #         #    [a2, a3]] Reshaped row-wise. Different from Matlab!
            #         h_float[i] += HEIGHT[surr_pts_ind[0, 4*i+j], surr_pts_ind[1, 4*i+j]] * w[i,j]

            #     self.floater_height[step+1, i] = h_float[i]


        #save obs locations to files
        filename = "%s/floaters_loc_isim%08d_nstep%08d.h5" 
        filename = filename %(self.params["floaters_info_dir"],self.params["isim"], self.nstep+1)
        fout = h5py.File(filename, "a")
        fout.create_dataset(name="obs_loc", data=self.floater_xy[step+1,:])
        fout.close()

        #here step = self.nstep:
        w = np.zeros((self.params["N_f"], 4))
        surr_pts_ind = np.zeros((2, 4 * self.params["N_f"]), dtype = int)
        surr_pts_xy = np.zeros((2, 4 * self.params["N_f"]))
        for i in range(self.params["N_f"]):
            c1 = (self.x_mesh >= self.floater_xy[step+1,2*i])
            right = self.x_ind[c1]
            c2 = (self.x_mesh < self.floater_xy[step+1,2*i])
            left  = self.x_ind[c2]
            c3 = (self.y_mesh >= self.floater_xy[step+1,2*i+1])
            above = self.y_ind[c3]
            c4 = (self.y_mesh < self.floater_xy[step+1,2*i+1])
            below = self.y_ind[c4]

            surr_pts_ind[0, 4*i : 4*(i+1)]  = np.array([right[0],right[0], left[-1], left[-1]])
            surr_pts_ind[1, 4*i : 4*(i+1)]  = np.array([above[0],below[-1],below[-1],above[0]])
            surr_pts_xy[0, 4*i : 4*(i+1)] = self.x_mesh[surr_pts_ind[0, 4*i : 4*(i+1)]]
            surr_pts_xy[1, 4*i : 4*(i+1)] = self.y_mesh[surr_pts_ind[1, 4*i : 4*(i+1)]]

            for j in range(4):
                dist_from_float = np.sqrt((surr_pts_xy[0,4*i+j] - self.floater_xy[step+1,2*i])**2\
                                        + (surr_pts_xy[1,4*i+j] - self.floater_xy[step+1,2*i+1])**2)
                w[i,j] = np.maximum(0.0, self.hyp - dist_from_float)
            w[i,:] = w[i,:] / np.sum(w[i,:])

        for i in range(self.params["N_f"]):
            if self.params['obs_loc_choice'] == 0:
                surr_pt_ind = np.argmax(w[i,:])
            else:
                surr_pt_ind = np.random.choice(4, 1, p=w[i,:])
            self.obs_indices[i] = surr_pts_ind[1, 4*i + surr_pt_ind] +\
                                             surr_pts_ind[0, 4*i + surr_pt_ind] * self.params["dgy"]
            self.obs_xy[2*i : 2*(i+1)] = [surr_pts_xy[0, 4*i + surr_pt_ind],
                                                  surr_pts_xy[1, 4*i + surr_pt_ind]]


        self.obs_indices = np.concatenate((self.obs_indices + self.params["dim2"], 
                                self.obs_indices + 2*self.params["dim2"])) 


    def plot_for_testing(self, nstep, params, mcmc_samples):

        if nstep == 0:
            cmap_= "RdBu" 

            x = np.mgrid[0:params["dgx"]]* params["dx"] # Zonal distance coordinate (m)
            y = np.mgrid[0:params["dgy"]]* params["dy"] # Meridional distance coordinate (m)

            self.fig = plt.figure(figsize=(28,8))
            plt.tight_layout()
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)

            signal = mcmc_samples[:, 1]
            h = signal[0:params["dim2"]].reshape((params["dgx"], params["dgy"]))
            u = signal[params["dim2"] : 2*params["dim2"]].reshape((params["dgx"], params["dgy"]))
            v = signal[2*params["dim2"]:].reshape((params["dgx"], params["dgy"]))

            self.pcm1 = self.ax1.imshow(np.transpose(u), cmap=cmap_, extent = [np.amin(x), np.amax(x), np.amin(y), np.amax(y)],
                             origin='lower',interpolation='spline16')
            divider = make_axes_locatable(self.ax1)
            cax = divider.append_axes('right', size='3%', pad=0.04, axes_class= plt.Axes)
            cb1 = self.fig.colorbar(self.pcm1, cax=cax)

            self.pcm2 = self.ax2.imshow(np.transpose(v), cmap=cmap_, extent = [np.amin(x), np.amax(x), np.amin(y), np.amax(y)],
                                     origin='lower',interpolation='spline16')
            divider = make_axes_locatable(self.ax2)
            cax = divider.append_axes('right', size='3%', pad=0.04, axes_class= plt.Axes)
            cb2 = self.fig.colorbar(self.pcm2, cax=cax)


            self.pcm3 = self.ax3.imshow(np.transpose(h), cmap=cmap_, extent = [np.amin(x), np.amax(x), np.amin(y), np.amax(y)],
                                     origin='lower',interpolation='spline16')
            divider = make_axes_locatable(self.ax3)
            cax = divider.append_axes('right', size='3%', pad=0.04, axes_class= plt.Axes)
            cb3 = self.fig.colorbar(self.pcm3, cax=cax)


            #self.pcm1.set_clim([0,2])
            self.ax1.set_aspect(1)
            self.ax2.set_aspect(1)
            self.ax3.set_aspect(1)


            self.ax1.set_xlabel('x')
            self.ax1.set_ylabel('y')
            self.ax2.set_xlabel('x')
            self.ax2.set_ylabel('y')
            plt.tight_layout()
        
            self.txt1 = self.ax1.text(np.amin(x), np.amax(y)+1, 'U - Time = %.3f hours' % (params["time"]/3600), fontsize=20)
            self.txt2 = self.ax2.text(np.amin(x), np.amax(y)+1, 'V - Time = %.3f hours' % (params["time"]/3600), fontsize=20)
            self.txt3 = self.ax3.text(np.amin(x), np.amax(y)+1, 'eta - Time = %.3f hours' % (params["time"]/3600), fontsize=20)
            self.fig.canvas.draw()
            plt.pause(0.01)

        else:
            signal = mcmc_samples[:, nstep + 1]
            h = signal[0:params["dim2"]].reshape((params["dgx"], params["dgy"]))
            u = signal[params["dim2"] : 2*params["dim2"]].reshape((params["dgx"], params["dgy"]))
            v = signal[2*params["dim2"]:].reshape((params["dgx"], params["dgy"]))


            self.pcm1.set_data(np.transpose(u))
            self.pcm2.set_data(np.transpose(v))
            self.pcm3.set_data(np.transpose(h))

            self.txt1.set_text('U - Time = %.3f hours' % (params["time"]/3600))
            self.txt2.set_text('V - Time = %.3f hours' % (params["time"]/3600));
            self.txt3.set_text('eta - Time = %.3f hours' % (params["time"]/3600))

            self.fig.canvas.draw()
            plt.pause(0.01)






