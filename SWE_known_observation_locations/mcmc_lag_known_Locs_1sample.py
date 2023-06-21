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
        self.samples = np.zeros((params['dimx'], params["mcmc_N"]))
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

        
        # noise_cov = block_diag_einsum(Q, params["dgx"]) #of shape ((dgx*dgy) , (dgx*dgy))
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


    def compute_pi_N(self, yn, zn, Czn, qznm1):
        """ Compute 
        pi_n^N propto gn(y_n|loc,z_n) (1/N) sum_{j=1}^N fn(z_n|z_{n-1}^{(j)})"""
        lg = self.logmvnpdf_g(yn - Czn, self.params['sig_y']) #of shape (dimo,)
        lf = self.logmvnpdf_f(zn - qznm1, self.params["mcmc_N"]) #of shape (dimx , N)
        max_lf = np.max(lf)
        return (lg + max_lf - np.log(self.params["mcmc_N"]) + np.log(np.mean(np.exp(lf - max_lf))))

    def compute_pi_1(self, yn, zn, Czn, qznm1):
        """ Compute 
        pi_n^1 propto gn(y_n|loc,z_n) fn(z_n|z_{n-1}^{(j)})"""
        lg = self.logmvnpdf_g(yn - Czn, self.params['sig_y']) #of shape (dimo,)
        lf = self.logmvnpdf_f(zn - qznm1, 1) #of shape (dimx , 1)
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
            y1 = get_data(self.params, self.nstep, "data")
            obs_indices = get_floater_info(self.params, self.nstep, 'obs_indices')
            noise = self.compute_noise(1) #only for one sample
            zn = sol + noise


            #MCMC at time 1 to sample from \pi_1(x_1|y_1)  = g(y_1|loc,x_1)  f(x_1|x_0)
            old_pi = self.logmvnpdf_g(y1 - zn[obs_indices],self.params['sig_y'] ) \
                    + self.logmvnpdf_f(noise, 1) #noise is zn - sol where sol is the mean of f(x1|x0)
                                
            #print("old_pi", old_pi)
            count = 0           
            for i in range (self.mcmc_filt_iter): 
                # noise = self.params["sig_mcmc_filt"] * \
                #                      np.random.normal(size=(self.params['dimx']))
                noise = self.compute_noise(1, const=self.params["sig_mcmc_filt"]/self.params["sig_x"] )
                znp = zn + noise
                new_pi = self.logmvnpdf_g(y1 - znp[obs_indices],self.params['sig_y'] ) \
                        + self.logmvnpdf_f(znp - sol, 1)
                alpha = new_pi - old_pi
                if np.log(np.random.uniform()) <= alpha:
                    zn = znp
                    old_pi = new_pi
                    count += 1
                    if i >= self.params["burn_in"]:
                        self.samples[:,i-self.params["burn_in"]] = znp     
                else:
                    if i >= self.params["burn_in"]:
                        self.samples[:,i-self.params["burn_in"]] = zn 

            self.accep_rate[0] = count / self.mcmc_filt_iter
            print('isim %d: Accep_rate at time step 0 is %0.3f, sig_mcmc = %.3e' % (self.params["isim"], self.accep_rate[0],
                         self.params["sig_mcmc_filt"]))

            if self.accep_rate[0] > 0.25000:
                self.params["sig_mcmc_filt"] *= 1.4
            elif self.accep_rate[0] < 0.20000: 
                self.params["sig_mcmc_filt"] *= 0.8

            self.mcmc_samples[:, 1] = np.mean(self.samples,axis=1)
        else:
            if self.params["add_noise_sig_every_dt"]:
                self.samples = np.transpose([sol] * self.params["mcmc_N"])
                noise = self.compute_noise(self.params["mcmc_N"])
                self.samples += noise
                self.mcmc_samples[:, 1] = np.mean(self.samples, axis=1)
            else:
                self.samples = np.transpose([sol] * self.params["mcmc_N"])
                self.mcmc_samples[:, 1] = sol


    def n_larger_than1_sampling(self):
        """ nstep >= 1 (t >= 2) """
        if (self.nstep+1) % self.params["t_freq"] == 0:

            yn = get_data(self.params, self.nstep, "data")
            obs_indices = get_floater_info(self.params, self.nstep, 'obs_indices')
            j = np.random.choice(self.params["mcmc_N"]) # initialize the chain at some random particle j
            noise = self.compute_noise(1)
            if self.params["emp_use_all_samples"]:
                znm1 = np.copy(self.samples)
                self.tsteps[self.nstep], sol = solve(self.params["mcmc_N"], znm1, self.params)
                zn = sol[:, j] + noise
            else:
                znm1 = self.samples[:,j]
                self.tsteps[self.nstep], sol = solve(1, znm1, self.params)
                zn = sol + noise

            self.t_simul += self.tsteps[self.nstep]
            self.params["time"] = self.t_simul
            
            
            if self.params["emp_use_all_samples"]:
                old_pi = self.compute_pi_N(yn, zn, zn[obs_indices], sol)
            else:
                old_pi = self.compute_pi_1(yn, zn, zn[obs_indices], sol)

            count = 0
            #print("old_pi", old_pi)
            for i in range (self.mcmc_filt_iter): 
                noise = self.compute_noise(1, const=self.params["sig_mcmc_filt"]/self.params["sig_x"] )
                znp = zn + 0.1 * noise
                if self.params["emp_use_all_samples"]:
                    new_pi = self.compute_pi_N(yn, znp, znp[obs_indices], sol)
                else:
                    new_pi = self.compute_pi_1(yn, znp, znp[obs_indices], sol)
                
                alpha = new_pi - old_pi
                lg_u = np.log(np.random.uniform())
                
                if lg_u <= alpha:
                    zn = znp
                    old_pi = new_pi
                    count += 1
                    if i >= self.params["burn_in"]:
                        self.samples[:,i-self.params["burn_in"]] = zn
                    
                else:
                    if i >= self.params["burn_in"]:
                    	self.samples[:,i-self.params["burn_in"]] = zn 

            self.accep_rate[self.nstep] = count / self.mcmc_filt_iter
            print('isim %d: Accep_rate at time step %d is %0.3f, sig_mcmc = %.3e'
                     % (self.params["isim"], self.nstep, self.accep_rate[self.nstep],
                         self.params["sig_mcmc_filt"]))

            if self.accep_rate[self.nstep] > 0.29000:
                self.params["sig_mcmc_filt"] *= 1.15
            elif self.accep_rate[self.nstep] < 0.18000: 
                self.params["sig_mcmc_filt"] *= 0.8

            self.mcmc_samples[:, self.nstep + 1] = np.mean(self.samples,axis=1)
        else:
            self.tsteps[self.nstep], self.samples = solve(self.params["mcmc_N"], self.samples, self.params) 
            self.t_simul += self.tsteps[self.nstep]
            self.params["time"] = self.t_simul
            if self.params["add_noise_sig_every_dt"]:
                noise = self.compute_noise(self.params["mcmc_N"])
                self.samples += noise

            self.mcmc_samples[:, self.nstep + 1] = np.mean(self.samples, axis=1)

      
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






