import os
import time
import numpy as np
import h5py
from data_tools import get_data



def mcmc_filter(params, isim):
    """main function of MCMC filtering"""
    np.random.seed(isim)
    mcmc_f = MCMC_Filter(isim, params)
    mcmc_f.run()



class MCMC_Filter():
    """Filtering Using MCMC main class
    pi_n(x_n|y_{1:n}) propto prod_{k=1}^n g(y_k|x_k) f(x_k|x_{k-1})
                        =   g(y_n|x_n) f(x_n|x_{n-1}) pi_{n-1}(x_{n-1})
    Approximate pi_{n-1} by an empirical distribution
    pi_{n-1}^N(x_{n-1}) = (1/N) sum_{j=1}^N delta_{X_{n-1}^{(j)}} (x_{n-1})
    Then we can approximate pi_n(x_n|y_{1:n}) by
    pi_n(x_n|y_{1:n})^N propto g(y_n|x_n) sum_{j=1}^N f(x_n|X_{n-1}^{(j)})
    Sample a uniform integer j from {1,...,N}. That is we use only one sample to approximate the emperical distribution
    We will use MCMC to sample from the following:
    tilde{pi}_n(x_n;j|y_{1:n})^N propto g(y_n|x_n) f(x_n|X_{n-1}^{(j)})
    """
    def __init__(self, isim, params):
        self.mcmc_samples = np.zeros((params["dx"],params["T"]+1))
        self.accep_rate = np.zeros(params["T"])
        self.mcmc_filt_iter = params["mcmc_N"] + params["burn_in"]
        self.samples = np.zeros((params['dx'],params["mcmc_N"]))
        params["isim"] = isim
        self.nstep = 0
        self.params = params
        

    def compute_pi_N(self, yn, zn, Czn, qznm1):
        """ Compute 
        pi_n^N propto gn(y_n|z_n) (1/N) sum_{j=1}^N fn(z_n|z_{n-1}^{(j)})"""
        lg = self.logmvnpdf(self.params['dy'], yn - Czn, self.params['sig_y']) #of shape (dy,)
        lf = self.logmvnpdf(self.params['dx'], zn.reshape(-1,1) - qznm1, self.params['sig_x']) #of shape (dx , N)
        max_lf = np.max(lf)
        return (lg + max_lf - np.log(self.params["mcmc_N"]) + np.log(np.mean(np.exp(lf - max_lf))))

    def compute_pi_1(self, yn, zn, Czn, qznm1j):
        """ Compute pi_n^1 propto gn(y_n|z_n) fn(z_n|z_{n-1}^{(j)})"""
        lg = self.logmvnpdf(self.params['dy'], yn - Czn,
                         self.params['sig_y'])
        lf = self.logmvnpdf(self.params['dx'], zn - qznm1j, 
                            self.params['sig_x'] )
        return (lg + lf)

    def logmvnpdf(self, d, X_minus_mu, sigd): 
        """evaluate the Multivariate Normal Distribution at X with mean mu and a
        a cov that is a diagonal matrix. 
        Inputs: d           - dimension
                X_minus_mu  - dxN vector (mean)
                sigd        - 1x1 value sqrt(const) when sigma = const * np.ones(d) ## dx1 vector (sqrt of diag of Sigma)  
        """
        xSigSqrtinv = X_minus_mu/sigd
        logSqrtDetSigma = np.log(sigd) * d #np.log(det(sigd * Identity)))
        quadform = np.sum(xSigSqrtinv**2, axis=0)
        ly = -0.5 * quadform - logSqrtDetSigma - 0.5 * d * np.log(2*np.pi)
        return ly

    def run(self):
        """Main function"""
        #print("Doing MCMC Filtering - Simul = %08d...." %self.params["isim"])
        starttime = time.time()
        self.n1_sampling()
        #self._dump_restart()
        for self.nstep in range(1, self.params["T"]):
            self.n_larger_than1_sampling()
            #self._dump_restart()
        # print('MCMC-Filtering: Writing Simulation %d Results to File' %self.params["isim"])
        self._dump_restart()

        endtime = time.time() - starttime
        # print('MCMC-Filtering: Simul = %d, dimx = %d, T = %d, Elapsed = %.3f'
        #     % (self.params["isim"], self.params["dx"], self.params["T"], endtime ))

    def _dump_restart(self):
        if "mcmc_file" not in self.params:
            filename = "%s/mcmc_%08d.h5" 
            self.params["mcmc_file"] = filename %(self.params["mcmc_dir"],
                                                            self.params["isim"])

        dir_ = os.path.dirname(self.params["mcmc_file"])
        os.makedirs(dir_, exist_ok=True)

        if os.path.isfile(self.params["mcmc_file"]):
            os.remove(self.params["mcmc_file"])

        with h5py.File(self.params["mcmc_file"], "w") as fout:
            fout.create_dataset(name="mcmc_filter", data=self.mcmc_samples)
            fout.attrs["ite"] = self.nstep

    def n1_sampling(self):
        """nstep=0"""
        
        #expectation of phi(x)= x_n given y_{1:n} using MCMC
        
        x_star = get_data(self.params, 0, "signal") 
        self.mcmc_samples[:,0] = x_star
        sol = self.params["A"] * x_star

        if (self.nstep+1) % self.params["t_freq"] == 0:
            y1 = get_data(self.params, self.nstep, "data")
            noise = self.params['sig_x'] * np.random.normal(size=(self.params['dx']))
            zn = sol + noise
            #MCMC at time 1 to sample from \pi_1(x_1|y_1)
            old_pi = self.logmvnpdf(self.params['dy'], y1 - zn[self.params["s_freq"]-1::self.params["s_freq"]],
                                self.params['sig_y']) \
                    +self.logmvnpdf(self.params['dx'], noise, 
                                self.params['sig_x'])
            count = 0           
            for i in range (self.mcmc_filt_iter): 
                noise = self.params["sig_mcmc_filt"] * \
                                     np.random.normal(size=(self.params['dx']))
                znp = zn + noise
                new_pi = self.logmvnpdf(self.params['dy'], y1 - znp[self.params["s_freq"]-1::self.params["s_freq"]],
                                self.params['sig_y']) \
                        +self.logmvnpdf(self.params['dx'], znp - sol, 
                                self.params['sig_x'])
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
            # print('isim %d: Accep_rate at time 1 = %0.3f' % (self.params["isim"], self.accep_rate[0]))

            if self.accep_rate[0] > 0.25000:
                self.params["sig_mcmc_filt"] *= 1.5
            elif self.accep_rate[0] < 0.20000: 
                self.params["sig_mcmc_filt"] *= 0.7

            self.mcmc_samples[:, 1] = np.mean(self.samples, axis=1)
        else:
            if self.params["add_noise_sig_every_dt"]:
                self.samples = np.transpose([sol] * self.params["mcmc_N"])
                noise = self.params['sig_x'] * np.random.normal(size=(self.params['dx'], self.params["mcmc_N"] ))
                self.samples += noise
                self.mcmc_samples[:, 1] = np.mean(self.samples)
            else:
                self.samples = np.transpose([sol] * self.params["mcmc_N"])
                self.mcmc_samples[:, 1] = sol


    def n_larger_than1_sampling(self):
        """ nstep >= 1 """
        if (self.nstep+1) % self.params["t_freq"] == 0:
            
            yn = get_data(self.params, self.nstep, "data")

            j = np.random.choice(self.params["mcmc_N"]) # initialize the chain at some random particle j
            noise = self.params['sig_x'] * np.random.normal(size=(self.params['dx']))

            if self.params["emp_use_all_samples"]:
                znm1 = np.copy(self.samples)
                sol = self.params["A"] * znm1
                zn = sol[:, j] + noise
            else:
                znm1 = self.samples[:,j]
                sol = self.params["A"] * znm1
                zn = sol + noise


            if self.params["emp_use_all_samples"]:
                old_pi = self.compute_pi_N(yn, zn, zn[self.params["s_freq"]-1::self.params["s_freq"]], sol)
            else:
                old_pi = self.compute_pi_1(yn, zn, zn[self.params["s_freq"]-1::self.params["s_freq"]], sol)
                
            count = 0
            for i in range (self.mcmc_filt_iter): 
                noise = self.params["sig_mcmc_filt"] * \
                                     np.random.normal(size=(self.params['dx']))
                znp = zn + noise
                if self.params["emp_use_all_samples"]:
                    new_pi = self.compute_pi_N(yn, znp, znp[self.params["s_freq"]-1::self.params["s_freq"]], sol)
                else:
                    new_pi = self.compute_pi_1(yn, znp, znp[self.params["s_freq"]-1::self.params["s_freq"]], sol)

                alpha = new_pi - old_pi
                if np.log(np.random.uniform()) <= alpha:
                    zn = znp
                    old_pi = new_pi
                    count += 1
                    if i >= self.params["burn_in"]:
                        self.samples[:,i-self.params["burn_in"]] = zn
                else:
                    if i >= self.params["burn_in"]:
                        self.samples[:,i-self.params["burn_in"]] = zn  

            self.accep_rate[self.nstep] = count / self.mcmc_filt_iter
            # if (self.nstep+1) % int(self.params["T"]/self.params["t_freq"]) == 0:
            #     print('isim %d: Accep_rate at time %d = %0.3f, sig_mcmc = %.5f'
            #          % (self.params["isim"], self.nstep, self.accep_rate[self.nstep],
            #              self.params["sig_mcmc_filt"]))

            if self.accep_rate[self.nstep] > 0.25000:
                self.params["sig_mcmc_filt"] *= 1.5
            elif self.accep_rate[self.nstep] < 0.20000: 
                self.params["sig_mcmc_filt"] *= 0.7

            self.mcmc_samples[:, self.nstep + 1] = np.mean(self.samples,axis=1)
        else:
            self.samples = self.params["A"] * self.samples
            if self.params["add_noise_sig_every_dt"]:
                noise = self.params['sig_x'] * np.random.normal(size=(self.params['dx'], self.params["mcmc_N"] ))
                self.samples += noise

            self.mcmc_samples[:, self.nstep + 1] = np.mean(self.samples, axis=1)

