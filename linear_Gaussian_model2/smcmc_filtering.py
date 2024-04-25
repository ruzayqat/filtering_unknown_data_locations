import os
import time
import numpy as np
import h5py
from data_tools import get_data



def smcmc_filter(params, isim):
    """main function of SMCMC filtering"""
    np.random.seed(isim)
    mcmc_f = SMCMC_Filter(isim, params)
    mcmc_f.run()



class SMCMC_Filter():
    """Filtering Using SMCMC main class
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
        self.smcmc_filter_mean = np.zeros((params["dx"],params["T"]+1))
        self.mcmc_filt_iter = params["mcmc_N"] + params["burn_in"]
        self.mcmc_samples = np.zeros((params['dx'],params["mcmc_N"]))
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
                            self.params['sig_x'])
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

        x_star = get_data(self.params, 0, "signal") 
        self.smcmc_filter_mean[:,0] = x_star

        #self._dump_restart()
        for self.nstep in range(self.params["T"]):
            self.run_and_sample()
            #self._dump_restart()
        # print('MCMC-Filtering: Writing Simulation %d Results to File' %self.params["isim"])
        self._dump_restart()

        endtime = time.time() - starttime
        # print('MCMC-Filtering: Simul = %d, dimx = %d, T = %d, Elapsed = %.3f'
        #     % (self.params["isim"], self.params["dx"], self.params["T"], endtime ))

    def _dump_restart(self):
        if "smcmc_file" not in self.params:
            filename = "%s/smcmc_%08d.h5" 
            self.params["smcmc_file"] = filename %(self.params["smcmc_dir"],
                                                            self.params["isim"])

        dir_ = os.path.dirname(self.params["smcmc_file"])
        os.makedirs(dir_, exist_ok=True)

        if os.path.isfile(self.params["smcmc_file"]):
            os.remove(self.params["smcmc_file"])

        with h5py.File(self.params["smcmc_file"], "w") as fout:
            fout.create_dataset(name="smcmc_filter", data=self.smcmc_filter_mean)
            fout.attrs["ite"] = self.nstep



    def run_and_sample(self):
        if (self.nstep+1)/self.params["t_freq"] == 1:
            #exact sampling at the first time of observing
            sol = self.params["A"] * self.smcmc_filter_mean[:,self.nstep]
            y1 = get_data(self.params, self.nstep, "data")
            noise = self.params['sig_x'] * np.random.normal(size=(self.params['dx']))
            zn = sol + noise
            #MCMC at observation time t_1 to sample from \pi_1(Z_{t_1}|y_1)
            old_pi = self.logmvnpdf(self.params['dy'], y1 - zn[self.params["s_freq"]-1::self.params["s_freq"]],
                                self.params['sig_y']) \
                    +self.logmvnpdf(self.params['dx'], noise, 
                                self.params['sig_x'])
            count = 0           
            for i in range (self.mcmc_filt_iter): 
                noise = self.params["sig_mcmc_filt"] * np.random.normal(size=(self.params['dx']))
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
                    self.mcmc_samples[:,i-self.params["burn_in"]] = zn 

            accep_rate = count / self.mcmc_filt_iter
            # print('isim %d: Accep_rate at time t_1 = %0.3f' % (self.params["isim"], accep_rate))

            if accep_rate > 0.25000:
                self.params["sig_mcmc_filt"] *= 1.2
            elif accep_rate < 0.20000: 
                self.params["sig_mcmc_filt"] *= 0.8

            self.smcmc_filter_mean[:, self.nstep+1] = np.mean(self.mcmc_samples, axis=1)
            

        elif (self.nstep+1) % self.params["t_freq"] == 0 and ((self.nstep+1)/self.params["t_freq"]) > 1:
            #at observation times > t_1, sample from the an approximate distribution 
            yn = get_data(self.params, self.nstep, "data")

            j = np.random.choice(self.params["mcmc_N"]) # initialize the chain at some random particle j
            
            berv_Z_tn = self.params["A"] * self.mcmc_samples
            noise = self.params['sig_x'] * np.random.normal(size=(self.params['dx']))
            Z_tn = berv_Z_tn[:, j] + noise
            old_pi = self.compute_pi_1(yn, Z_tn, Z_tn[self.params["s_freq"]-1::self.params["s_freq"]], berv_Z_tn[:, j])
            old_j = j
            choices = np.array([0,1,-1], dtype=int)
            q_prob = 0.33
            prob = np.array([1-2*q_prob, q_prob, q_prob]) #q =  0.33

            count = 0
            for i in range (self.mcmc_filt_iter): 

                #propose new sample from a RW proposal 
                noise = self.params["sig_mcmc_filt"] * np.random.normal(size=(self.params['dx']))
                Z_tn_p = Z_tn + noise
                j = old_j + np.random.choice(choices, p=prob)

                if old_j == 0:
                    j += 1
                if old_j == self.params["mcmc_N"]-1:
                    j -= 1
                #compute new pi at the proposal (Z_tn_p, j)
                new_pi = self.compute_pi_1(yn, Z_tn_p, Z_tn_p[self.params["s_freq"]-1::self.params["s_freq"]], berv_Z_tn[:, j])

                if old_j == 0 or old_j == self.params["mcmc_N"]-1:
                    new_pi += np.log(q_prob) #since the proposal for moving from j=1 to j=0 and from j=mcmc_N to j = MCMC_N - 1 is not symmetric
                                            #Therefore p(j_old| j) = 1 and p(j | j_old) = q


                alpha = new_pi - old_pi
                if np.log(np.random.uniform()) <= alpha:
                    Z_tn = Z_tn_p
                    old_pi = new_pi
                    old_j = j
                    count += 1

                if i >= self.params["burn_in"]:
                    self.mcmc_samples[:,i-self.params["burn_in"]] = Z_tn  

            accep_rate = count / self.mcmc_filt_iter
            # if (self.nstep+1) % 5 == 0:
            #     print('isim %d: Accep_rate at time %d = %0.3f, sig_mcmc = %.5f'
            #          % (self.params["isim"], self.nstep, accep_rate,
            #              self.params["sig_mcmc_filt"]))

            if accep_rate > 0.28000:
                self.params["sig_mcmc_filt"] *= 1.2
            elif accep_ratelf.nstep < 0.20000: 
                self.params["sig_mcmc_filt"] *= 0.8

            self.smcmc_filter_mean[:, self.nstep + 1] = np.mean(self.mcmc_samples,axis=1)
        else:
            #just forward the dynamics until next observational time
            if self.nstep == 0:
                sol = self.params["A"] * self.smcmc_filter_mean[:,0]
                if self.params["add_noise_sig_every_dt"]:
                    self.mcmc_samples = np.transpose([sol] * self.params["mcmc_N"])
                    noise = self.params['sig_x'] * np.random.normal(size=(self.params['dx'], self.params["mcmc_N"] ))
                    self.mcmc_samples += noise
                    self.smcmc_filter_mean[:, self.nstep + 1] = np.mean(self.mcmc_samples)
                else:
                    self.mcmc_samples = np.transpose([sol] * self.params["mcmc_N"])
                    self.smcmc_filter_mean[:, self.nstep + 1] = sol
            else:    
                self.mcmc_samples = self.params["A"] * self.mcmc_samples
                if self.params["add_noise_sig_every_dt"]:
                    noise = self.params['sig_x'] * np.random.normal(size=(self.params['dx'], self.params["mcmc_N"] ))
                    self.mcmc_samples += noise

                self.smcmc_filter_mean[:, self.nstep + 1] = np.mean(self.mcmc_samples, axis=1)

