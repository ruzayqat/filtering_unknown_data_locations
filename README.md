# Sequential Markov Chain Monte Carlo for Lagrangian Data Assimilation with Applications to Unknown Data Locations

We illustrate the performance of our Algorithms in the paper (https://arxiv.org/abs/2305.00484) in three different cases:

1) A linear Gaussian model. Here we show both efficiency and accuracy compared to competing ensemble methodologies like EnKF, ETKF and ESTKF. In contrast to how practitioners use the latter, we use moderately high number of ensembles 500-10^4 noting that for this model these methods are consistent and increasing number of ensembles improves accuracy.
To run this case do the following:

    - First we run the KF to generate the data and the true solution for the given parameters in ```example_input.yml ```, simply run this in   	the terminal. This will create a folder named data and generate the observations (data) and save it in a file data.h5 to be used by the other methods.
    ```
  	cd linear_Gaussian_model
  	python3 ./run_KF.py
    ```

    - The run the rest of the methods as follows:
      ```
      python3 ./run_mcmc_filter.py
      python3 ./run_enkf.py
      python3 ./run_etkf.py
      python3 ./run_estkf.py
      ```
    - After finishing running the rest of the methods, run the following matlab code to generate a histogram plot.
      ```
      matlab ./read_h5.m
      ```


2) A Rotating Shallow-Water Model Observed at Known Locations. We use NOAA data to set the initial conditions and boundaries and then simulate $Z_t,x_t$ to provide observations. The point is to assess the algorithm using synthetic drifter locations and observations, but we note the simulation scenario is set using real data from NOAA to make the case study as realistic as possible.

   - First make sure you download the data attached to the "Releases" named ```X_int0_121x121.npy``` and add it to the folder ```data``` inside the folder ```SWE_known_observation_locations```. Then run the following code from inside the folder ```SWE_known_observation_locations```:
     ```
      python3 ./run_mcmc_filter.py
     ```
     This will create a directory named ```example``` which will have the simulation results in.
     WARNING! This took about 10 hours to run on my machine. You can change the parameters settings in ```example_input.yml``` file.

   - To plot the results simply run this matlab code:
     ```
     matlab ./readplot_mcmcf_h5.m
     ```
   
3) A Rotating Shallow-Water Model Observed with Unknown Locations. We use real data for observer positions and velocities and show that our algorithm is effective at estimating the unknown velocity fields even when the locations of the observers are unknown.
   
    - First make sure you download the data attached to the "Releases" named ```X_int0_121x121.npy``` and add it to the folder ```data``` inside the folder ```SWE_unknown_observation_locations```. Then run the following code from inside the folder ```SWE_unknown_observation_locations```:
       ```
       python3 ./run_mcmc_filter.py
       python3 ./generate_prior_mean.py
       ```

       This will create a directory named ```example``` which will have the simulation results and the prior information.
       WARNING! This took about 10 hours to run on my machine. You can change the parameters settings in ```example_input.yml``` file.

     - To plot the results simply run this matlab code:
       ```
       matlab ./readplot_mcmcf_unknown_h5.m
       ```

It is worth noting that in Case 1. the true filter is known and is obtained through the Kalman filter (KF), however, in Cases 2. and 3. the true filter is unknown, and therefore, in these two cases we compare our results to a reference that is chosen to be the hidden signal used to generate the observations in Case 2. and the prior distribution in Case 3., respectively, where the later is estimated using 50 different simulations of the shallow-water dynamics with noise as will be described below.

