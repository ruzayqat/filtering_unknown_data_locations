# Linear Gaussian Model
This code compares between sequential MCMC (SMCMC) method and different ensemble methods: Kalman filter (KF) (exact in this case), ensemble KF (EnKF), ensemble transform KF (ETKF) and error-subspace transform KF (ESTKF).

example_input.yml conatins the parameters values. Any changes should be made only in this file.

First we run the KF to generate the data and the true solution, simply run this in the terminal:
```
python3 ./run_KF.py
```
This will create a folder named ``` data ``` and generate the observations (data) and save it in a file ```data.h5``` to be used by the other methods.

To run any other method like SMCMC, e.g., type this in the terminal:
```
python3 ./run_mcmc_filter.py
```

After finishing running the rest of the methods, run the following matlab code:
```
read_h5.m
```
to generate histograms of the absolute errors.
