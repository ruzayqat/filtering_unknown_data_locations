# Sequential Markov Chain Monte Carlo for Lagrangian Data Assimilation with Applications to Unknown Data Locations

We illustrate the performance of our Algorithm (SMCMC) in the paper (https://arxiv.org/abs/2305.00484) published in QJRMS in four different cases; two linear cases and two nonlinear cases (rotating shallow-water equations):

1) A linear Gaussian model 1. Here we show both efficiency and accuracy compared to competing ensemble methodologies: EnKF, ETKF and ESTKF. In contrast to how practitioners use the latter, we use moderately high number of ensembles 500-10^4. Note that for linear Gaussian models these methods are consistent and increasing the number of ensembles improves accuracy. The goal is to compare these methods at a higher accuracy.
To run this case do the following:

    - First we run the KF to generate the data and the true filter for the given parameters in ```example_input.yml ```, simply run the following in the terminal: 
      ```
  	   cd linear_Gaussian_model1
  	   python3 ./run_KF.py
      ```
      This will create a folder named "example_dx=#_dy=#_sigx=#_sigy=#" with # corresponds to the parameters dx, dy, sig_x and sig_y from ```example_input.yml ```.  It will also generate the observations (the data) ```data.h5``` and save it in
      the subfolder ```data``` to be used by the other methods. It will also save the true filter (the Kalman Filter) ```kalman_filter.h5``` in a subfolder ```kf```.
    - Then run the rest of the methods from the same main folder as follows:
      ```
      python3 ./run_smcmc_filter.py
      python3 ./run_enkf.py
      python3 ./run_etkf.py
      python3 ./run_estkf.py
      ```
      These runs will generate subfolders with the corresponding filter.
    - After finishing running the rest of the methods, open the matlab file ```read_h5.m``` and edit the parameters then run the following matlab code to generate histogram plots of the absolute errors.
      ```
      matlab ./read_h5.m
      ```
      
2) A linear Gaussian model 2. Here we show both efficiency and accuracy compared to the R-localaized EnKF methodology. This is similar to the previous example. First run the Kalman Filter code to generate the data and the true filter. Then call ```run_smcmc_filter.py``` and ```run_enkf_loc.py```. Finally, edit the matlab file ```read_h5.m``` and run to generate histogram plots.
   
The code for the shallow-water cases is available upon request.
