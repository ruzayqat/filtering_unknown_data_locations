"""Main module"""
from functools import partial
import numpy as np
import multiprocessing as mp
from smcmc_filtering import smcmc_filter
from data_tools import get_data
import yaml
import os
import shutil
import os.path
import time
from joblib import Parallel, delayed

def get_params(input_file):
    """Get simulation parameters"""
    print('Loading parameters from %s...' % input_file)
    with open(input_file, 'r') as (fin):
        params = yaml.safe_load(fin)
        params["dy"] = int(np.floor(params["dx"]/params["s_freq"]))

    return params



def main():
    """Mult-simulations of SMCMC - filtering"""

    print("Parsing input file...")
    input_file = "example_input.yml"
    params = get_params(input_file)

    dir_ = "example_dx=%d_dy=%d_sigx=%.2E_sigy=%.2E" %(params["dx"], params["dy"], params["sig_x"], params["sig_y"])
    path_ = "./" + dir_
    
    ########### Files and Folders to save results ###########
    params["data_dir"] = path_ + "/data"
    params["data_file"] = path_ + "/data/data.h5"
    params["smcmc_dir"] = path_ + "/smcmc"
    
    #create an smcmc filtering dir
    if os.path.isdir(params["smcmc_dir"]):
        answer = input("Going to remove the current directory (%s).\
         Do you want to remove the directory? [y]. To change its name enter [n]. "  %params["smcmc_dir"])
        if answer.lower() in ["y","yes"]:
            shutil.rmtree(params["smcmc_dir"]) #remove 
        elif answer.lower() in ["n","no"]:
            print("You need to change the name of the directory %s. " %params["smcmc_dir"])
            answer = input("Please enter a new name: ")
            os.rename(params["smcmc_dir"], answer)

    os.makedirs(params["smcmc_dir"], exist_ok=True)

    params["x_star"] = get_data(params, 0, "signal")

    simu = range(0, params["nsimu"])

    starttime = time.time()

    #Parallel(n_jobs=params["nsimu"])(delayed(smcmc_filter)(params, h) for h in simu)
    run_smcmc = partial(smcmc_filter, params)
    print("Performing %d simulations on %d processors" %(params["nsimu"], params["ncores"]))
    pool = mp.Pool(processes = params["ncores"])
    pool.map(run_smcmc, simu)
    
    endtime = time.time() - starttime
    print("Finished MCMC Filtering in ", endtime, "seconds")

if __name__ == '__main__':
    main()