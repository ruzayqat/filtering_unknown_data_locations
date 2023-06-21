"""Main module"""
from functools import partial
import numpy as np
import multiprocessing as mp
from mcmc_filtering import mcmc_filter
from data_tools import get_data
import yaml
import os
import shutil
import os.path
import time


def get_params(input_file):
    """Get simulation parameters"""
    print('Loading parameters from %s...' % input_file)
    with open(input_file, 'r') as (fin):
        params = yaml.safe_load(fin)
        params["dy"] = int(np.floor(params["dx"]/params["s_freq"]))


    #create an mcmc filtering dir
    os.makedirs(params["mcmc_dir"], exist_ok=True)


    return params



def main():
    """Mult-simulations of MCMC - filtering"""
    dir_ = "example_mcmc"
    path_ = "./example_mcmc"
    if os.path.isdir(path_):
        answer = input("Going to remove the current directory (%s).\
         Do you want to remove the directory? [y]. To change its name enter [n]. "  %path_)
        if answer.lower() in ["y","yes"]:
            shutil.rmtree(dir_) #remove 
        elif answer.lower() in ["n","no"]:
            print("You need to change the name of the directory %s. " %dir_)
            answer = input("Please enter a new name: ")
            os.rename(dir_, answer)

    print("Parsing input file...")
    input_file = "example_input.yml"
    params = get_params(input_file)
    params["x_star"] = get_data(params, 0, "signal")

    simu = range(0, params["nsimu"])

    starttime = time.time()
    run_mcmc = partial(mcmc_filter, params)
    print("Performing %d simulations on %d processors" %(params["nsimu"], params["ncores"]))
    pool = mp.Pool(processes = params["ncores"])
    pool.map(run_mcmc, simu)
    endtime = time.time() - starttime
    print("Finished MCMC Filtering in ", endtime, "seconds")

if __name__ == '__main__':
    main()