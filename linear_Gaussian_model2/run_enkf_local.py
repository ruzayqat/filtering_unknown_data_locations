"""Main module"""
import numpy as np
from kf_and_ensembles import enkf_local
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

    return params



def main():
    """Mult-simulations of MCMC - filtering"""
    print("Parsing input file...")
    input_file = "example_input.yml"
    params = get_params(input_file)

    dir_ = "example_dx=%d_dy=%d_sigx=%.2E_sigy=%.2E" %(params["dx"], params["dy"], params["sig_x"], params["sig_y"])
    path_ = "./" + dir_
    
    ########### Files and Folders to save results ###########
    params["data_dir"] = path_ + "/data"
    params["data_file"] = path_ + "/data/data.h5"
    params["enkf_loc_dir"] = path_ + "/enkf_loc"

    if os.path.isdir(params["enkf_loc_dir"]):
        answer = input("Going to remove the current directory (%s).\
         Do you want to remove the directory? [y]. To change its name enter [n]. "  %params["enkf_loc_dir"])
        if answer.lower() in ["y","yes"]:
            shutil.rmtree(params["enkf_loc_dir"]) #remove 
        elif answer.lower() in ["n","no"]:
            print("You need to change the name of the directory %s. " %params["enkf_loc_dir"])
            answer = input("Please enter a new name: ")
            os.rename(params["enkf_loc_dir"], answer)

    os.makedirs(params["enkf_loc_dir"], exist_ok=True)
    params["x_star"] = get_data(params, 0, "signal")

    print("Now running EnKF Local....")
    starttime = time.time()
    enkf_local(params)
    endtime = time.time() - starttime
    print("Finished EnKF Local in ", endtime, "seconds")


if __name__ == '__main__':
    main()