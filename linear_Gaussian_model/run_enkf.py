"""Main module"""
from functools import partial
import numpy as np
import multiprocessing as mp
from kf_and_ensembles import enkf
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



    #create an enkf dir
    os.makedirs(params["enkf_dir"], exist_ok=True)



    return params



def main():
    """Mult-simulations of MCMC - filtering"""
    dir_ = "example_enkf"
    path_ = "./example_enkf"
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

    print("Now running EnKF....")
    starttime = time.time()
    enkf(params)
    endtime = time.time() - starttime
    print("Finished EnKF in ", endtime, "seconds")


if __name__ == '__main__':
    main()