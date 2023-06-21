"""Main module"""
from functools import partial
import numpy as np
import multiprocessing as mp
from kf_and_ensembles import kalman_filter
from data_tools import (initial_condition, generate_data)
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

    
    #create a data dir
    os.makedirs(params["data_dir"], exist_ok=True)

    #create an Kalman filtering dir
    os.makedirs(params["kalman_filter_dir"], exist_ok=True)


    return params



def main():
    """Mult-simulations of MCMC - filtering"""
    dir_ = "example_KF"
    path_ = "./example_KF"
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
    initial_condition(params)
    generate_data(params)

    simu = range(0, params["nsimu"])

    print("Now running KF....")
    starttime = time.time()
    kalman_filter(params)
    endtime = time.time() - starttime
    print("Finished KF in ", endtime, "seconds")


if __name__ == '__main__':
    main()