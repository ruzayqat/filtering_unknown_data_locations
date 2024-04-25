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


    return params



def main():
    """Kalman Filter"""
    print("Parsing input file...")
    input_file = "example_input.yml"
    params = get_params(input_file)

    dir_ = "example_dx=%d_dy=%d_sigx=%.2E_sigy=%.2E" %(params["dx"], params["dy"], params["sig_x"], params["sig_y"])
    path_ = "./" + dir_
    
    if os.path.isdir(path_):
        answer = input("Going to remove the current directory (%s).\
         Do you want to remove the directory? [y]. To change its name enter [n]. "  %path_)
        if answer.lower() in ["y","yes"]:
            shutil.rmtree(dir_) #remove 
        elif answer.lower() in ["n","no"]:
            print("You need to change the name of the directory %s. " %dir_)
            answer = input("Please enter a new name: ")
            os.rename(dir_, answer)

    
    ########### Files and Folders to save results ###########
    params["data_dir"] = path_ + "/data"
    params["data_file"] = path_ + "/data/data.h5"
    params["kalman_filter_dir"] = path_ + "/kf"


    #create a data dir where observations are saved
    os.makedirs(params["data_dir"], exist_ok=True)
    #create a Kalman filtering dir
    os.makedirs(params["kalman_filter_dir"], exist_ok=True)

    #generate initial conditions
    initial_condition(params)
    #generate the obserrvations
    generate_data(params)


    print("Now running KF....")
    starttime = time.time()
    kalman_filter(params)
    endtime = time.time() - starttime
    print("Finished KF in ", endtime, "seconds")


if __name__ == '__main__':
    main()