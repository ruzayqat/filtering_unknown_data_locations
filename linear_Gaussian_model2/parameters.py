"""Parameter module"""
import numpy as np, yaml, os
import h5py
from scipy.interpolate import CubicSpline 
from scipy.interpolate import RegularGridInterpolator as RGI

def get_params(input_file):
    """Get simulation parameters"""
    print('Loading parameters from %s...' % input_file)
    with open(input_file, 'r') as (fin):
        params = yaml.safe_load(fin)
    

    
    #create a data dir
    os.makedirs(params["data_dir"], exist_ok=True)

    #create an mcmc filtering dir
    os.makedirs(params["mcmc_filter_dir"], exist_ok=True)

    #create an Kalman filtering dir
    os.makedirs(params["kalman_dir"], exist_ok=True)

    #create an enkf dir
    os.makedirs(params["enkf_dir"], exist_ok=True)

    #create an etkf dir
    os.makedirs(params["etkf_dir"], exist_ok=True)

    #create an etkf_sqrt dir
    os.makedirs(params["etkf_sqrt_dir"], exist_ok=True)

    return params
