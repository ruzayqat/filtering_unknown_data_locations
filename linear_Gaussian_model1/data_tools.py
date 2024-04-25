"""Module for data functions"""
import os, numpy as np
from tqdm import tqdm
import h5py
from tools import mvnrnd, mat_mul



def initial_condition(params):
    """initialcondition"""
    x_star = -0.45* np.random.rand(params["dx"])
    params["x_star"] = x_star


def generate_data(params):
    """doctsring"""
    signal = params["x_star"]
    dump_data(params, signal, 0, "signal")
    
    print('Generating data....')
    for step in tqdm(range(params['T'])):
        if np.isscalar(params["A"]): 
            signal = params["A"] * signal + params["sig_x"] *  np.random.normal(size=(params['dx']))
        else:
            signal = params["A"] @ signal + params["sig_x"] *  np.random.normal(size=(params['dx']))
        dump_data(params, signal, step+1, "signal")
        if (step + 1) % params["t_freq"] == 0:
            data =  signal[params["s_freq"]-1::params["s_freq"]] + params['sig_y'] * np.random.normal(size=(params['dy']))
            dump_data(params, data, step, 'data')
        
    return (signal, data)



def dump_data(params, array, step, name):
    """Dumping array step"""
    #name options: data, signal, time
    #print('Dumping to %s\n' %params["data_file"])
    dir_ = os.path.dirname(params["data_file"])
    os.makedirs(dir_, exist_ok=True)
    grp_name = "step_%08d" %step
    with h5py.File(params["data_file"], "a") as fout:
        if grp_name in fout:
            grp = fout[grp_name]
        else:
            grp = fout.create_group(name=grp_name)
        grp.create_dataset(name=name, data=array)



def get_data(params, step, name):
    """get_data"""
    with h5py.File(params["data_file"], "r") as fin:
        data = fin["step_%08d"%step][name][...]
    return data



