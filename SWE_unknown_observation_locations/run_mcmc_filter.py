"""Main module"""
from functools import partial
import multiprocessing as mp
from parameters import get_params
#from mcmc_lag_unknown_Locs_Nsamples import mcmc_filter
from mcmc_lag_unknown_Locs_1sample_No_Averaging import mcmc_filter
import os
import shutil
import os.path

def main():
    """Mult-simulations of MCMC - filtering"""
    dir_ = "example"
    path_ = "./example"
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

    simu = range(0, params["nsimu"])

    if params["run_simuls_in_parallel"]:
        params["run_solver_in_parallel"] = False #has to be false here
        run = partial(mcmc_filter, params)
        print("Performing %d simulations on %d processors" %(params["nsimu"], params["ncores"]))
        pool = mp.Pool(processes = params["ncores"])
        pool.map(run, simu)
    else:
        params["run_solver_in_parallel"] = True # True or False
        for n in simu:
            print("Performing simulation %d on 1 processor" %n)
            mcmc_filter(params, n)


if __name__ == '__main__':
    main()