"""Module for data functions"""
import os, numpy as np
from tqdm import tqdm
import h5py
from tools import mvnrnd, mat_mul
from solver1 import solve
from joblib import Parallel, delayed
import yaml
from scipy.interpolate import CubicSpline 
from scipy.interpolate import RegularGridInterpolator as RGI

def get_params(input_file):
    """Get simulation parameters"""
    print('Loading parameters from %s...' % input_file)
    with open(input_file, 'r') as (fin):
        params = yaml.safe_load(fin)
    params['dim2'] = params['dgx'] * params['dgy']
    params['dimx'] = 3 * params['dim2']

    params["obs_per_floater"] = 2
    params['dimo'] = params["obs_per_floater"] * params['N_f']
    params['dimo'] = int(params['dimo'])

    
    file = "data/X_int0_%dx%d.npy" %(params["dgx"], params["dgy"])
    X_int0 = np.load(file)

    file = "data/H_%dx%d.npy" %(params["dgx"], params["dgy"])
    params["H"] = np.load(file)
    params["H"] = np.abs(np.amin(X_int0[:,0,:,:])) * np.ones_like(params["H"])
    X_int0[:,0,:,:] += params["H"]



    h0 = X_int0[0,0,:,:]
    u0 = X_int0[0,1,:,:]
    v0 = X_int0[0,2,:,:]
    x_star = np.concatenate((h0.flatten(),u0.flatten(),v0.flatten()))
    params["x_star"] = x_star

    X_int0 = np.transpose(X_int0, (0,1,3,2)) #shape is now 3 x Ngy x Ngx

    t_ = np.arange(0, 24*params['simu_days'])
    X_int = CubicSpline(t_, X_int0[0:24*params['simu_days']], axis=0) 

    X_int_array = np.zeros((params["T"], 3, params["dgy"], params["dgx"]))
    time = 0.
    for n in range(params["T"]):
        t = time/3600.
        X_int_array[n] = X_int(t)
        time += params["dt"]
    
    params["X_int"] = X_int_array




    m_per_deg_lat = 111132.954 - 559.822 * np.cos( 2.0 * params["lat"][0] * np.pi/180 )\
                                + 1.175 * np.cos( 4.0 * params["lat"][0]* np.pi/180)\
                                - 0.0023 * np.cos( 6.0 * params["lat"][0]* np.pi/180)
    m_per_deg_lat += 111132.954 - 559.822 * np.cos( 2.0 * params["lat"][1] * np.pi/180 )\
                                + 1.175 * np.cos( 4.0 * params["lat"][1]* np.pi/180)\
                                - 0.0023 * np.cos( 6.0 * params["lat"][1]* np.pi/180)
    m_per_deg_lat /= 2
    m_per_deg_lon = 111412.84 * np.cos (params["lat"][0]* np.pi/180) \
                    - 93.5 * np.cos (3.0 * params["lat"][0]* np.pi/180)\
                    + 0.118 * np.cos (5.0 * params["lat"][0]* np.pi/180)
    m_per_deg_lon += 111412.84 * np.cos (params["lat"][1]* np.pi/180) \
                    - 93.5 * np.cos (3.0 * params["lat"][1]* np.pi/180)\
                    + 0.118 * np.cos (5.0 * params["lat"][1]* np.pi/180)  

    m_per_deg_lon /= 2

    m_per_deg_lon += 350
    m_per_deg_lat += 350
    params["dx"] = np.ceil(m_per_deg_lon * (params["long"][1]-params["long"][0])/(params["dgx"]-1))
    params["dy"] = np.ceil(m_per_deg_lat * (params["lat"][1]-params["lat"][0])/(params["dgy"]-1))

    print("dx = ", params["dx"], "dy = ", params["dy"])

    params["m_per_deg_lon"] = m_per_deg_lon
    params["m_per_deg_lat"] = m_per_deg_lat



    x = np.arange(0, params['dgx']) * params["dx"] # Zonal distance coordinate (m)
    y = np.arange(0, params['dgy']) * params["dy"] # Meridional distance coordinate (m)
    x_mesh, y_mesh = np.meshgrid(x, y)

    F = params["f0"] + params["beta"] * (y_mesh - np.mean(y))

    params["F"] = F

    J = np.arange(0,params["noise_modes_num"])
    params["A"] = np.sin(2 * np.pi/x[-1] * x[:,None] * J) #of shape (dgx, noise_modes_num )
    #self.A[:,0] = np.ones(self.A.shape[0]) #first mode is random variable *  1
    params["B"] = np.sin(2 * np.pi/y[-1] * y[:,None] * J) #of shape (dgy, noise_modes_num )
    #self.B[:,0] = np.ones(self.B.shape[0])  #first mode is random variable *  1
    params["var_sqrt"] =  np.zeros((params["noise_modes_num"],params["noise_modes_num"]))
    for i in range(params["noise_modes_num"]):
        for j in range(params["noise_modes_num"]):
            params["var_sqrt"][i,j] = params['sig_x']/np.sqrt(np.maximum(i,j)+1)

    #create a data dir
    os.makedirs(params["prior_dir"], exist_ok=True)

    return params

def get_drifters_data(params):
    hyp = np.sqrt(params["dx"]**2 + params["dy"]**2)
    x_ind = np.arange(0, params["dgx"])
    y_ind = np.arange(0, params["dgy"])
    x_mesh = x_ind * params["dx"]
    y_mesh = y_ind * params["dy"]
    height = params["x_star"][0:params["dim2"]].reshape(params["dgx"],params["dgy"])
    floater_xy = np.zeros((24*params["simu_days"], 2 * params["N_f"]))
    floater_uv = np.zeros((24*params["simu_days"], 2 * params["N_f"]))
    floater_uv_error = np.zeros((24*params["simu_days"], 2 * params["N_f"]))
    if params["floaters_init_data_provided"]:
        floater_xy_uv = np.zeros((params["N_f"], 8))
        #read them
        for floater in range(params["N_f"]):
            for i,key in enumerate(["x","y","ve","vn","ve_error","vn_error"]):
                filename = "data/drifters/floater_%s%05d.npy" %(key,floater)
                if i == 0:
                    floater_xy[:,2*floater] = np.load(filename) #lon
                elif i == 1:
                    floater_xy[:,2*floater+1] = np.load(filename) #lat
                elif i == 2:
                    floater_uv[:,2*floater] = np.load(filename) #u
                elif i == 3:
                    floater_uv[:,2*floater+1] = np.load(filename) #v
                elif i == 4:
                    floater_uv_error[:,2*floater] = np.load(filename) #u-error
                elif i == 5:
                    floater_uv_error[:,2*floater+1] = np.load(filename) #v-error

        
                        
            floater_xy[:,2*floater] = params["m_per_deg_lon"] * np.abs(floater_xy[:,2*floater] - params["long"][0])
            floater_xy[:,2*floater+1] = params["m_per_deg_lat"] * np.abs(floater_xy[:,2*floater+1] - params["lat"][0])

        print(np.mean(floater_uv_error))
        print(np.amin(floater_uv_error), np.amax(floater_uv_error))

    else:
        raise Exception("Should provide the initial positions of the drifters!")


    return floater_xy[0], floater_uv, floater_uv_error


def generate_prior_mean(params, isim):
    """doctsring"""
    np.random.seed(isim)
    signal = np.zeros((params["dimx"],params["T"]+1))
    params["isim"] = isim
    signal[:,0] = params["x_star"]
    params["time"] = 0.0

    height = signal[0:params["dim2"], 0].reshape(params["dgx"],params["dgy"])
    u = signal[params["dim2"]:2*params["dim2"],0].reshape(params["dgx"],params["dgy"])
    v = signal[2*params["dim2"]:3*params["dim2"],0].reshape(params["dgx"],params["dgy"])
    hyp = np.sqrt(params["dx"]**2 + params["dy"]**2)
    x_mesh = np.arange(0, params['dgx']) * params["dx"] # Zonal distance coordinate (m)
    y_mesh = np.arange(0, params['dgy']) * params["dy"] # Meridional distance coordinate (m)
    n_o = int(np.floor(params["T"]/params["t_freq"])) 
    obs_xy = np.zeros(2 * params["N_f"])
    obs_indices = np.zeros(params["N_f"], dtype = int)
    floater_xy = np.zeros((params["T"]+1, 2 * params["N_f"]))
    surr_pts_ind = np.zeros((2, 4 * params["N_f"]), dtype = int)
    surr_pts_xy = np.zeros((2, 4 * params["N_f"]))
    x_ind = np.arange(params["dgx"])
    y_ind = np.arange(params["dgy"])
    w = np.zeros((params["N_f"], 4))

    if params["floaters_init_data_provided"]:
        floater_xy[0], floater_uv, floater_uv_error = get_drifters_data(params)
    
    print('Forwarding the dynamics....')
    for step in tqdm(range(params['T'])):
        tstep, signal[:,step+1] = solve(1, signal[:,step], params)
        params["time"] += tstep
        

        #Update floaters positions
        HEIGHT = signal[0:params["dim2"],step+1].reshape(params["dgx"],params["dgy"])
        UVELOCITY = signal[params["dim2"]:2*params["dim2"],step+1].reshape(params["dgx"],params["dgy"])
        VVELOCITY = signal[2*params["dim2"]:3*params["dim2"],step+1].reshape(params["dgx"],params["dgy"])
        #floaters height
        # h_float = np.zeros(params["N_f"])
        for i in range(params["N_f"]):
            c1 = (x_mesh >= floater_xy[step, 2*i])
            right = x_ind[c1]
            c2 = (x_mesh <= floater_xy[step, 2*i])
            left  = x_ind[c2]
            c3 = (y_mesh >= floater_xy[step, 2*i+1])
            above = y_ind[c3]
            c4 = (y_mesh <= floater_xy[step, 2*i+1])
            below = y_ind[c4]

            surr_pts_ind[0, 4*i : 4*(i+1)]  = np.array([right[0],right[0], left[-1], left[-1]])
            surr_pts_ind[1, 4*i : 4*(i+1)]  = np.array([above[0],below[-1],below[-1],above[0]])
            surr_pts_xy[0, 4*i : 4*(i+1)] = x_mesh[surr_pts_ind[0, 4*i : 4*(i+1)]]
            surr_pts_xy[1, 4*i : 4*(i+1)] = y_mesh[surr_pts_ind[1, 4*i : 4*(i+1)]]

            for j in range(4):
                dist_from_float  = np.sqrt((surr_pts_xy[0,4*i+j] - floater_xy[step, 2*i])**2\
                                        + (surr_pts_xy[1,4*i+j] - floater_xy[step, 2*i+1])**2)
                w[i,j] = np.maximum(0.0, hyp - dist_from_float)
            w[i,:] = w[i,:] / np.sum(w[i,:])

            u_float = np.zeros(params["N_f"])
            v_float = np.zeros(params["N_f"])
            ### IMPORTANT NOTE: 1d Array [a0,a1,a2,a3] in Python when reshaped to 2d array, it is 
            ### [[a0, a1]
            ###  [a2, a3]] Reshaped row-wise. Different from Matlab!
            for j in range(4):
                u_float[i] += UVELOCITY[surr_pts_ind[0, 4*i+j], surr_pts_ind[1, 4*i+j]] * w[i,j]
                v_float[i] += VVELOCITY[surr_pts_ind[0, 4*i+j], surr_pts_ind[1, 4*i+j]] * w[i,j]

            floater_xy[step+1, 2*i : 2*(i+1)] = floater_xy[step, 2*i : 2*(i+1)]\
                                 + tstep * np.array([u_float[i], v_float[i]])
            #print("u_float, v_float = ", u_float[i], v_float[i])
            ### check if u_float and v_float pushed the floater outside the simulation region
            
            if floater_xy[step+1, 2*i+1] < y_mesh[0]:
                floater_xy[step+1, 2*i+1] = y_mesh[0] + (y_mesh[0] - floater_xy[step+1, 2*i+1])
            if floater_xy[step+1, 2*i+1] > y_mesh[-1]:
                floater_xy[step+1, 2*i+1] = y_mesh[-1] - (floater_xy[step+1, 2*i+1] - y_mesh[-1])
        

        if (step + 1) % params["t_freq"] == 0:
            
            noise_x = compute_noise(params,1)
            print("maximum of noise_x =", np.amax(noise_x))
            signal[:,step+1] +=  noise_x
        else:
            if params["add_noise_sig_every_dt"]:
            ##add noise to the signal at every time step
                noise_x = compute_noise(params,1)
                signal[:,step+1] += noise_x

    dump_data(params, signal, "prior")
    dump_floaters(params, floater_xy, "floater_xy")


def compute_noise(params, N_samples, const=1):
    if N_samples == 1:
        noise = np.zeros(params["dimx"])
    else:
        noise = np.zeros((params["dimx"], N_samples))

    for i in range(3):
        if N_samples == 1:
            # a =  params["var_sqrt"][:,None] * np.random.normal(size=(params['noise_modes_num'], 1)) #sample ~ N(0, sig_x^2/j^2)
            # b =  params["var_sqrt"] * np.random.normal(size=(1, params['noise_modes_num']))   #sample ~ N(0, sig_x^2/j^2)
            # rand = (params["B"] @ a) @ (b @ params["A"].T)
            # noise[i*params["dim2"]:(i+1)*params["dim2"]] = rand.reshape(-1)

            epsil = const * params["var_sqrt"] * np.random.normal(size=(params['noise_modes_num'], params['noise_modes_num']))
            Xi = params["B"] @ epsil @ params["A"].T
            noise[i*params["dim2"]:(i+1)*params["dim2"]] = Xi.reshape(-1)
        else:
            # a =  params["var_sqrt"][:,None] * np.random.normal(size=(N_samples,params['noise_modes_num'], 1)) #sample ~ N(0, sig_x^2/j^2)
            # b = params["var_sqrt"] * np.random.normal(size=(N_samples, 1, params['noise_modes_num'])) #sample ~ N(0, sig_x^2/j^2)
            # c = a@b #of shape (N_samples, noise_modes_num, noise_modes_num)
            # rand = params["B"] @ c @ params["A"].T #of shape (N, dgx, dgy)
            # rand = rand.reshape(rand.shape[0], rand.shape[1]*rand.shape[2]) #of shape (N, dgx*dgy)
            # noise[i*params["dim2"]:(i+1)*params["dim2"],:] =  rand.T #of shape (dim2 , N)
            epsil = const * params["var_sqrt"] * np.random.normal(size=(N_samples, params['noise_modes_num'], params['noise_modes_num']))
            #epsil of shape (N, J, J)
            Xi = params["B"] @ epsil @ params["A"].T #of shape (N, dgy, dgx)
            Xi = Xi.reshape(Xi.shape[0], Xi.shape[1]*Xi.shape[2]) #of shape (N, dgy*dgx)
            noise[i*params["dim2"]:(i+1)*params["dim2"],:] =  Xi.T #of shape (dgy*dgx , N)

    #print("max_noise= ", np.amax(noise))
    return noise #of shape (3*dgx*dgy, N)


def dump_data(params, array, name):
    """Dumping array step"""
    filename = "%s/prior_%08d.h5" 
    filename = filename %(params["prior_dir"],params["isim"])

    with h5py.File(filename, "w") as fout:
        fout.create_dataset(name=name, data=array)

def dump_floaters(params, array, name):
    """Dumping array step"""
    filename = "%s/floaters_%08d.h5" 
    filename = filename %(params["prior_dir"],params["isim"])

    with h5py.File(filename, "w") as fout:
        fout.create_dataset(name=name, data=array)


def get_floater_info(params, step, name):
    """get_floater_info"""
    with h5py.File(params["floater_info_file"], "r") as fin:
        data = fin["step_%08d"%step][name][...]
    return data


####################################
input_file = "example_input.yml"
params = get_params(input_file)
nsimuls = 50
Parallel(n_jobs=50)(delayed(generate_prior_mean)(params, isim) for isim in range(nsimuls))
