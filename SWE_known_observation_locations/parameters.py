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
    params['dim2'] = params['dgx'] * params['dgy']
    params['dimx'] = 3 * params['dim2']

    params["obs_per_floater"] = 2
    params['dimo'] = params["obs_per_floater"] * params['N_f']
    params['dimo'] = int(params['dimo'])


    # filename = "data/bathy_region.h5"
    # with h5py.File(filename, "r") as fin:
    #     H = fin["bathy"][...] #abs. dist from sea floor topography to geiod 

    # ssh = np.zeros((24* params['simu_days'],params['dgx'], params['dgy']))
    # #read ssh from ssh file. 
    # filename = "data/ssh_u_v_region.h5"
    # with h5py.File(filename, "r") as fin:
    #     for it in range(24*params['simu_days']):
    #         step = "time_%05d" %it
    #         ssh[it,:,:] = fin[step]["ssh"][...]
   
    # uin = np.zeros((24*params['simu_days'], params['dgx'], params['dgy']))
    # vin = np.zeros((24*params['simu_days'], params['dgx'], params['dgy']))
    # #read height from height file or from pressure file. 
    # filename = "data/ssh_u_v_region.h5"
    # with h5py.File(filename, "r") as fin:
    #     for it in range(24*params['simu_days']):
    #         step = "time_%05d" %it
    #         uin[it,:,:] = fin[step]["u"][...]
    #         vin[it,:,:] = fin[step]["v"][...]
    
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
    os.makedirs(params["data_dir"], exist_ok=True)

    #create an mcmc filtering dir
    os.makedirs(params["mcmc_filter_dir"], exist_ok=True)

    #create mcmc filtering restart dir
    os.makedirs(params["mcmc_restart_dir"], exist_ok=True)

    #create floaters_info_dir
    os.makedirs(params["floaters_info_dir"], exist_ok=True)


    return params
