"""Module for data functions"""
import os, numpy as np
from tqdm import tqdm
import h5py
from tools import mvnrnd, mat_mul
from solver1 import solve


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


def generate_data(params):
    """doctsring"""
    signal = params["x_star"]
    params["time"] = 0.0
    dump_data(params, signal, 0, "signal")
    dump_data(params, 0.0, 0, "time")
    height = signal[0:params["dim2"]].reshape(params["dgx"],params["dgy"])
    u = signal[params["dim2"]:2*params["dim2"]].reshape(params["dgx"],params["dgy"])
    v = signal[2*params["dim2"]:3*params["dim2"]].reshape(params["dgx"],params["dgy"])
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
    else:
        #initialzie locations of floaters
        tmp_cond = False
        a = np.zeros(params["N_f"], dtype = int)
        b = np.zeros(params["N_f"], dtype = int)
        cond = np.random.uniform(size = (params["N_f"], params["N_f"]))
        cond = cond > 0
        med_h = np.median(height)
        max_h = np.amax(height)
        threshold = med_h
        ind_sa = np.where(height >= threshold)
        aa1 = np.unique(ind_sa[0])
        aa2 = np.unique(ind_sa[1])

        if x_ind[0] in aa1:
            indices = np.where(aa1==x_ind[0])
            aa1 = np.delete(aa1, indices)
        if x_ind[-1] in aa1:
            indices = np.where(aa1==x_ind[-1])
            aa1 = np.delete(aa1, indices)
        if y_ind[0] in aa2:
            indices = np.where(aa2==y_ind[0])
            aa2 = np.delete(aa2, indices)
        if y_ind[-1] in aa2:
            indices = np.where(aa2==y_ind[-1])
            aa2 = np.delete(aa2, indices)


        while not tmp_cond:
            #Try to put first N0 floaters in the region of high action
            for i in range(params["N_f0"]):
                a[i] = np.random.choice(aa1)
                b[i] = np.random.choice(aa2)
            
            for j in range(params["N_f0"]):
                for k in range(params["N_f0"]):
                    if k != j:
                        cond[j,k] = (np.abs(a[j] - a[k]) >= params["f_sep"])\
                        or (np.abs(b[j] - b[k]) >= params["f_sep"])
            tmp_cond = cond.all()

        for i in range(params["N_f0"]):
            # floater_height[0,i] = height[a[i],b[i]] #+ H[a[i], b[i]] 
            floater_xy[0, 2*i : 2*(i+1)] = np.array([x_mesh[a[i]],y_mesh[b[i]]])

        #put the rest of the floaters N_f0+1 : N_f at some random locations so that they are at least "f_sep" 
        #cells far from each others
        far = False
        ind_sa = np.where(height < threshold) #change this according to the above number in h
        aa1 = np.unique(ind_sa[0])
        aa2 = np.unique(ind_sa[1])
        while not far:
            for i in range(params["N_f0"],params["N_f"]):
                a[i] = np.random.choice(aa1)
                b[i] = np.random.choice(aa2)
            for j in range(params["N_f"]):
                for k in range(params["N_f"]):
                    if k != j:
                        cond[j,k] = (np.abs(a[j] - a[k]) >= params["f_sep"])\
                        or (np.abs(b[j] - b[k]) >= params["f_sep"])

            far = cond.all()
     
        for i in range(params["N_f0"],params["N_f"]):
            floater_xy[0, 2*i : 2*(i+1)] = np.array([x_mesh[a[i]],y_mesh[b[i]]])
            # floater_height[0,i] = height[a[i],b[i]] 

    dump_floater_info(params, floater_xy[0,:], 0, 'floaters_xy')
    
    print('Generating data....')
    for step in tqdm(range(params['T'])):
        tstep, signal = solve(1, signal, params)
        dump_data(params, tstep, step+1, "time")
        params["time"] += tstep
        #Update floaters positions
        HEIGHT = signal[0:params["dim2"]].reshape(params["dgx"],params["dgy"])
        UVELOCITY = signal[params["dim2"]:2*params["dim2"]].reshape(params["dgx"],params["dgy"])
        VVELOCITY = signal[2*params["dim2"]:3*params["dim2"]].reshape(params["dgx"],params["dgy"])
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
        
            #compute floaters height
            c1 = (x_mesh >= floater_xy[step+1, 2*i])
            right = x_ind[c1]
            c2 = (x_mesh <= floater_xy[step+1, 2*i])
            left  = x_ind[c2]
            c3 = (y_mesh >= floater_xy[step+1, 2*i+1])
            above = y_ind[c3]
            c4 = (y_mesh <= floater_xy[step+1, 2*i+1])
            below = y_ind[c4]

            # print("x=",floater_xy[step+1,2*i], "x_mesh[end] = ", x_mesh[-1])
            # print(right[0])
            # print(left[-1]) 
 
            surr_pts_ind[0, 4*i : 4*(i+1)]  = np.array([right[0],right[0], left[-1], left[-1]])
            surr_pts_ind[1, 4*i : 4*(i+1)]  = np.array([above[0],below[-1],below[-1],above[0]])
            surr_pts_xy[0, 4*i : 4*(i+1)] = x_mesh[surr_pts_ind[0, 4*i : 4*(i+1)]]
            surr_pts_xy[1, 4*i : 4*(i+1)] = y_mesh[surr_pts_ind[1, 4*i : 4*(i+1)]]

            for j in range(4):
                dist_from_float = np.sqrt((surr_pts_xy[0,4*i+j] - floater_xy[step+1, 2*i])**2\
                                        + (surr_pts_xy[1,4*i+j] - floater_xy[step+1, 2*i+1])**2)
                w[i,j] = np.maximum(0.0, hyp - dist_from_float)
            w[i,:] = w[i,:] / np.sum(w[i,:])
            

            # for j in range(4):
            #     ### IMPORTANT NOTE: 1d Array [a0,a1,a2,a3] in Python when reshaped to 2d array, it is 
            #     ### [[a0, a1]
            #     #    [a2, a3]] Reshaped row-wise. Different from Matlab!
            #     h_float[i] += HEIGHT[surr_pts_ind[0, 4*i+j], surr_pts_ind[1, 4*i+j]] *  w[i,j]
            #                   # (HEIGHT[surr_pts_ind[0, 4*i+j], surr_pts_ind[1, 4*i+j]]\
            #                   #  + H[surr_pts_ind[0, 4*i+j], surr_pts_ind[1, 4*i+j]] ) * w[i,j]

            # floater_height[step+1, i] = h_float[i]

        # dump_floater_info(params, floater_height[step+1, :], step+1, 'floaters_height')
        dump_floater_info(params, floater_xy[step+1,:], step+1, 'floaters_xy')

        obs_indices = np.zeros(params["N_f"],dtype=int)

        
        if (step + 1) % params["t_freq"] == 0:
            
            noise_x = compute_noise(params,1)
            print("maximum of noise_x =", np.amax(noise_x))
            signal = signal + noise_x
            for i in range(params["N_f"]):
                if params['obs_loc_choice'] == 0:
                    surr_pt_ind = np.argmax(w[i,:])
                else:
                    surr_pt_ind = np.random.choice(4, 1, p=w[i,:])
                obs_indices[i] = surr_pts_ind[1, 4*i + surr_pt_ind] +\
                                                 surr_pts_ind[0, 4*i + surr_pt_ind] * params["dgy"]
                obs_xy[2*i : 2*(i+1)] = [surr_pts_xy[0, 4*i + surr_pt_ind],
                                                      surr_pts_xy[1, 4*i + surr_pt_ind]]
            
            if params["obs_h"] and params["obs_uv"]:
                obs_indices = np.concatenate((obs_indices, obs_indices + params["dim2"], 
                                obs_indices + 2*params["dim2"]))
                data = signal[obs_indices]
            elif params["obs_h"] and not params["obs_uv"]: 
                data = signal[obs_indices]
            elif not params["obs_h"] and params["obs_uv"]:
                obs_indices = np.concatenate((obs_indices + params["dim2"], 
                                obs_indices + 2*params["dim2"])) 
                data = signal[obs_indices]

            noise_o = np.random.normal(size=(params['dimo']))
            data =  data + params['sig_y'] * noise_o


            dump_data(params, data, step, 'data')
            dump_floater_info(params, obs_indices, step, 'obs_indices')
            dump_floater_info(params, obs_xy, step, 'obs_xy')
        else:
            if params["add_noise_sig_every_dt"]:
            ##add noise to the signal at every time step
                noise_x = compute_noise(params,1)
                signal += noise_x

        dump_data(params, signal, step+1, "signal")
    return (signal, data)



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

def dump_floater_info(params, array, step, name):
    """Dumping array step"""
    # name options: floaters_height, floaters_xy, obs_indices, obs_xy
    dir_ = os.path.dirname(params["floater_info_file"])
    os.makedirs(dir_, exist_ok=True)
    grp_name = "step_%08d" %step
    with h5py.File(params["floater_info_file"], "a") as fout:
        if grp_name in fout:
            grp = fout[grp_name]
        else:
            grp = fout.create_group(name=grp_name)
        grp.create_dataset(name=name, data=array)

def get_predictor_stats(params, step):
    """get_predictor_stats"""
    with h5py.File(params["predictor_file"], "r") as fin:
        pred_mean = fin["step%08d"%step]["mean"][...]
        pred_inv_cov = fin["step%08d"%step]["inv_cov_array"][...]
        logdet_cov = fin["step%08d"%step]["logdet_cov"][...]
    return pred_mean, pred_inv_cov, logdet_cov

def get_data(params, step, name):
    """get_data"""
    with h5py.File(params["data_file"], "r") as fin:
        data = fin["step_%08d"%step][name][...]
    return data

def get_floater_info(params, step, name):
    """get_floater_info"""
    with h5py.File(params["floater_info_file"], "r") as fin:
        data = fin["step_%08d"%step][name][...]
    return data


