import numpy as np
#from pathos.multiprocessing import ProcessingPool as Pool
from joblib import Parallel, delayed



class SWE_Solver():
    """docstring for SWE_Solver"""
    def __init__(self, params):
        self.params = params
        self.dx = params["dx"]
        self.dy = params["dy"]
        self.dt0 = params["dt"]
        self.dgx = params["dgx"]
        self.dgy = params["dgy"]
        self.dim2 = params["dim2"]
        self.g = params["g"]
        self.CFL = params["c"]
        #self.BC = params["bound_cond"]
        #self.mrc = params["mrc"]
        self.F = np.transpose(params["F"])
        #self.gdHdx = np.transpose(params["gdHdx"])
        #self.gdHdy = np.transpose(params["gdHdy"])

        self.sol = np.zeros((3, self.dgy + 2, self.dgx + 2))
        self.eps = 1e-12
        #params["X_int"] is an interpolation object: An interpolation over time (in seconds) of real data
        


    def _return_sol(self, prev_sol):
        """Solving for next step"""
        self.time = self.params["time"]
        
        for i in range(3):
            self.sol[i, 1:-1, 1:-1] = np.transpose(prev_sol[i*self.dim2:(i+1)*self.dim2].reshape(self.dgx, self.dgy))

        self.X_int = self.params["X_int"][int(self.time/self.params["dt"])] #self.X_int is of shape Tx 3 x Ngx x Ngy
        self._apply_bc()
        
        self._solve()

        self.sol = np.transpose(self.sol,(0,2,1)) #make prev_sol of shape (3,dgx,dgy)
        #return h, u, v
        for i in range(1,3):
            self.sol[i,1:-1, 1:-1] /= self.sol[0,1:-1, 1:-1]
  
        return self.dt, self.sol[:,1:-1, 1:-1].flatten()


    def _apply_bc(self):
        """Implements the boundary conditions
        Args:
            sol (3D array) :  the state variables, populating a x,y grid. sol of shape 3 x (Ngy+2) x (Ngx+2)
        Returns:
            updates self.sol
        """
        # left wall (0 <= x < 1) and right wall (h)
        #self.sol[0, 1:-1, [0,-1]]  = 2 * self.X_int[0,:,[0,-1]] - self.sol[0,1:-1,[1,-2]] 
        self.sol[0, 1:-1, [0,-1]]  = self.X_int[0,:,[0,-1]]  
        #==> self.sol at the left and right walls = X_int

        # top wall (0 <= y < 1) and bottom walls (Ngy + 1 <= y < Ngy + 2) (h)
        #self.sol[0, [0,-1], 1:-1]  = 2 * self.X_int[0,[0,-1],:] - self.sol[0,[1,-2], 1:-1] 
        self.sol[0, [0,-1], 1:-1]  = self.X_int[0,[0,-1],:] 
        #==> self.sol at the left and right walls = X_int

        # left wall (0 <= x < 1) and right wall (u & v)
        self.sol[1:, 1:-1, [0,-1]]  = self.X_int[1:,:,[0,-1]]
        #==> self.sol at the left and right walls = X_int

        # top wall (0 <= y < 1) and bottom walls (Ngy + 1 <= y < Ngy + 2) (u & v)
        self.sol[1:, [0,-1], 1:-1]  = self.X_int[1:, [0,-1],:] 
        #==> self.sol at the left and right walls = X_int
        
        # corners:
            

    def _solve(self):
        """Evaluates the state variables (h, hu, hv) at a new time-step.
        It can be used in a for/while loop, iterating through each time-step.
        Args:
            U (3D array)       :  the state variables, populating a x,y grid
            dt (float)    :  time discretization step
        Returns:
            U
        """
        
        self.h_old = np.copy(self.sol[0, 1:-1, 1:-1])
        self.u_old = np.copy(self.sol[1, 1:-1, 1:-1])
        self.v_old = np.copy(self.sol[2, 1:-1, 1:-1])

        # set sol = [h, hu, hv]
        self.sol[1, :, :] *= self.sol[0, :, :]
        self.sol[2, :, :] *= self.sol[0, :, :]

        # Retrieve the mesh
        cellArea = self.dx * self.dy


        # Numerical scheme
        # _flux() returns the total flux entering and leaving each cell.
        
        
        # sol_old = self.sol
        # flux_sol, self.dt = self._flux(self.sol) #it also update self.dt
        # #self.dt = 10
        # print("dt = ", self.dt)
        # f = 1/ cellArea * flux_sol
        # f[1] += (self.v_old * self.F + self.gdHdx) * self.h_old
        # f[2] += (-self.u_old * self.F + self.gdHdy) * self.h_old
        # self.sol[:, 1: -1, 1: -1] += self.dt * f
        # self._apply_bc()
        # f_old = f
        # flux_sol, _ = self._flux(self.sol)
        # f = 1/ cellArea * flux_sol
        # h = self.sol[0, 1:-1, 1:-1]
        # u = self.sol[1, 1:-1, 1:-1]
        # v = self.sol[2, 1:-1, 1:-1]
        # f[1] += (v * self.F + self.gdHdx) * h
        # f[2] += (-u * self.F + self.gdHdy) * h
        # # f[1] += (self.v_old * self.F + self.gdHdx) * h
        # # f[2] += (-self.u_old * self.F + self.gdHdy) * h
        # self.sol[:, 1: -1, 1: -1] = sol_old[:, 1: -1, 1: -1] + self.dt/2 * (f_old + f)
        # self._apply_bc()
        

        # #2-stage Runge-Kutta:
        # # 1st stage
        # sol_pred = self.sol
        # sol_pred[0, 1: -1, 1: -1] = self.X_int[0]
        # flux_sol, self.dt = self._flux(self.sol)
        # self.dt = 60
        # print("dt = ", self.dt)

        # sol_pred[1:, 1: -1, 1: -1] += self.dt/cellArea * flux_sol[1:]

        
        # # 2nd stage
        # self.sol[1:, 1: -1, 1: -1] = \
        #         0.5 * (self.sol[1:, 1: -1, 1: -1]
        #                + sol_pred[1:, 1: -1, 1: -1]
        #                + self.dt / cellArea * self._flux(sol_pred)[0][1:] #it wont update dt again
        #                )

        # self.sol[0, 1: -1, 1: -1] = self.X_int[0]
        # self._source_forces()


        # 1st stage
        sol_pred = np.copy(self.sol)
        flux_sol, self.dt = self._flux(self.sol)
        self.dt = self.dt0
        # print("dt = ", self.dt)
        # print("time =", self.time/3600)
        sol_pred[:, 1: -1, 1: -1] += self.dt/cellArea * flux_sol

        
        # 2nd stage
        self.sol[:, 1: -1, 1: -1] = \
                0.5 * (self.sol[:, 1: -1, 1: -1]
                       + sol_pred[:, 1: -1, 1: -1]
                       + self.dt / cellArea * self._flux(sol_pred)[0] #it wont update dt again
                       )

        self._source_forces()

        
    
    
    def _flux(self, U):
        """Evaluates the total flux that enters or leaves a cell, using the \
        Lax-Friedrichs scheme.
        Args:
            U (3D array)        : the state variables 3D matrix
        Returns:
            total_flux (3D array)
        """

        total_flux = np.zeros((3, self.dgy + 2, self.dgx + 2))

        # Vertical interfaces - Horizontal flux {
        #
        # Max horizontal speed between left and right cells for every interface
        maxHorizontalSpeed, dt_x = self._max_horizontal_speed(U)

        # Lax-Friedrichs scheme
        # flux = 0.5 * (F_left + F_right) - 0.5 * maxSpeed * (U_right - U_left)
        # flux is calculated on each interface.
        horizontalFlux = self._horizontal_flux(U, maxHorizontalSpeed)

        # horizontalFlux is subtracted from the left and added to the right cells.
        total_flux[:, 1: -1, 0: -1] -= horizontalFlux
        total_flux[:, 1: -1, 1:] += horizontalFlux
        # }

        # Horizontal interfaces - Vertical flux {
        #
        # Max vertical speed between top and bottom cells for every interface.
        # (for the vertical calculations the extra horizontal cells are not needed)
        maxVerticalSpeed, dt_y = self._max_vertical_speed(U)

        # Lax-Friedrichs scheme
        # flux = 0.5 * (F_top + F_bottom) - 0.5 * maxSpeed * (U_bottom - U_top)
        verticalFlux = self._vertical_flux(U, maxVerticalSpeed)

        # verticalFlux is subtracted from the top and added to the bottom cells.
        total_flux[:, 0: -1, 1: -1] -= verticalFlux
        total_flux[:, 1: , 1: -1] += verticalFlux
        # }

        dt = min(dt_x, dt_y) * self.CFL
        # No need to keep ghost cells --> removes 2*(self.dgx + self.dgy) operations stepwise
        # Also, 1st and last nodes of each column are removed (they were only
        # needed from the numerical scheme, to calculate the other nodes).

        return total_flux[:, 1: -1, 1: -1], dt


    def _max_horizontal_speed(self, U):
        """Max horizontal speed between left and right cells for every vertical
        interface"""
        max_h_speed = np.maximum(
                        # x dim slicing of left values:  0: -1
                        np.abs(U[1, 1: -1, 0: -1] / (U[0, 1: -1, 0: -1]+ self.eps))
                        + np.sqrt(self.g * np.abs(U[0, 1: -1, 0: -1])) ,

                        # x dim slicing of right values:  1:
                        np.abs(U[1, 1: -1, 1:] / (U[0, 1: -1, 1:]+ self.eps))
                        + np.sqrt(self.g * np.abs(U[0, 1: -1, 1:]))
                        )

        dt_x = self.dx / np.amax(max_h_speed)
        return max_h_speed, dt_x


    def _max_vertical_speed(self, U):
        """Max vertical speed between top and bottom cells for every horizontal
        interface"""
        max_v_speed = np.maximum(
                        # y dim slicing of top values:  0: -1
                        np.abs(U[2, 0: -1, 1: -1]
                                / (U[0, 0: -1, 1: -1] + self.eps))
                        + np.sqrt(self.g * np.abs(U[0, 0: -1, 1: -1])) ,

                        # y dim slicing of bottom values:  1: 
                        np.abs(U[2, 1: , 1: -1]
                                / (U[0, 1: , 1: -1] + self.eps))
                        + np.sqrt(self.g * np.abs(U[0, 1: , 1: -1]))
                        )

        dt_y = self.dy / np.amax(max_v_speed)
        return max_v_speed, dt_y


    def _F(self,U):
        """Evaluates the x-dimention-fluxes-vector, F.
        Args:
            U (3D array) : the state variables 3D matrix
        """
        # h = U[0]
        # u = U[1] / h
        # v = U[2] / h

        # # 0.5 * self.g = 0.5 * 9.81 = 4.905
        # return np.array([h * u, h * u**2 + 4.905 * h**2, h * u * v])
        return np.array([U[1],
                         U[1]**2 / (U[0]+ self.eps) + 4.905 * U[0]**2,
                         U[1] * U[2] / (U[0]+ self.eps) ])


    def _G(self,U):
        """Evaluates the y-dimention-fluxes-vector, G.
        Args:
            U (3D array) : the state variables 3D matrix
        """
        # h = U[0]
        # u = U[1] / h
        # v = U[2] / h

        # # 0.5 * self.g = 0.5 * 9.81 = 4.905
        # return np.array([h * v, h * u * v, h * v**2 + 4.905 * h**2])
        return np.array([U[2],
                         U[1] * U[2] / (U[0] + self.eps),
                         U[2]**2 / (U[0]+ self.eps) + 4.905 * U[0]**2])


    def _horizontal_flux(self, U, maxHorizontalSpeed):
        """Lax-Friedrichs scheme (flux is calculated on each vertical interface)
        flux = 0.5 * (F_left + F_right) - 0.5 * maxSpeed * (U_right - U_left)
        """
        h_flux = (
            0.5 * self.dy
            * ( self._F(U[:, 1: -1, 0: -1]) + self._F(U[:, 1: -1, 1:]) )
            - 0.5 * self.dy * maxHorizontalSpeed
            * (U[:, 1: -1, 1:] - U[:, 1: -1, 0: -1])
            )
        return h_flux


    def _vertical_flux(self, U, maxVerticalSpeed):
        """Lax-Friedrichs scheme (flux is calculated on each horizontal interface)
        flux = 0.5 * (F_top + F_bottom) - 0.5 * maxSpeed * (U_bottom - U_top)
        """ 
        v_flux = (
            0.5 * self.dx
            * ( self._G(U[:, 0:-1 , 1: -1]) + self._G(U[:, 1: , 1: -1]) )
            - 0.5 * self.dx * maxVerticalSpeed
            * (U[:, 1: , 1:-1] - U[:, 0: -1, 1:-1])
            )
        return v_flux
        


    def _source_forces(self):
        """
        """ 
        h = self.sol[0,1:-1,1:-1]
        u = self.sol[1,1:-1,1:-1]/h
        v = self.sol[2,1:-1,1:-1]/h
        ##add the terms representing varying bathymetry and Coriolis force
        #h_mean_dt = self.dt * (0.5*(self.h_old + h))
        h_mean_dt_x = self.dt * (0.5*(self.sol[0,1:-1, 1:-1] + self.sol[0, 2:, 1:-1]) + self.h_old) * 0.5
        h_mean_dt_y = self.dt * (0.5*(self.sol[0,1:-1, 1:-1] + self.sol[0, 1:-1:, 2:]) + self.h_old) * 0.5
        # sqrtu2pv2 = self.dt * np.sqrt((0.5*(self.u_old + u))**2 +(0.5* (self.v_old + v))**2)
        #h_new_dt =  self.dt * self.sol[0,1:-1,1:-1]
        
        # self.sol[1,1:-1,1:-1] += (self.v_old * self.F + self.gdHdx) *  h_mean_dt \
        #                         - self.dt * self.mrc * self.u_old #bottom friction force
        # self.sol[2,1:-1,1:-1] += (-self.u_old * self.F  + self.gdHdy) *  h_mean_dt \
        #                         - self.dt * self.mrc * self.v_old

        self.sol[1,1:-1,1:-1] += (self.F*0.5*(self.v_old + v) ) *  h_mean_dt_x 
                                #- (self.g / self.mrc**2) * 0.5*(self.u_old + u) * sqrtu2pv2 #bottom friction force
        self.sol[2,1:-1,1:-1] += (-self.F* 0.5*(self.u_old + u) ) *  h_mean_dt_y 
                                #- (self.g/ self.mrc**2) * 0.5*(self.v_old + v) * sqrtu2pv2

        # self.sol[1,1:-1,1:-1] +=  ( self.F * self.v_old + self.gdHdx) * h_mean_dt_x #\
        #                         #- (self.g / self.mrc**2) * self.u_old * sqrtu2pv2 #bottom friction force
        # self.sol[2,1:-1,1:-1] += (- self.F * self.u_old + self.gdHdy) * h_mean_dt_y #\
        #                         #- (self.g/ self.mrc**2) * self.v_old * sqrtu2pv2

SOLVER = None
def _init_SOLVER(params):
    global SOLVER
    SOLVER = SWE_Solver(params)

def solve(num, vec_in, params):
    """solver"""
    if SOLVER is None:
        _init_SOLVER(params)
    if num == 1:
        tstep, sol = SOLVER._return_sol(vec_in)
    else:
        sol = np.zeros_like(vec_in)
        tstep = 0
        # #p = Pool()
        # #results = p.map(SOLVER._return_sol, vec_in)

        if params["run_solver_in_parallel"]:
            dt, sol = zip(*Parallel(n_jobs=params["ncores"])(delayed(SOLVER._return_sol)(row)
                for row in vec_in.T
                ))
            dt = np.array(dt)
            sol = np.array(sol).T
            tstep = np.sum(dt)/num
        else:
            for i in range(num):
                dt, sol[:, i] = SOLVER._return_sol(vec_in[:, i])
                tstep += dt
            tstep /= num
    return tstep, sol