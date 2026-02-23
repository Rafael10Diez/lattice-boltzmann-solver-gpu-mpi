# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    scipy.interpolate  import  RectBivariateSpline
import  matplotlib.pyplot as plt
from    os.path import  dirname, abspath
from    sys     import  path as sys_path

# ------------------------------------------------------------------------
#                  Defaults
# ------------------------------------------------------------------------

_folder_  =  dirname(abspath(__file__))

sys_path.append(dirname(_folder_))
from python_version_lbm_engine import *
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
#                  Utilities
# ------------------------------------------------------------------------

def midpoints(L,n):
    u = torch.linspace(0,float(L),n+1,dtype=torch.double)
    return (u[:-1] + 0.5*torch.diff(u)).tolist()

# ------------------------------------------------------------------------
#                  Part 2
# ------------------------------------------------------------------------

def breuer_setup(Re):
    U_max     = 1
    Ly        = 2

    D         = Ly/8
    Lx_full   = 50*D
    Lx_front  = Lx_full/4

    ny        = 320

    U_box     = 0.
    Fx        = 0.
    advancing = False
    nu        = U_max*D/Re
    K_delta_t = 1
    ideal_tau = 1
    params    = {'U_box'                : [U_box,0,0]                             ,
                 'body_F'               : [Fx,0,0]                                 ,
                 'L_xyz'                : tuple([float('nan'), Ly, float('nan')]) ,
                 'n_xyz'                : (int(round(ny*Lx_full/Ly,0)), ny, 1)    ,
                 'nu'                   : nu                                      ,
                 'model_type_force'     :  'guo'                                  ,
                 'scheme'               :  'D3Q19'                                ,
                 'BC_type_xyz'          : ['Periodic', 'Walls', 'Periodic']       ,
                 'model_type_force'     :  'pressure_bc'                          ,
                 'scheme'               :  'D3Q19'                                ,
                 'dir_delta_P'          :  0                                      ,
                 'mode_inlet_outlet_bc' : ['U','P']                               ,
                 'nsteps'               :  200_000                                ,
                 'snap_iter_start'      :  5_000                                  ,
                 'snap_freq'            :  5_000                                  }
    obstacle  = Obstacle_Box(dict(U    = params['U_box'], 
                                  bbox = [[Lx_front     , Lx_front+D ], # [1            , 4 - 2.7     ]
                                          [Ly/2 - D/2   , Ly/2+D/2   ], # [0.6          , 2 - 0.6     ]
                                          [float('nan') , float('nan')]], advancing=advancing))
    params['size_elem'] = params['L_xyz'][1]/params['n_xyz'][1]
    params['delta_t']   = LBM_Engine.get_dt_ideal(params, ideal_tau = ideal_tau) * K_delta_t
    yp            =  torch.tensor(midpoints(Ly,ny)      , dtype=torch.double, device=device)
    empty         =  lambda: torch.zeros(params['n_xyz'], dtype=torch.double, device=device)
    U_ini, F_ini  =  [[empty() for _ in range(3)] for _ in range(2)]
    U_ini[0]     +=  (U_max*(1-(2*yp/Ly-1)**2)).reshape(1,-1,1)
    F_ini[0]     +=  Fx
    rho_ini       =  torch.ones_like(U_ini[0])
    obj                =  LBM_Engine(params, rho_ini,  U_ini, F_ini, obstacle=obstacle)
    folder_name        =  f'run_breuer_Re_{Re}'
    obj.iterate(params['nsteps'], params['snap_iter_start'], params['snap_freq'], pjoin(_folder_, 'local_runs', folder_name), None, only_export_inputs= True)

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    if isdir(pjoin(_folder_,"test_runs")):
        os_system(f'rm -rf "{pjoin(_folder_,"local_runs")}"')
    breuer_setup(1)
    breuer_setup(5)
    breuer_setup(10)
