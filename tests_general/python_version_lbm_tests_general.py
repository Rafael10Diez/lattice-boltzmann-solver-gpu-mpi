# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    scipy.interpolate  import  RectBivariateSpline
import  matplotlib.pyplot as plt
from    os.path import  dirname, abspath
from    sys     import  path as sys_path
import  zipfile

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

memo_id = [0]
def unique_id():
    result      = f"run_{memo_id[0]:03d}_"
    memo_id[0] += 1
    return result

reader_zip = lambda fname: zipfile.Path(dirname(fname),at=basename(fname)).read_text().split('\n')

# ------------------------------------------------------------------------
#                  Utilities
# ------------------------------------------------------------------------

def midpoints(L,n):
    u = torch.linspace(0,L,n+1)
    return (u[:-1] + 0.5*torch.diff(u)).tolist()

# ------------------------------------------------------------------------
#                  Sine Wave
# ------------------------------------------------------------------------

class Sine_Wave_Vel:
    def __init__(self,params):
        self._A      =  params['A']
        self._lambda =  params['lambda']
        self._nu     =  params['nu']
        self._kappa  = 2*math_pi/self._lambda
        self._omega  = self._nu*(self._kappa**2)
        self._stable = params['stable_wave']
    
    def __call__(self, y, t0, sin = math_sin, exp = math_exp):
        t     =  0 if self._stable else t0
        u0    =  self._A*sin(self._kappa*y)
        return [exp(-self._omega*t)*u0, 0*y*t, 0*y*t]

    def get_force(self, y, sin = math_sin):
        if self._stable:  return [self._nu*(self._kappa**2)*self._A*sin(self._kappa*y), y*0, y*0]
        else           :  return [y*0, y*0, y*0]

# ------------------------------------------------------------------------
#                  Poiseuille Flow
# ------------------------------------------------------------------------

class Poiseuille_Flow:
    def __init__(self, params):
        # ux = u0*(1 - (y**2) / (L**2))
        # G = 2*nu*u0/(L**2)
        self._u0     =  params['A']
        self._nu     =  params['nu']
        self._R      =  params['L_xyz'][1]/2
        self._stable = params['stable_wave']
        self._G      =  (2*self._nu*self._u0/(self._R**2)) * int(self._stable)
        self._omega  = 100*self._nu/self._R**2
        assert self._stable
    
    def __call__(self, y, t, sin = math_sin, exp = math_exp):
        assert self._stable
        y_adim = y/self._R - 1
        return [self._u0*(1 - y_adim**2) + 0*y*t, 0*y*t, 0*y*t]

    def get_force(self, y, sin = math_sin):
        assert self._stable
        return [y*0+self._G, y*0, y*0]

# ------------------------------------------------------------------------
#                  Part 1 (Sine Wave Decay Test)
# ------------------------------------------------------------------------

def part_1_main_runner(type_flow,frac_nstep_max, ny, K_lambda, params_exact_sol, model_type_force, verbose=False, t_end0=None):
    params_exact_sol['lambda']  =  params_exact_sol['L_xyz'][1]/K_lambda
    exact_sol                   =   {'Sine_Wave': Sine_Wave_Vel, 'Poiseuille_Flow': Poiseuille_Flow,
                                     }[type_flow.split('+')[0]](params_exact_sol)
    
    params = dict( L_xyz            =  params_exact_sol['L_xyz']                       ,
                   n_xyz            =  ({'shan_chen'  : 3, 
                                         'guo'        : 3,
                                         'pressure_bc': 8,
                                         'none'       : 3}[model_type_force], ny*K_lambda, 1),
                   model_type_force =  model_type_force                                ,
                   scheme           =  'D3Q19'                                         ,
                   BC_type_xyz      =  ['Periodic', {'Sine_Wave': 'Periodic', 'Poiseuille_Flow': 'Walls'}[type_flow.split('+')[0]], 'Periodic'],
                   nu               =  params_exact_sol['nu']                          ,
                   dir_delta_P      =  0                                               ,
                   mode_inlet_outlet_bc = list(type_flow.split('+')[1]) if type_flow.split('+')[0] == 'Poiseuille_Flow' else ['P','P'])
    assert params['n_xyz'][0] >= 3
    params['size_elem'] = params['L_xyz'][1]/params['n_xyz'][1]
    
    xyz_p            =  [torch.tensor(midpoints(params['L_xyz'][ax], 
                                                params['n_xyz'][ax]),dtype=torch.double,device=device) for ax in range(3)]
    nstep_max        =  128*(4 if params_exact_sol['stable_wave'] else 1)
    dt_ideal         =  LBM_Engine.get_dt_ideal(params, {'Sine_Wave': 1., 'Poiseuille_Flow': (math_sqrt(3./16)+0.5)}[type_flow.split('+')[0]])
    if t_end0 is None:
        t_end        =  dt_ideal*nstep_max
    else:
        t_end        =  t_end0
    
    # if verbose: 
    # print(f"Time scale check: {t_end = :.3e} {t_end/exact_sol._omega = :.3e}")
    if frac_nstep_max is None:
        nstep = int(round(t_end/dt_ideal,0))
    else:
        nstep                  =  nstep_max//frac_nstep_max
    params['nsteps']           =  nstep
    params['snap_iter_start']  =  nstep*100
    params['snap_freq']        =  nstep*100
    params['delta_t'] =  t_end/params['nsteps']

    def get_rho_vel(t,also_force=False,also_noise=False):
        U      =  [torch.zeros(params['n_xyz'],dtype=torch.double,device=device) for _ in range(3)]
        U[0]  +=  exact_sol(xyz_p[1].view(1,-1,1), t, sin=torch.sin)[0] + U[0]
        rho    =  torch.ones_like(U[0])
        if also_force:
            F_xyz    = lmap(torch.zeros_like, U)
            F_xyz[0] = exact_sol.get_force(xyz_p[1].view(1,-1,1), sin=torch.sin)[0] + F_xyz[0]
        if also_noise:
            U[0] *= 1 # + 0.01*(2*torch.rand(U[0].shape) - 1)
        return [rho, U, F_xyz] if also_force else [rho, U]
    obj              =  LBM_Engine(params, *get_rho_vel(0,also_force=True,also_noise=params_exact_sol['stable_wave']))
    def get_error(t_now):
        rho_now, U_exact_now = get_rho_vel(t_now)
        L_inf                = lambda x: float(x.abs().max())
        # print(obj.get_rho_phys()[:,0,0].tolist())
        return  max((L_inf(A - B)/L_inf(A)) for A,B in zip([*U_exact_now],
                                                           [*obj.get_U_phys()]) if L_inf(A)>1e-12)
    if verbose: print(f'\n-------------- Run with {nstep} --------------')
    error_0 = get_error(0)
    obj.iterate(params['nsteps'], params['snap_iter_start'], params['snap_freq'], pjoin(_folder_,'test_runs',f"{unique_id()}_part_1_{type_flow}_{model_type_force}_stable_wave_{params_exact_sol['stable_wave']}"))
    iters_full = params['nsteps']
    error_cfd_out = get_error(params['delta_t']* iters_full)
    if verbose: print(f"{iters_full:4d} {nstep:4d} {error_cfd_out:.3e} {error_0:.3e} {obj._tau:.3e} {obj._nu_lbm:.3e} {params['delta_t']:.3e} {dt_ideal:.3e}")
    return deepcopy(dict(params           = params           , 
                         params_exact_sol = params_exact_sol , 
                         obj_dict         = obj.__dict__     ,
                         error_cfd_out    = error_cfd_out    ,
                         dt_ideal         = dt_ideal         ,
                         t_end            = t_end            ))

def part_1_script(type_flow):
    line             = '-'*60
    params_exact_sol = {'A'      : 1e-3*rand_between(1,1.2)                         ,
                        'L_xyz'  : tuple([float('nan'), rand_between(1.1,1.2), float('nan')]) ,
                        'nu'     : {'Sine_Wave'    : 0.025, 'Poiseuille_Flow':0.25}[type_flow.split('+')[0]]}
    for     stable_wave      in {'Sine_Wave': [0,1], 'Poiseuille_Flow': [1]}[type_flow.split('+')[0]]:
        for model_type_force in ({'Sine_Wave'    : ['shan_chen', 'guo'],
                                  'Poiseuille_Flow': ['shan_chen', 'guo','pressure_bc']}[type_flow.split('+')[0]] if stable_wave else ['none']):
            print(f"\n{line} Stable Wave ({type_flow}) (stable_wave: {bool(stable_wave)}) (model_type_force: {model_type_force}) {line}")
            params_exact_sol['stable_wave']  =  stable_wave
            all_frac_nstep_max = [  4,   2,   1]
            all_ny             = {'Sine_Wave': [ 16,  32,  64], 'Poiseuille_Flow': [ 16,  32,  64]}[type_flow.split('+')[0]]
            all_K_lambda       = [  4,   2,   1]

            def convergence_timestep():
                print(f"\n{line} Convergence Time step {line}")
                print("steps     ny        time_end   error_cfd   ratio (error)    tau        delta_t    delta_t_ideal")
                prev = float('nan')
                for frac_nstep_max in all_frac_nstep_max:
                    info      =  part_1_main_runner(type_flow,frac_nstep_max, min(all_ny), min(all_K_lambda), params_exact_sol, model_type_force)
                    ratio     =  prev/info['error_cfd_out']
                    prev      =  info['error_cfd_out']
                    obj_dict  =  info['obj_dict']
                    params    =  info['params']
                    print(f"{params['nsteps']:8d}  {params['n_xyz'][1]:8d}  {info['t_end']:.3e}  {info['error_cfd_out']:.3e}  {ratio:16.3f}  {obj_dict['_tau']:.3e}  {params['delta_t']:.3e}  {info['dt_ideal']:.3e}")
                return info['t_end']
            t_end = convergence_timestep()

            def convergence_grid_spacing(t_end):
                write_title = True
                prev = float('nan')
                for ny in all_ny:
                    info      =  part_1_main_runner(type_flow,None, ny, min(all_K_lambda), params_exact_sol, model_type_force,t_end0=float(t_end))
                    ratio     =  prev/info['error_cfd_out']
                    prev      =  info['error_cfd_out']
                    obj_dict  =  info['obj_dict']
                    params    =  info['params']
                    if write_title:
                        print(f"\n{line} Convergence Grid Spacing (tau = {obj_dict['_tau']:.3e}) {line}")
                        print("steps     ny        time_end   error_cfd   ratio (error)    tau        delta_t    delta_t_ideal")
                        write_title = False
                    print(f"{params['nsteps']:8d}  {params['n_xyz'][1]:8d}  {info['t_end']:.3e}  {info['error_cfd_out']:.3e}  {ratio:16.3f}  {obj_dict['_tau']:.3e}  {params['delta_t']:.3e}  {info['dt_ideal']:.3e}")
            convergence_grid_spacing(t_end)

            def changes_lambda():
                print(f"\n{line} Convergence Wave Length {line}")
                print("steps     ny       K_lambda   time_end   error_cfd   ratio (error)    tau        delta_t    delta_t_ideal")
                prev = float('nan')
                for K_lambda in all_K_lambda:
                    info      =  part_1_main_runner(type_flow,1, max(all_ny), K_lambda, params_exact_sol, model_type_force)
                    ratio     =  prev/info['error_cfd_out']
                    prev      =  info['error_cfd_out']
                    obj_dict  =  info['obj_dict']
                    params    =  info['params']
                    print(f"{params['nsteps']:8d}  {params['n_xyz'][1]:8d}  {K_lambda:8d}  {info['t_end']:.3e}  {info['error_cfd_out']:.3e}  {ratio:16.3f}  {obj_dict['_tau']:.3e}  {params['delta_t']:.3e}  {info['dt_ideal']:.3e}")
                return info['t_end']
            if type_flow.split('+')[0] != 'Poiseuille_Flow':
                changes_lambda()

# ------------------------------------------------------------------------
#                  Part 2
# ------------------------------------------------------------------------

def interp_fluent_vel(params, fluent_dtype='node'):
    as_key     =  lambda x: tuple([(float(y) if abs(float(y) - int(y)) > 1e-10 else int(y)) for y in lmap(bool,x)])
    case_type  =  {((1,0,0), (0,0,0)): 'gradp' , 
                   ((0,0,0),(1,0,0)): 'moving'}[as_key(params['body_F']), as_key(params['U_box'])]
    def get_raw_data():
        fname      =  pjoin(_folder_, 'flow_data_fluent.zip', f'flow_data_{fluent_dtype}_{case_type}.dat')
        A          =  [[y.strip(' ') for y in x.split(',')] for x in reader_zip(fname) if x.strip(' \n')]
        header     =  A[0]
        dtypes     =  [type(eval(x)) for x in A[1]]
        for f,key in zip(dtypes,header):
            if f==int: print('Fluent integer field:', key, fname)
            else     : assert f==float
        data       =  [[f(y) for f,y in zip(dtypes,A[i])] for i in range(1,len(A))]
        return {key: [x[j] for x in data] for j,key in enumerate(header)}
    def get_ids(x,size_elem):
        keys     = [f"{val:.10e}" for val in x]
        uids     = {k:i for i,k in enumerate(sorted(set(keys),key=float))}
        result   = [uids[k] for k in keys]
        x1d      = [None for _ in uids]
        for i,val in zip(result,lmap(float,x)):
            if not (x1d[i] is None): assert abs(x1d[i]-val)<1e-10
            x1d[i] = val
        min_diff = min([(x1d[i]-x1d[i-1]) for i in range(1,len(x1d))])
        assert min_diff>(size_elem/10)
        return result,x1d
    def mk_2d(size_elem):
        data   = get_raw_data()
        [data.pop(key) for key in ['nodenumber', 'cellnumber'] if key in data]
        x,y              =  [data[f'{c}-coordinate'] for c in 'xy']
        (ii,x1),(jj,y1)  =   get_ids(x,size_elem), get_ids(y,size_elem)
        shape  =  max(ii)+1, max(jj)+1
        print('Imported Fluent shape: ', shape)
        for key in list(data.keys()):
            A = [[float('nan') for _ in range(shape[1])] for _ in range(shape[0])]
            for i,j,val in zip(ii,jj,data[key]):
                A[i][j] = val
            data[key]  =  torch.tensor(A,dtype={float:torch.double, int:torch.long}[type(data[key][0])], device=device)
        assert (not 'x1' in data) and (not 'y1' in data)
        data['x1'] = torch.tensor(x1,dtype=torch.double, device=device)
        data['y1'] = torch.tensor(y1,dtype=torch.double, device=device)
        return data
    def reinterp():
        size_elem  =   params['L_xyz'][1]/params['n_xyz'][1]
        data       =   mk_2d(size_elem)
        x,y,_      =  [(torch.tensor(range(n),dtype=torch.double,device=device)+0.5)*size_elem for n in params['n_xyz']]
        def padded(x,y,A,n):
            import numpy as np
            extrap_1d  =  lambda x: [2*x[0]-x[1]] + tolist(x) + [2*x[-1]-x[-2]]
            period_1d  =  lambda x,n: tolist(x)[-n:] + tolist(x) + tolist(x)[:n]
            def period_1d(x,n):
                x = tolist(x)
                L = len(x)
                return [x[(L-1-i)%L] for i in range(n)] + x + [x[i%L] for i in range(n)]
            x,y  =  deepcopy(lmap(tolist,[x,y]))
            A    =  deepcopy(tolist(A))
            for _ in range(n): x,y = lmap(extrap_1d,[x,y])
            A    =  period_1d([period_1d(row,n) for row in A],n)
            return lmap(np.array,[x,y,A])
        def get_field(key,fill_val=float('nan')):
            fluent_x1                                  =  data['x1'].cpu().numpy()
            fluent_y1                                  =  data['y1'].cpu().numpy()
            fluent_z                                   =  data[key].cpu().numpy() + 0.
            fluent_z[data[key].isnan().cpu().numpy()]  =  fill_val
            return torch.tensor(RectBivariateSpline(*padded(fluent_x1, fluent_y1, fluent_z, len(fluent_y1)))(x.reshape(-1,1).cpu().numpy(),
                                                                                                             y.reshape(1,-1).cpu().numpy()),
                                dtype=torch.double,device=device)
        new_u = get_field('x-velocity',fill_val=0.)
        new_v = get_field('y-velocity',fill_val=0.)
        return new_u, new_v
    new_u, new_v  =  [arr.reshape(*arr.shape,1) for arr in reinterp()]
    if case_type == 'moving':
        new_u -= 1
    return [new_u, new_v, 0*new_v], case_type

def part_2_script_setup(Ux,Fx,ny,advancing, nu, K_delta_t = 1, from_zero = False, K_geom = 1, w_box=0.2, h_box=0.8, 
                        report_avoid_fluent_corners = None, nstep_K_mult = 1, ideal_tau = None, save_freq = None):
    line      = '-'*60
    
    obstacle  = Obstacle_Box(dict(U = [Ux,0,0], bbox = [[1           , 1+w_box   ], # [1            , 4 - 2.7     ]
                                                        [1 - h_box/2 , 1+h_box/2 ], # [0.6          , 2 - 0.6     ]
                                                        [float('nan') , float('nan')]], advancing=advancing))
    params    = {'U_box'            : [Ux,0,0]                                ,
                 'body_F'           : [Fx,0,0]                                ,
                 'L_xyz'            : tuple([float('nan'), 2., float('nan')]) ,
                 'n_xyz'            : [None, ny, None]                        ,
                 'nu'               : nu                                      ,
                 'model_type_force' :  'guo'                                  ,
                 'scheme'           :  'D3Q19'                                ,
                 'BC_type_xyz'      : ['Periodic', 'Walls', 'Periodic']       ,
                 'nsteps'           : 20_000                                  ,
                 'snap_iter_start'  :  200_000                                ,
                 'snap_freq'        :  200_000                                ,
                 }
    if abs(K_geom-1)>1e-10: assert from_zero
    get_n_grid = lambda L: int_ceil(float(L)/params['L_xyz'][1]*params['n_xyz'][1])
    params['n_xyz'][0]  =  get_n_grid(4.*K_geom)
    params['n_xyz'][2]  =  1
    params['n_xyz']     =  tuple(params['n_xyz'])
    params['size_elem'] = params['L_xyz'][1]/params['n_xyz'][1]
    if abs(K_geom-1)<1e-10:  assert params['n_xyz'][0] == (2 * K_geom * params['n_xyz'][1])
    if not (ideal_tau is None):
        params['delta_t']   =  LBM_Engine.get_dt_ideal(params, ideal_tau = ideal_tau) * K_delta_t
    else:
        params['delta_t']   =  LBM_Engine.get_dt_ideal(params) * K_delta_t

    U_ini, case_type  =  interp_fluent_vel(params)
    U_fluent           =  [(arr+0.) for arr in U_ini]
    scale_U_fluent     =  max(float(arr.abs().max()) for arr in U_fluent)
    F_ini              =  lmap(torch.zeros_like,U_ini)
    F_ini[0]          +=  Fx
    rho_ini            =  torch.ones_like(U_ini[0])
    obj                =  LBM_Engine(params, rho_ini, 
                                     U_ini if not from_zero else lmap(lambda x: 1e-3*torch.rand(x.shape,device=device,dtype=x.dtype),U_ini), 
                                     F_ini, obstacle=obstacle)
    nfreq              =  100
    return dict(obj            = obj,
                params         = params,
                U_fluent       = U_fluent,
                nfreq          = nfreq, 
                scale_U_fluent = scale_U_fluent, 
                U_ini          = U_ini,
                F_ini          = F_ini,
                rho_ini        = rho_ini,
                case_type      = case_type,
                nstep_K_mult   = nstep_K_mult,
                save_freq      = save_freq if not (save_freq is None) else (params['nsteps']*1_000),
                report_avoid_fluent_corners = report_avoid_fluent_corners)


def part_2_iterator(info):
    if info['report_avoid_fluent_corners'] is None:
        ftrim = lambda x: x
    else:
        i_trim = info['report_avoid_fluent_corners']
        assert (i_trim>0) and (type(i_trim) == int)
        ftrim  = lambda x: x[i_trim:-i_trim,i_trim:-i_trim,:]
    obj,params,nfreq,U_fluent,scale_U_fluent = [info[key] for key in 'obj,params,nfreq,U_fluent,scale_U_fluent'.split(',')]
    last_iter, t0   =  [0], [time()]
    folder_name  =  f"test_runs/{unique_id()}_flows_past_rectangle_nu_{info['params']['nu']}_{info['tag']}_{info['case_type']}_adv_{int(obj.obstacle.advancing)}"
    def quick_pic(iters_full):
        fig,ax     =  plt.subplots()
        u_field    =  info['obj'].get_U_phys()[0].cpu().numpy()[:,:,0]
        if not (info['obj'].obstacle is None):
            (i0,i1),(j0,j1),(k0,k1) = info['obj'].obstacle_ijk_bbox
            if i0<i1:
                u_field[i0:i1+1,j0:j1+1] = float('nan')
            else:
                u_field[:i1+1,j0:j1+1] = float('nan')
                u_field[  i0:,j0:j1+1]  = float('nan')
        title      =  f"Flow past rectangle: nu = {info['params']['nu']}"
        mk_edges   =  lambda L,n: torch.linspace(0,L,n+1).tolist()
        xyz_edges  =  [mk_edges(info['params']['n_xyz'][i]*info['params']['L_xyz'][1]/info['params']['n_xyz'][1], 
                                info['params']['n_xyz'][i]) for i in range(3)]
        pcm = plt.pcolormesh(xyz_edges[0], xyz_edges[1], u_field.transpose())
        cbar = fig.colorbar(  pcm                                        ,
                              ax           =  [ax]                       ,
                              location     =  'right'                    ,
                              shrink       =  0.5                        ,
                              pad          =  0.025                      ,
                              # ticks        =  np.linspace(vmin,vmax, 5)  ,
                              # format       =  fmt_cbar                   ,
                              aspect       =  15                         )
        ax.set_aspect('equal', adjustable='box')
        plt.title(title)
        plt.axis('equal')
        # ax.set_axis_off()
        fig.set_size_inches([6,3])
        cbar.ax.set_title('U')#,ha='left',y=1.05)
        ax.margins (x=0) ; plt.margins(0,0)
        pic_fname    =  f"{folder_name}_iters_{iters_full:06d}.png"
        os_system(f'mkdir -p "{pjoin(_folder_, folder_name)}"')
        fig.savefig(  pjoin(_folder_, folder_name, basename(pic_fname))  ,
                      bbox_inches  =  'tight'                            ,
                      dpi          =  800                                ,
                      pad_inches   =  0.02                               )
        plt.close(fig)
    def f_max_error(iters_full):
        if (iters_full-1)%nfreq == 0:
            max_error_cfd  =  max([float((ftrim(a - b)).abs().max()) for a,b in zip(obj.get_U_phys(),U_fluent)])/scale_U_fluent
            print(iters_full,f"{(max_error_cfd*100):.3f} %", f"(iters/sec: {((iters_full-last_iter[0])/(time()-t0[0])):.3f})")
            last_iter[0], t0[0] = iters_full, time()
        if ((iters_full>1) and ((iters_full-1)%info['save_freq'] == 0)) or (iters_full == params['nsteps']):
            quick_pic(iters_full)
    obj.iterate(params['nsteps'], params['snap_iter_start'], params['snap_freq'], pjoin(_folder_, folder_name), f_max_error)
    return info

def part_2_iterator_add_noise(info):
    nsteps = info['params']['nsteps']
    info['params']['nsteps'] = nsteps // 2
    info['tag'] = '(before_noise)'
    part_2_iterator(info)
    rho_0 = info['obj'].F.sum()
    info['obj'].F *= 1+1e-3*torch.rand(info['obj'].F.shape, dtype  = info['obj'].F.dtype ,
                                                            device = info['obj'].F.device)
    info['obj'].F *= rho_0 / info['obj'].F.sum()
    info['tag'] = '(after_noise)'
    info['params']['nsteps'] = int((nsteps // 2) * info['nstep_K_mult'])
    before_noise = deepcopy(info)
    part_2_iterator(info)
    info['params']['nsteps'] = nsteps
    info['before_noise'] = before_noise
    return info

def parabolic_distrib(umax,n):
    import numpy as np 
    U =((np.array(range(n)) + 0.5)/float(n) - 0.5)**2
    U /= U.max()
    return (1 - U)*umax

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    if isdir(pjoin(_folder_,"test_runs")):
        os_system(f'rm -rf "{pjoin(_folder_,"test_runs")}"')
    
    memo_id[0] = 30
    part_1_script('Sine_Wave')

    memo_id[0] = 40
    part_1_script('Poiseuille_Flow+PP')

    memo_id[0] = 50
    part_1_script('Poiseuille_Flow+PU')

    memo_id[0] = 60
    info_high_visc_gradp = part_2_iterator_add_noise(part_2_script_setup(0., 1, 200, False, 0.1, w_box=0.3)) # Ux, Fx, ny, advancing, nu 

    memo_id[0] = 70
    info_low_visc_gradp  = part_2_iterator_add_noise(part_2_script_setup(0., 1, 400, False, 0.0017, K_delta_t = 0.1,from_zero=True, w_box=0.1, h_box=0.1, nstep_K_mult=1)) # Ux, Fx, ny, advancing, nu
    
    memo_id[0] = 80
    info_moving          = part_2_iterator_add_noise(part_2_script_setup(-1, 0., 400, False, 0.1, w_box=0.3, report_avoid_fluent_corners=10)) # Ux, Fx, ny, advancing, nu 
    # # info_adv           = part_2_iterator_add_noise(part_2_script_setup(-1, 0., 200, True , 0.1, report_avoid_fluent_corners=5, K_delta_t = 0.6*10,ideal_tau=1, save_freq=10))
    
    memo_id[0] = 90
    info_adv             = part_2_iterator_add_noise(part_2_script_setup(-1, 0., 200, True , 0.1, w_box=0.3, report_avoid_fluent_corners=5, K_delta_t = 0.6*10*8/10, ideal_tau = 1, save_freq=100))

    # final post-processing:
    def detailed_pics(info, pic_fname,tag,custom_title=None):
        import numpy as np
        if tag == 'before':
            info = info['before_noise']
        else:
            assert tag == 'after'
        for mode in ['normal', 'error']:
            fig,ax     =  plt.subplots()
            u_field    =  info['obj'].get_U_phys()[0].cpu().numpy()[:,:,0]
            if not (info['obj'].obstacle is None):
                (i0,i1),(j0,j1),(k0,k1) = info['obj'].obstacle_ijk_bbox
                if i0<i1:
                    u_field[i0:i1+1,j0:j1+1] = float('nan')
                else:
                    u_field[:i1+1,j0:j1+1] = float('nan')
                    u_field[  i0:,j0:j1+1]  = float('nan')
            title      =  f"Flow past rectangle: nu = {info['params']['nu']}\n{tag.title()} noise"
            if custom_title: title += ' '+custom_title
            mk_edges   =  lambda L,n: torch.linspace(0,L,n+1).tolist()
            xyz_edges  =  [mk_edges(info['params']['n_xyz'][i]*info['params']['L_xyz'][1]/info['params']['n_xyz'][1], 
                                    info['params']['n_xyz'][i]) for i in range(3)]
            if mode == 'error':
                u0_fluent_field =  info['U_fluent'][0].cpu().numpy()[:,:,0]
                arr = np.fabs(u_field - u0_fluent_field)/np.fabs(u0_fluent_field).max()
                fmt_cbar = lambda x,_: f"{(x*100):.1f}%"
                cbar_title = '|Error U|'
                ii = 5
                arr = arr[ii:-ii,ii:-ii]
                xyz_edges[0] = xyz_edges[0][ii:-ii]
                xyz_edges[1] = xyz_edges[1][ii:-ii]
                vmin = 0.
                vmax = arr[5:-5,5:-5][~np.isnan(arr[5:-5,5:-5])].max()
            else:
                assert mode == 'normal'
                arr = u_field
                fmt_cbar = lambda x,_: f"{x:.1f}"
                cbar_title = 'U'
                vmin = arr[~np.isnan(arr)].min()
                vmax = arr[~np.isnan(arr)].max()
            print(f"{arr.min() =} {arr.max() =} {vmin =} {vmax =}")
            pcm = plt.pcolormesh(xyz_edges[0], xyz_edges[1], arr.transpose(), vmin = vmin, vmax = vmax)
            cbar = fig.colorbar(  pcm                                        ,
                                  ax           =  [ax]                       ,
                                  location     =  'right'                    ,
                                  shrink       =  0.5                        ,
                                  pad          =  0.025                      ,
                                  # ticks        =  np.linspace(vmin,vmax, 4)  ,
                                  format       =  fmt_cbar                   ,
                                  aspect       =  15                         )
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.title(title)
            plt.axis('equal')
            ax.set_axis_off()
            folder_name = 'pics_ready'
            fig.set_size_inches([6,3])
            cbar.ax.set_title(cbar_title,ha='left',y=1.05)
            ax.margins (x=0) ; plt.margins(0,0)
            os_system(f'mkdir -p "{pjoin(_folder_, folder_name)}"')
            fig.savefig(  pjoin(_folder_, folder_name, f"{basename(pic_fname)}_{mode}_{tag}_noise.png")  ,
                          bbox_inches  =  'tight'                            ,
                          dpi          =  800                                ,
                          pad_inches   =  0.02                               )
            plt.close(fig)

    detailed_pics(info_high_visc_gradp, 'high_visc_gradp','before')
    detailed_pics(info_high_visc_gradp, 'high_visc_gradp','after')


    detailed_pics(info_low_visc_gradp, 'low_visc_gradp','before')
    detailed_pics(info_low_visc_gradp, 'low_visc_gradp','after')

    detailed_pics(info_moving, 'moving','before')
    detailed_pics(info_moving, 'moving','after')

    detailed_pics(info_adv, 'advancing','before',custom_title='(2 loops)')
    detailed_pics(info_adv, 'advancing','after', custom_title='(4 loops)')
