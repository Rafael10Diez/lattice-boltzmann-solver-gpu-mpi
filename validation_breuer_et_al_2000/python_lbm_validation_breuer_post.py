# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    scipy.interpolate  import  RectBivariateSpline
import  matplotlib.pyplot as plt
from    os.path import  dirname, abspath
from    sys     import  path as sys_path
import  numpy as np
import struct

# ------------------------------------------------------------------------
#                  Defaults
# ------------------------------------------------------------------------

try:
    _folder_
except:
    _folder_  =  dirname(abspath(__file__))

sys_path.append(dirname(_folder_))
from python_version_lbm_engine       import *
sys_path.pop()
sys_path.append(_folder_)
from data_Cd_breuer.data_drag_Breuer import get_curve_drag_Breuer
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

def import_array_bin(fname):
    with open(fname, "rb") as f:
        n0,n1,n2  =  [ pop1(struct.unpack('i',f.read(4))) for _ in range(3)]
        A         =  np.zeros((n0,n1,n2), dtype=float)
        for          i in range(n0):
            for      j in range(n1):
                for  k in range(n2):
                    A[i,j,k]  =  pop1(struct.unpack('d',f.read(8)))
    return A 

# ------------------------------------------------------------------------
#                  Process Subfolder
# ------------------------------------------------------------------------

def scan_params(subfolder):
    A = reader(pjoin(subfolder,'params.dat'))
    get_line = lambda tag: [x for x in A if tag in x]
    result = {}
    for tags in [' // size_elem_phys dt_phys nu_phys scale_rho',
                 ' // cs',
                 ' // obstacle_xyz_ini_0[0:2]',
                 ' // obstacle_xyz_ini_1[0:2]',
                 ' // total_num_iters snap_iter_start snap_freq',
                 ' // nx ny nz']:
        line,  =  get_line(tags)
        tags2  =  tags
        if '[0:2]' in tags2: 
            tags2  = tags2.replace('//','').replace('[0:2]','').strip(' ')
            tags2  =  ' // ' + ' '.join([f"{tags2}[{i}]"  for i in range(3)])
        else:
            tags2  =  tags
        result.update({k:eval(v) for k,v in zip(tags2.split('//')[1].split(),
                                                line .split('//')[0].split())})
    result.update({f"L{c}": (result[f"n{c}"] * result["size_elem_phys"]) for c in 'xyz'})
    result['L_box']                  = [(result[f'obstacle_xyz_ini_1[{i}]'] - result[f'obstacle_xyz_ini_0[{i}]']) for i in range(3)]
    result['L_xyz'], result['n_xyz'] = [[result[f"{t}{c}"] for c in 'xyz'] for t in 'Ln']
    return result

def import_iter(subfolder, iter_last, info, U_drag_ref, rho_drag_ref):
    result = {}
    for tag in ['U','V','W','rho']:
        result[tag] = import_array_bin(pjoin(subfolder,f"array_iter_{iter_last:09d}_{tag}.dat"))
    result['P']  = result['rho'] * (info['params']['cs']**2) * ((info['params']['size_elem_phys']/info['params']['dt_phys'])**2)
    result['P'] -= np.mean(result['P'][0])
    result['drag_ref'] = get_drag(result, info, U_drag_ref, rho_drag_ref)
    return result

def get_drag(data, info, U_drag_ref, rho_drag_ref):
    P_inlet      =  np.mean(data['P'][1,:,:])
    P_outlet     =  np.mean(data['P'][-2,:,:])
    delta_L      =  info['params']['size_elem_phys']*(info['params']['nx']-3)
    Lx,Ly,Lz     =  [info['params'][f'L{c}'] for c in 'xyz']
    body_force   =  (P_inlet-P_outlet)*Ly*Lz*Lx/delta_L
    shear_force  =  info['params']['nu_phys']*(np.mean(data['U'][:,0,:]) + np.mean(data['U'][:,-1,:]))/(info['params']['size_elem_phys']/2)*Lx*Lz
    box_force    =  body_force - shear_force
    L_box        =  info['params']['L_box']
    return  dict(Cd          = box_force/(L_box[1]*Lz)/(0.5*(U_drag_ref**2)*rho_drag_ref),
                 body_force  = body_force ,
                 shear_force = shear_force,
                 box_force   = box_force  )
    # mu*dUdy*Lx*Lz

def process_folder(subfolder, U_drag_ref = None, rho_drag_ref = None):
    info              =  {}
    info['params']    =  scan_params(subfolder)
    assert type(info['params']['total_num_iters']) == int
    info['data'] = {info['params']['total_num_iters']: import_iter(subfolder, info['params']['total_num_iters'], info, U_drag_ref, rho_drag_ref)}
    info['subfolder'] = subfolder
    return info

def get_inner(xx,a,b):
    all_i = [i for i,c in enumerate(xx) if (a<=c<=b)]
    return all_i[0], all_i[-1]

def plot_2d(info):
    fields  =  pick_highest_key(info['data'])
    for tag in ['U', 'V', 'W', 'rho', 'P']:
        fig, ax    =  plt.subplots()
        mk_edges   =  lambda L,n: torch.linspace(0,L,n+1).tolist()
        xyz_edges  =  [mk_edges(info['params']['n_xyz'][i]*info['params']['L_xyz'][1]/info['params']['n_xyz'][1], 
                                info['params']['n_xyz'][i]) for i in range(3)]
        arr        =  fields[tag][:,:,0]
        i0, i1  =  get_inner(xyz_edges[0], info['params'][f'obstacle_xyz_ini_0[0]'], info['params'][f'obstacle_xyz_ini_1[0]'])
        j0, j1  =  get_inner(xyz_edges[1], info['params'][f'obstacle_xyz_ini_0[1]'], info['params'][f'obstacle_xyz_ini_1[1]'])
        arr[i0:i1+1,j0:j1+1] = float('nan')
        title      =  f"Field: {tag}, Re: {info['Re']}"
        fmt_cbar   = lambda x,_: f"{x:.2f}"
        cbar_title = tag
        dx_plot =  info['params']['L_xyz'][1]
        i0,i1   =  get_inner(xyz_edges[0], info['params'][f'obstacle_xyz_ini_0[0]'] - dx_plot, info['params'][f'obstacle_xyz_ini_1[0]'] + dx_plot)
        vmin    =  arr[i0:i1][~np.isnan(arr[i0:i1])].min()
        vmax    =  arr[i0:i1][~np.isnan(arr[i0:i1])].max()
        pcm = plt.pcolormesh(xyz_edges[0][i0:i1+1], xyz_edges[1], arr[i0:i1,:].transpose(), vmin = vmin, vmax = vmax)
        cbar = fig.colorbar(  pcm                                        ,
                              ax           =  [ax]                       ,
                              location     =  'right'                    ,
                              shrink       =  0.5                        ,
                              pad          =  0.025                      ,
                              format       =  fmt_cbar                   ,
                              aspect       =  15                         )
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.title(title)
        plt.axis('equal')
        ax.set_axis_off()
        folder_name = 'post_breuer'
        fig.set_size_inches([6,3])
        cbar.ax.set_title(cbar_title,ha='left',y=1.05)
        ax.margins (x=0) ; plt.margins(0,0)
        os_system(f'mkdir -p "{pjoin(_folder_, folder_name)}"')
        fig.savefig(  pjoin(_folder_, folder_name, f"field_{tag}_Re_{info['Re']}.png")  ,
                      bbox_inches  =  'tight'                            ,
                      dpi          =  800                                ,
                      pad_inches   =  0.02                               )
        plt.close(fig)

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    f_Re_Cd_Brueuer = get_curve_drag_Breuer(do_plot=False)
    pick_highest_key = lambda d: d[sorted(d.keys())[-1]]
    def get_data_Cd():
        all_Re, all_Cd_now, all_Cd_paper = [], [], []
        print(f"{'Folder':30s} {'Re':3s} {'Cd_now':8s} {'Cd_Breuer':8s} {'Error_Cd (%)':8s}")
        for subfolder in [pjoin(_folder_, "local_runs", subfolder) for subfolder in 
                          ['run_breuer_Re_1', 'run_breuer_Re_5', 'run_breuer_Re_10']]:
            info      =  process_folder(subfolder, U_drag_ref = 1, rho_drag_ref = 1)
            info['Re'] = int(basename(subfolder).split('_Re_')[-1])
            all_Re      .append( info['Re'] )
            all_Cd_now  .append( pick_highest_key(info['data'])['drag_ref']['Cd'] )
            all_Cd_paper.append( f_Re_Cd_Brueuer(all_Re[-1])                      )
            error_Cd  = all_Cd_now[-1]/all_Cd_paper[-1]-1
            plot_2d(info)
            print(f"{basename(subfolder):30s} {all_Re[-1]:3d} {all_Cd_now[-1]:8.3f} {all_Cd_paper[-1]:8.3f} {(error_Cd*100):8.3f}%")
        return all_Re, all_Cd_now, all_Cd_paper
    
    all_Re, all_Cd_now, all_Cd_paper  =  get_data_Cd()

    xx_Re    =  np.linspace(0.6,20,100)
    fig, ax  =  plt.subplots()
    ax.plot( xx_Re, f_Re_Cd_Brueuer(xx_Re), 'k' , lw=2, label = 'Breuer (2000)')
    ax.plot(all_Re, all_Cd_now            , 'bs', lw=4, label='LBM Solver')
    ax.set_xlabel('$Re$')
    ax.set_ylabel('$C_d$')
    plt.legend(loc='best')
    fig.set_size_inches([3,2])
    pic_fname  =  pjoin(_folder_, 'post_breuer', f"drag_coeff_comparison.png")
    os_system(f'mkdir -p "{dirname(pic_fname)}"')
    fig.savefig(  pic_fname  , bbox_inches  =  'tight', dpi =  800, pad_inches =  0.02)
    plt.close(fig)