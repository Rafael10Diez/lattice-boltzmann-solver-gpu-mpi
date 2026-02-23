# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os.path            import  basename, dirname, abspath, isfile, isdir
from    os.path            import  join                                       as  pjoin
from    os                 import  listdir                                    as  os_listdir
from    math               import  sqrt                                       as  math_sqrt
from    math               import  pi                                         as  math_pi
from    math               import  sin                                        as  math_sin
from    math               import  exp                                        as  math_exp
from    math               import  isnan                                      as  math_isnan
from    time               import  time
from    copy               import  deepcopy
from    os                 import  system                                     as  os_system
import  torch
import  random
import  struct

# ------------------------------------------------------------------------
#                  Defaults
# ------------------------------------------------------------------------

device    = 'cuda'
torch.set_default_device(device)
torch.set_default_dtype(torch.double)
torch.manual_seed(1)
random.seed(1)

RECOMPUTE_PRESSURE_BC  =  True

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

lmap                  =  lambda f,x: list(map(f,x))
lfilter               =  lambda f,x: list(filter(f,x))
listdir_full          =  lambda   x: [pjoin(x,y) for y in os_listdir(x)]
listdir_full_files    =  lambda   x: lfilter(isfile, listdir_full(x))
listdir_full_folders  =  lambda   x: lfilter(isdir , listdir_full(x))

tolist                =  lambda   x: x.tolist() if hasattr(x,'tolist') else x

format_dt             =  lambda   x:  "%02d:%02d:%02d [hh:mm:ss]" % (x//3600, (x%3600)//60, x%60)
rand_between          =  lambda a,b: float(a + (b-a)*random.random())

def pick_d(A,ax,d):
    if   ax == 0:  A = A[d,:,:]
    elif ax == 1:  A = A[:,d,:]
    elif ax == 2:  A = A[:,:,d]
    return A

def as_float(x):
    assert abs(x.max()-x.min())<1e-10
    return float(torch.mean(x))

def reader(fname):
    with open(fname,'r') as f:
        return [x.rstrip('\n') for x in f]

def writer(fname, A):
    with open(fname,'w') as f:
        for x in A:
            f.write(x.rstrip('\n')+'\n')

def writer_bin_arr(fname, A):
    os_system(f'mkdir -p "{dirname(fname)}"')
    with open(fname,'wb') as fw:
        [fw.write(struct.pack('i',s))          for s   in A.shape      ]
        [fw.write(struct.pack('d',float(val))) for val in A.reshape(-1)]

def import_params(fname):
    A = lfilter(None, [x.strip(' ') for x in reader(fname)])
    return {k:eval(v) for k,v in [lmap(lambda x: x.strip(' '), x.split('=')) for x in A]}

def pop1(A):
    x, = A 
    return x

def get_shape3(A):
    assert all((type(x) in [int,float]) for row in A for col in row for x in col)
    return [len(A), pop1(set(len(row) for row in A)), 
                    pop1(set(len(col) for row in A for col in row))]

# ------------------------------------------------------------------------
#                  Utilities
# ------------------------------------------------------------------------

def int_ceil(x):
    y = int(float(x))
    if abs(x-y) > 1e-10:
        while y<x: y+= 1
    return y

# ------------------------------------------------------------------------
#                  LBM
# ------------------------------------------------------------------------

class LBM_Engine:
    def __init__(self, params, rho_ini, U_ini, F_ini, obstacle=None):
        self.params                =  params
        self.scheme                =  self.get_scheme(self.params['scheme'])
        self.pairs_xyz             =  self.get_pairs_xyz(self.scheme)
        self._dt_phys              =  params['delta_t']
        self._size_elem_phys       =  params['size_elem']
        self._n_xyz                =  params['n_xyz']
        self._scale_nu             =    (self._size_elem_phys**2)/self._dt_phys
        self._scale_u              =     self._size_elem_phys/self._dt_phys
        self._scale_rho            =     torch.mean(rho_ini)
        self._scale_body_f         =     self._scale_nu*self._scale_u/(self._size_elem_phys**2)
        self.ini_rho_phys          =   (rho_ini +0.).cpu().numpy()
        self.ini_U_phys            =  [(U_ini[i]+0.).cpu().numpy() for i in range(3)]
        self.ini_body_F_phys       =  [(F_ini[i]+0.).cpu().numpy() for i in range(3)]
        self._nu_lbm               =   params['nu']/self._scale_nu
        self._tau                  =   self._nu_lbm/(self.scheme['cs']**2) + 0.5

        self.A_model, self.guo_Si, self.use_inletoutlet_bc  =  dict(shan_chen   = [self._tau, False, False], 
                                                                 guo         = [   0.5   , True , False], 
                                                                 none        = [   0     , False, False], 
                                                                 pressure_bc = [   0     , False, True ])[params['model_type_force']]
        
        assert all(tuple(arr.shape) == params['n_xyz'] for arr in [*U_ini,rho_ini])
        self.body_F    = [arr/self._scale_body_f for arr in F_ini]
        self.F         = torch.zeros([self.scheme['numel_ci'],*params['n_xyz']],dtype=torch.double,device=device)
        self.F         =  self.get_equilibrum(self.F, [arr/self._scale_u for arr in U_ini], self.scheme, rho_ini/self._scale_rho)
        self.Feq       =  torch.zeros_like(self.F)
        self.get_macroscopic_fields()
        if self.use_inletoutlet_bc :
            self.stencil_inletoutlet_bc          =  dict(ax = self.params['dir_delta_P'])
            self.stencil_inletoutlet_bc['mode']  =  list(params['mode_inlet_outlet_bc'])
            self.stencil_inletoutlet_bc['d']  =  [0, max(1,params['n_xyz'][self.stencil_inletoutlet_bc['ax']]//2) if not ('U' in self.stencil_inletoutlet_bc['mode']) else 
                                                          (params['n_xyz'][self.stencil_inletoutlet_bc['ax']]-1)]

            self.stencil_inletoutlet_bc.update({ii: {vk:eval(vv) for vk,vv in self.individual_pressure_bc(self.scheme, self.stencil_inletoutlet_bc['ax'], ii).items()}  for ii in range(2)})
            # print("self.stencil_inletoutlet_bc = ", {ii: {vk:vv for vk,vv in self.individual_pressure_bc(self.scheme, self.stencil_inletoutlet_bc['ax'], ii).items()}  for ii in range(2)})
            rho_inlet  = torch.ones_like(self.body_F[self.stencil_inletoutlet_bc['ax']])
            # rho_outlet = rho_inlet + (1/(params['n_xyz'][self.stencil_inletoutlet_bc['ax']]-1))*self.body_F[self.stencil_inletoutlet_bc['ax']]*(params['n_xyz'][self.stencil_inletoutlet_bc['ax']] - 1)/(self.scheme['cs']**2)
            rho_outlet = rho_inlet + self.body_F[self.stencil_inletoutlet_bc['ax']]/(self.scheme['cs']**2)*(self.stencil_inletoutlet_bc['d'][1] - self.stencil_inletoutlet_bc['d'][0])
            self.stencil_inletoutlet_bc['all_rho_BC'] = {0: rho_inlet ,  1: rho_outlet}
            self.stencil_inletoutlet_bc['all_rho_BC_scalar'] = {0: as_float(rho_inlet) ,  1: as_float(rho_outlet)}
            self.stencil_inletoutlet_bc['all_U_BC'] = {0: self.U[self.stencil_inletoutlet_bc['ax']]+0.,  
                                                       1: self.U[self.stencil_inletoutlet_bc['ax']]+0.}
            # print(f"{rho_inlet.min():.12e} {rho_inlet.max():.12e} {rho_outlet.min():.12e} {rho_outlet.max():.12e}")
            self.body_F = [torch.zeros_like(arr) for arr in self.body_F]
        self.obstacle = obstacle
        self.num_iters   = 0
        if (not (self.obstacle is None)):
                self.recalculate_BCs_obstacle()

    @staticmethod
    def get_dt_ideal(params, ideal_tau = (math_sqrt(3./16)+0.5)):
        # self._nu_lbm/(self.scheme['cs']**2) = ideal_tau - 0.5
        # nu/scale_nu = (ideal_tau - 0.5) * (cs2)
        # scale_nu  = nu/((ideal_tau - 0.5) * (cs2))
        # dx2/dt = nu/((ideal_tau - 0.5) * (cs2))
        # dt = dx2/(nu/((ideal_tau - 0.5) * (cs2)))
        scheme = LBM_Engine.get_scheme(params['scheme'])
        return (params['size_elem']**2)/(params['nu']/((ideal_tau - 0.5) * (scheme['cs']**2)))

    def get_U_phys(self):
        return [arr*self._scale_u for arr in self.U]
    
    def get_rho_phys(self):
        return self._scale_rho*self.rho

    def get_macroscopic_fields(self):
        self.rho  =  self.F.sum(dim=0)
        get_vel   =  lambda F,rho,ax,scheme: sum(scheme['c_xyz'][ax][i]*F[i,:,:,:] for i in range(scheme['numel_ci']))/rho + self.A_model*self.body_F[ax]/rho
        self.U    =  [get_vel(self.F,self.rho,ax,self.scheme) for ax in range(3)]

    def export_inputs(self):
        folder = self.case_folder
        os_system(f'mkdir -p "{folder}"')
        as_str  = lambda x: ' '.join(lmap(str,x))
        is_per  = [['Walls', 'Periodic'].index(bc) for bc in self.params['BC_type_xyz']]
        if (not (self.obstacle is None)):
            use_obstacle                            =  1
            advancing                               =  int(self.obstacle.advancing)
            bbox, obstacle_U                        =  self.obstacle(0.)
            assert list(bbox.shape) == [3,2]
            bbox                                    =  bbox.tolist()
            obstacle_U                              =  lmap(float, obstacle_U.view(-1).tolist())
            obstacle_ax_active                      =  [int(not any(math_isnan(b) for b in bbox[i])) for i in range(3)]
            obstacle_xyz_ini_0, obstacle_xyz_ini_1  =  [[(float(b[j]) if obstacle_ax_active[i] else -1.) for i,b in enumerate(bbox)] for j in range(2)]
        else:
            use_obstacle  =  advancing                                  =   0
            obstacle_U    =  obstacle_xyz_ini_0  =  obstacle_xyz_ini_1  =  [0.,0.,0.]
            obstacle_ax_active                                          =  [0, 0, 0]
        A_write = [f"{as_str(self.params['n_xyz'])} // nx ny nz",
                   f"{self.total_num_iters} {self.snap_iter_start} {self.snap_freq} // total_num_iters snap_iter_start snap_freq",
                   f"{as_str(is_per)} // is_per_xyz",
                   f"{self._size_elem_phys} {self._dt_phys} {self.params['nu']} {self._scale_rho} // size_elem_phys dt_phys nu_phys scale_rho",
                   f"{self.A_model} {int(self.guo_Si)} {int(self.use_inletoutlet_bc)} // A_model guo_Si use_inletoutlet_bc",
                   f"{self.scheme['cs']} // cs",
                   f"{int(use_obstacle)} {int(advancing)} // use_obstacle advancing",
                   f"{as_str(obstacle_U)} // obstacle_U[0:2]",
                   f"{as_str(obstacle_xyz_ini_0)} // obstacle_xyz_ini_0[0:2]",
                   f"{as_str(obstacle_xyz_ini_1)} // obstacle_xyz_ini_1[0:2]",
                   f"{as_str(obstacle_ax_active)} // obstacle_ax_active[0:2]",
                   f"{self.scheme['numel_ci']} // numel_ci (next lines are: c_xyz[:][i] wi[i])",
                   *lmap(as_str,list(zip(*self.scheme['c_xyz'],self.scheme['wi'])))]
        for ax in range(3):
            shape = get_shape3(self.pairs_xyz[ax])
            A_write.append(f"{shape[0]} {shape[1]} {shape[2]} // shape(pairs_xyz[{ax}])")
            A_write.extend([str(self.pairs_xyz[ax][i][j][k]) for i in range(shape[0]) for j in range(shape[1]) for k in range(shape[2])])
        
        if self.use_inletoutlet_bc:
            m2i  =  dict(P=1, U=0)
            A_write.extend([f"{int(self.stencil_inletoutlet_bc['ax'])} // pressure_ax",
                            f"{self.stencil_inletoutlet_bc['d'][0]} {self.stencil_inletoutlet_bc['d'][1]} // stencil_inletoutlet_bc_inds[0,1]",
                            f"{m2i[self.stencil_inletoutlet_bc['mode'][0]]} {m2i[self.stencil_inletoutlet_bc['mode'][1]]} // stencil_inletoutlet_modes[0,1] (1=P, 0=U)",
                            f"{self.stencil_inletoutlet_bc['all_rho_BC_scalar'][0]} {self.stencil_inletoutlet_bc['all_rho_BC_scalar'][1]} // rho_inlet rho_outlet"])
        writer(pjoin(folder,'params.dat'), A_write)
        writer_bin_arr(pjoin(folder,"array_ini_rho.dat"    ), self.ini_rho_phys)
        writer_bin_arr(pjoin(folder,"array_ini_U.dat"      ), self.ini_U_phys[0])
        writer_bin_arr(pjoin(folder,"array_ini_V.dat"      ), self.ini_U_phys[1])
        writer_bin_arr(pjoin(folder,"array_ini_W.dat"      ), self.ini_U_phys[2])
        writer_bin_arr(pjoin(folder,"array_ini_body_Fx.dat"), self.ini_body_F_phys[0])
        writer_bin_arr(pjoin(folder,"array_ini_body_Fy.dat"), self.ini_body_F_phys[1])
        writer_bin_arr(pjoin(folder,"array_ini_body_Fz.dat"), self.ini_body_F_phys[2])

    def export_outputs(self):
        U_phys    =  self.get_U_phys()
        rho_phys  =  self.get_rho_phys()
        writer_bin_arr(pjoin(self.case_folder,"array_end_rho_phys.dat"    ), rho_phys.cpu().numpy())
        writer_bin_arr(pjoin(self.case_folder,"array_end_U_phys.dat"      ), U_phys[0].cpu().numpy())
        writer_bin_arr(pjoin(self.case_folder,"array_end_V_phys.dat"      ), U_phys[1].cpu().numpy())
        writer_bin_arr(pjoin(self.case_folder,"array_end_W_phys.dat"      ), U_phys[2].cpu().numpy())
        writer_bin_arr(pjoin(self.case_folder,"array_end_F.dat"           ), (self.F + 0.).cpu().numpy())

    def export_snap(self):
        U_phys    =  self.get_U_phys()
        rho_phys  =  self.get_rho_phys()
        writer_bin_arr(pjoin(self.case_folder,f"array_iter_{self.num_iters:09d}_rho.dat"    ), rho_phys.cpu().numpy())
        writer_bin_arr(pjoin(self.case_folder,f"array_iter_{self.num_iters:09d}_U.dat"      ), U_phys[0].cpu().numpy())
        writer_bin_arr(pjoin(self.case_folder,f"array_iter_{self.num_iters:09d}_V.dat"      ), U_phys[1].cpu().numpy())
        writer_bin_arr(pjoin(self.case_folder,f"array_iter_{self.num_iters:09d}_W.dat"      ), U_phys[2].cpu().numpy())

    def iterate(self, total_num_iters, snap_iter_start, snap_freq, case_folder, f_max_error = None,only_export_inputs=False):
        self.total_num_iters  =  total_num_iters
        self.case_folder      =  case_folder
        self.snap_freq        =  snap_freq
        self.snap_iter_start  =  snap_iter_start
        self.export_inputs()
        if only_export_inputs: return

        for _ in range(total_num_iters):
            self.num_iters += 1

            # -> Get macroscopic fields & equilibrum function
            self.Feq                 = self.get_equilibrum(self.Feq, self.U, self.scheme, self.rho       )
            if self.guo_Si: self.Feq = self.get_equilibrum(self.Feq, self.U, self.scheme, self._tau-0.5, self.body_F)

            # -> Collision operator
            self.F += (self.Feq - self.F) * (1./self._tau)

            # -> Streaming
            F0     = self.F + 0.
            self.F = self.streaming(self.F,self.scheme)
            # -> Apply BCs
            self.apply_BCs(self.F, F0, self.params['BC_type_xyz'], self.pairs_xyz)

            if (not (self.obstacle is None)): self.apply_BCs_obstacle(self.F, F0, self.U_obstacle, self.rho, self.BCs_obstacle, self.scheme, self.pairs_xyz, self.params['n_xyz'])

            if self.use_inletoutlet_bc: self.apply_inletoutlet_bc(self.F, self.U, self.body_F, self.A_model, self.stencil_inletoutlet_bc)

            self.get_macroscopic_fields()

            if (not (self.obstacle is None)) and self.obstacle.advancing: self.recalculate_BCs_obstacle()
            
            if ((self.num_iters>=self.snap_iter_start) and (self.num_iters%self.snap_freq == 0)) or (total_num_iters==self.num_iters):
                self.export_snap()

            if not (f_max_error is None):
                f_max_error(self.num_iters)
        self.export_outputs()

    def recalculate_BCs_obstacle(self):
        bbox, U_box  =  self.obstacle(self.num_iters*self._dt_phys)
        def mk_inds_bbox(bbox):
            result     =  [[None,None] for _ in range(3)]
            ax_active  =  [True for _ in range(3)]
            for i,(row,n) in enumerate(zip(bbox, self._n_xyz)):
                for j,x in enumerate(row):
                    if math_isnan(x):
                        result[i][j] = [j*(n-1),j*(n-1)]
                        ax_active[i] = False
                    else:
                        result[i][j] = [(int_ceil(x/self._size_elem_phys) + k)%n for k in [-1, 0]]
                # assert all(0<=val<n for A in result[i] for val in A)
            return result, ax_active
        inds_bbox, ax_active  =  mk_inds_bbox(bbox)
        assert all((type(inds_bbox[i][j][k]) == int) for i in range(len(inds_bbox)) for j in range(len(inds_bbox[i])) for k in range(len(inds_bbox[i][j])))
        ijk_defaults          =  [[row[0][1],row[1][0]] for row in inds_bbox]
        self.BCs_obstacle = []
        for     ax, row  in  enumerate(inds_bbox):
            if ax_active[ax]:
                for sides    in  row:
                    for ii,p in enumerate(sides):
                        ijk = deepcopy(ijk_defaults)
                        ijk[ax] = [p,p]
                        self.BCs_obstacle.append([ax, ii, *[y for x in ijk for y in x]])
        self.U_obstacle        = lmap(float,(U_box.view(-1)/self._scale_u).tolist())
        self.obstacle_ijk_bbox = ijk_defaults
        
    @staticmethod
    def get_equilibrum(Feq,U,scheme,rho,Ua_special=None):
        Ua       = U if (Ua_special is None) else Ua_special
        cs2      = scheme['cs']**2
        for i in range(scheme['numel_ci']):
            val_ini     =  1 if (Ua_special is None) else 0
            ca          =  lambda     ax: scheme['c_xyz'][ax][i]
            loc_factor  =  lambda ax,ax2: ((ca(ax)*scheme['c_xyz'][ax2][i] - cs2*int(ax==ax2))/(2*(cs2**2)))
            Feq[i]      =  (val_ini + sum( (Ua[ax]*(ca(ax)/cs2 + sum((U[ax2]*loc_factor(ax,ax2)) for ax2 in range(3)))) for ax in range(3) )
                           )*scheme['wi'][i]*rho + (1-val_ini)*Feq[i]
        return Feq
    
    @staticmethod
    def streaming(F,scheme):
        for i in range(scheme['numel_ci']):
            for ax in range(3):
                c    = scheme['c_xyz'][ax][i]
                F[i] = torch.roll(F[i],c,ax)
        return F
    
    @staticmethod
    def apply_BCs(F, F0, BC_type_xyz, pairs_xyz):
        for ax in range(3):
            if BC_type_xyz[ax] == 'Walls':
                for ii,d in enumerate([0,-1]):
                    for a,r in pairs_xyz[ax][ii]:
                        if   ax == 0: F[r,d,:,:] = F0[a,d,:,:]
                        elif ax == 1: F[r,:,d,:] = F0[a,:,d,:]
                        elif ax == 2: F[r,:,:,d] = F0[a,:,:,d]
                        else:         raise Exception(f'ERROR: Axis unrecognized ({ax = })')
            else:
                assert BC_type_xyz[ax] == 'Periodic'
    
    @staticmethod
    def apply_BCs_obstacle(F, F0, Uw, rho, BCs_obstacle, scheme, pairs_xyz, n_xyz):
        # 2*wi*rho*uw*ci/(cs**2)
        use_moving = any((abs(vel) > 1e-10) for vel in Uw)
        if use_moving:
            ci_uw  =  [sum((scheme['c_xyz'][ax][i]*Uw[ax]) for ax in range(3)) for i in range(scheme['numel_ci'])]
        for ax,ii,i0_,i1_,j0,j1,k0,k1 in BCs_obstacle:
            for i0,i1 in ([(i0_,i1_)] if i0_<=i1_ else [(0  , i1_),
                                                        (i0_, n_xyz[0]-1)]):
                for a,r in pairs_xyz[ax][ii]:
                    F[r,i0:i1+1,j0:j1+1,k0:k1+1]      = F0[a,i0:i1+1,j0:j1+1,k0:k1+1]
                    if use_moving:
                        F[r,i0:i1+1,j0:j1+1,k0:k1+1] += ((2*scheme['wi'][r]*ci_uw[r]/(scheme['cs']**2))*
                                                          rho[i0:i1+1,j0:j1+1,k0:k1+1])

    @staticmethod
    def get_scheme(scheme):
        assert scheme == 'D3Q19'
        return dict(c_xyz    = ((0, 1, -1, 0,  0, 0,  0, 1, -1, 1, -1, 0,  0,  1, -1,  1, -1,  0,  0),
                                (0, 0,  0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1,  0,  0,  1, -1),
                                (0, 0,  0, 0,  0, 1, -1, 0,  0, 1, -1, 1, -1,  0,  0, -1,  1, -1,  1)) ,
                    ci_len   = tuple([0.  ] + [1.   ]*6 + [math_sqrt(2)]*12)                           ,
                    wi       = tuple([1/3.] + [1/18.]*6 + [1/36.       ]*12)                           ,
                    cs       = 1/math_sqrt(3.)                                                         ,
                    numel_ci = 19 ) # Table 3.5
    @staticmethod
    def get_pairs_xyz(scheme):
        key2i   =  {tuple([scheme['c_xyz'][j][i] for j in range(3)]):i for i in range(scheme['numel_ci'])}
        get_rev =  lambda p: key2i[tuple([-scheme['c_xyz'][j][p] for j in range(3)])]
        pairs   =  []
        for ax in range(3):
            get_points = lambda p: [i for i in range(scheme['numel_ci']) if scheme['c_xyz'][ax][i]==p]
            pairs.append([[[p,get_rev(p)] for p in get_points(-1)],  # pairs_down & rev
                          [[p,get_rev(p)] for p in get_points( 1)]]) # pairs_up   & rev
        assert all(scheme['c_xyz'][j][p0] == (-scheme['c_xyz'][j][p1]) for row in pairs for p01 in row for p0,p1 in p01 for j in range(3))
        return pairs

    @staticmethod
    def apply_inletoutlet_bc(F, U, body_F, A_model, stencil_bc):
        ax = stencil_bc['ax']
        # assert ax == 0
        for ii in range(2):
            d    =  stencil_bc['d'][ii]
            mode =  stencil_bc['mode'][ii]
            rho_bc, U_ax = None, None
            if mode == 'P':
                rho_bc  =  pick_d(stencil_bc['all_rho_BC'][ii], ax, d)
                U_ax    =  stencil_bc[ii]['U_ax'](F, d, U, body_F, rho_bc, A_model, U_ax)
            elif mode == 'U':
                U_ax    =  pick_d(stencil_bc['all_U_BC'][ii], ax, d)
                rho_bc  =  stencil_bc[ii]['rho_calc'](F, d, U, body_F, rho_bc, A_model, U_ax)
            else: 1/0
            for k,ff in stencil_bc[ii].items():
                if not (k in ['U_ax','rho_calc']):
                    if   ax == 0:  F[k,d,:,:] = ff(F, d, U, body_F, rho_bc, A_model, U_ax)
                    elif ax == 1:  F[k,:,d,:] = ff(F, d, U, body_F, rho_bc, A_model, U_ax)
                    elif ax == 2:  F[k,:,:,d] = ff(F, d, U, body_F, rho_bc, A_model, U_ax)
                    else        :  1/0
    @staticmethod
    def individual_pressure_bc(scheme, ax, ii):
        pairs      =  LBM_Engine.get_pairs_xyz(scheme)
        # Note:
        #     - Recomputing the pressure BC is not necessary, unless the scheme (D3Q19) or the index-definition is changed (ci[0], ... ci[18], etc.)
        #     - To run the code below, "Sympy" is required.
        #           - Not all Python environments have this package installed (especially in servers).
        #           - This is the main motivation to avoid recomputing the pressure BC mandatorily. (The running times are relatively fast.)
        if RECOMPUTE_PRESSURE_BC:
            import sympy as sp
            def get_equilibrum(U,scheme,rho,Ua_special=None):
                Ua       = U if (Ua_special is None) else Ua_special
                cs2      = scheme['cs']**2
                Feq      = [(1 if (Ua_special is None) else 0) for _ in range(scheme['numel_ci'])]
                for i in range(scheme['numel_ci']):
                    for ax in range(3):
                        ca      =  scheme['c_xyz'][ax][i]
                        Feq[i] +=  (ca/cs2)*Ua[ax]
                        for ax2 in range(3):
                            cb      = scheme['c_xyz'][ax2][i]
                            Feq[i] += Ua[ax]*U[ax2]*((ca*cb - cs2*int(ax==ax2))/(2*(cs2**2))) # Eq. (3.54) or (6.5)
                    Feq[i] *= scheme['wi'][i]*rho
                return Feq
            assert (ax in [0,1,2]) and (ii in [0,1])
            other_ax  =  [j for j in range(3) if (j!=ax)]
            def get_center_crosses():
                get_cen_points = lambda ax,p: [i for i in range(scheme['numel_ci']) if scheme['c_xyz'][ax][i]==p and 
                                               all(scheme['c_xyz'][j][i]==0 for j in other_ax)]
                base                   =  pairs[ax][ii]
                i_center,              =  get_cen_points(ax,[-1,1][ii])
                center,                =  [row for row in base if (row[0]==i_center)]
                L0                     =  len(base)
                base                   =  [row for row in base if (row!=center)]
                crosses = [lfilter(lambda row: [abs(scheme['c_xyz'][j][row[0]]),scheme['c_xyz'][3-j-ax][row[0]]]==[1,0], base) for j in other_ax]
                assert all(len(rows) == 2 for rows in crosses)
                assert (sum(map(len,crosses))+1) == L0 == (len(base)+1)
                return center,crosses
            marker_k                     =  '_markerk_'
            marker_i                     =  '_markeri_'
            center, crosses        =  get_center_crosses()
            all_F                  =  [sp.symbols(f"F_{marker_k}{i}{marker_k}_",nonzero=True,real=True,positive=True) for i in range(scheme['numel_ci'])]
            A_model,rho_BC,K_a,K_b =  sp.symbols("A_model,rho_BC,K_a,K_b",nonzero=True,real=True,positive=True)
            body_F                 =  [sp.symbols(f"body_{marker_i}{i}{marker_i}_",nonzero=True,real=True,positive=True) for i in range(3)]
            get_vel                =  lambda all_F,rho,ax,scheme: sum(scheme['c_xyz'][ax][i]*all_F[i] for i in range(scheme['numel_ci']))/rho + A_model*body_F[ax]/rho
            U_eq                   =  [get_vel(all_F,rho_BC,ax,scheme) for ax in range(3)]
            U                      =  [sp.symbols(f"U_{marker_i}{i}{marker_i}_",nonzero=True,real=True,positive=True) for i in range(3)]
            all_Feq                =  get_equilibrum(U,scheme,rho_BC,Ua_special=None)
            # rho                    =  sum(all_F)
            all_F_noneq            =  [(a-b) for a,b in zip(all_F,all_Feq)]
            eqs                    =  [] # [rho - rho_BC]
            eqs.extend([U_eq[j]-U[j] for j in other_ax])
            eqs.append(all_F_noneq[center[0]]-all_F_noneq[center[1]])
            get_corr = lambda i: sum(scheme['c_xyz'][j][i]*KK for j,KK in zip(other_ax, [K_a, K_b]))
            eqs.extend([(all_F_noneq[p[0]]-all_F_noneq[p[1]]+get_corr(p[0])) for rows in crosses for p in rows])
            vars_solve  =  [K_a, K_b, all_F[center[0]],*[all_F[p[0]] for rows in crosses for p in rows]]
            assert len(vars_solve) == len(eqs)
            eqs = [e.expand().simplify().expand().simplify() for e in eqs]
            sol = sp.solve(eqs,*vars_solve)
            for j in other_ax: sol = {k:v.subs(U[j],0) for k,v in sol.items()}
            del j
            # U_ax = U_eq[ax]
            # for k,v in sol.items(): U_ax = U_ax.subs(k,v)
            # del k,v
            U_ax = sp.symbols("U_ax",nonzero=True,real=True,positive=True) 
            sol = {k:v.subs(U[ax],U_ax).expand().simplify().expand().simplify() for k,v in sol.items()}
            key2int = lambda k:  int(str(k).removeprefix(f'F_{marker_k}').removesuffix(f'{marker_k}_'))
            def as_func(x):
                my_indexes = ','.join(('d' if j==ax else ':') for j in range(3))
                return 'lambda F,d,U,body,rho_BC,A_model,U_ax: ' + str(x).replace(f'_{marker_k}','[').replace(f'{marker_k}_',  f',{my_indexes}]')\
                                                                         .replace(f'_{marker_i}','[').replace(f'{marker_i}_', f'][{my_indexes}]')
            sol = {key2int(k): as_func(sol[k]) for k in vars_solve[2:]}
            K_dir = [-1,1][ii]
            U_ax_formula  =  (U_eq[ax] + K_dir*(1 - sum(all_F)/rho_BC)).simplify()
            sol['U_ax']   =  as_func(U_ax_formula) 
            rho_formula,  =  sp.solve(U_ax_formula - U_ax, rho_BC)
            sol['rho_calc'] = as_func(rho_formula.simplify()) 
        return sol

# ------------------------------------------------------------------------
#                  Obstacle Box
# ------------------------------------------------------------------------

class Obstacle_Box:
    def __init__(self, params):
        self.bbox       =  torch.tensor([lmap(float,row        ) for row in params['bbox']] , dtype=torch.double,device=device)
        self.U_box      =  torch.tensor( lmap(float,params['U'])                            , dtype=torch.double,device=device).reshape(-1,1)
        self.advancing  =  int(bool(params['advancing']))
        assert self.U_box.shape == (3,1)
        assert self.bbox .shape == (3,2)
        
    def __call__(self, t):
        return [(self.bbox + t*self.U_box*self.advancing), (self.U_box+0.)]
    
# ------------------------------------------------------------------------
#                  Print Pressure BCs
# ------------------------------------------------------------------------

def print_pressure_BCs():
    scheme = LBM_Engine.get_scheme('D3Q19')
    line = '-'*30

    dconv = {}
    for ax2 in range(3):
      for ax in range(19):
        ijk  = ':,:,:'.split(',')
        ijk2 = 'i,j,k'.split(',')
        ijk [ax2] = 'd'
        ijk2[ax2] = 'd'
        ijk  = ','.join(ijk)
        ijk2 = ','.join(ijk2)
        if ax < 3:
            dconv[f"[{ax}][{ijk}]"] = f"[ind_ii_ijk({ax},{ijk2},ni,nj,nk)]"
        dconv[f"[{ax},{ijk}]"]      = f"[ind_ii_ijk({ax},{ijk2},ni,nj,nk)]"
      dconv[f"rho_BC[{ijk}]"]      = f"rho_BC[?]"

    for ax in range(3): 
      for ii in range(2): 
        print(f"// {line} (ax = {ax}, ii = {ii}) {line}")
        for i,expr in sorted(LBM_Engine.individual_pressure_bc(scheme,ax,ii).items(),key=lambda x: [0 if x[0] in ['U_ax','rho_calc'] else 1,x]):
          if not (i in ['U_ax','rho_calc']):
            myF = [':', ':', ':']
            myF[ax] = 'd'
            myF = 'F[' + str(i) + ',' + ','.join(myF) + ']'
          else:
            myF = f'double {i}'
          expr = expr[[i for i,c in enumerate(expr) if (c==':')][0]+1:]
          result = f"{myF} = {expr};"
          for k,v in dconv.items(): result = result.replace(k,v)
          result = result.replace('rho_BC[?]',['rho_inlet','rho_outlet'][ii])
          print(result)
        print('')

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    print_pressure_BCs()