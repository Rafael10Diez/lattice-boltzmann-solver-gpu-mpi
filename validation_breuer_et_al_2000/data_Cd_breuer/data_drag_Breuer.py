# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os.path            import  basename, dirname, abspath, isfile, isdir
from    os.path            import  join                                       as  pjoin
from    os                 import  listdir                                    as  os_listdir
from    PIL                import  Image                                      as  PIL_Image
import  matplotlib.pyplot as  plt
import  numpy             as  np
from    scipy.interpolate import interp1d

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

_folder_              =  dirname(abspath(__file__))

lmap                  =  lambda f,x: list(map(f,x))
lfilter               =  lambda f,x: list(filter(f,x))
listdir_full          =  lambda   x: [pjoin(x,y) for y in os_listdir(x)]
listdir_full_files    =  lambda   x: lfilter(isfile, listdir_full(x))
listdir_full_folders  =  lambda   x: lfilter(isdir , listdir_full(x))

tolist                =  lambda   x: x.tolist() if hasattr(x,'tolist') else x

get_uni               =  lambda x: sorted(set(list(x)))

def reader(fname):
    with open(fname,'r') as f:
        return [x.rstrip('\n') for x in f]

def writer(fname, A):
    with open(fname,'w') as f:
        for x in A:
            f.write(x.rstrip('\n')+'\n')

def pop1(A):
    x, = A 
    return x

# ------------------------------------------------------------------------
#                  Utilities
# ------------------------------------------------------------------------

def find_clusters(A0,key,cempty):
    A   = A0.copy()
    key = np.array(key, dtype = A.dtype)
    assert (key>=0).all() and (A>=0).all()
    def dfs(A,i,j,new):
        if (A[i,j]==key).all():
            new.append([i,j])
            A[i,j] = cempty
            for     di in [-1,0,1]:
                for dj in [-1,0,1]:
                    if (abs(di)+abs(dj))==1:
                        dfs(A,i+di,j+dj,new)
    result = []
    for     i in range(A.shape[0]):
        for j in range(A.shape[1]):
            new = []
            dfs(A,i,j,new)
            if new:
                result.append(new)
    return result

def match_anchors(clusters):
    xy_clusters =  [np.mean(cc,axis=0) for cc in clusters]
    _,_,x0,x1   =  sorted(xy_clusters, key=lambda x: x[0])
    _,_,y0,y1   =  sorted(xy_clusters, key=lambda x: x[1])
    return dict(x0 = x0[0], 
                x1 = x1[0],
                y0 = y0[1], 
                y1 = y1[1])

class Convert_Axes:
    def __init__(self, c_type, phys_coords, img_coords):
        self.is_log       =  phys_coords[f"log_{c_type}"]
        if self.is_log: f = np.log10 
        else:           f = lambda x: x
        self.c0_img, self.c1_img  =  [   img_coords[f"{c_type}{i}"]  for i in range(2)]
        self.c0_phy, self.c1_phy  =  [f(phys_coords[f"{c_type}{i}"]) for i in range(2)]
    def __call__(self, c_img):
        c_phy = (c_img - self.c0_img)/(self.c1_img - self.c0_img)*(self.c1_phy - self.c0_phy) + self.c0_phy
        if self.is_log: c_phy = 10**c_phy
        return c_phy

def avg_repeated(A):
    x1      =  get_uni([x for x,_ in A]) # unique x-coords
    result  =  {k:[] for k in x1}        # empty list for each unique x-coords
    for x,y in A: result[x].append(y)
    return [[k, float(np.mean(result[k]))] for k in sorted(result.keys())]

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

def get_curve_drag_Breuer(do_plot=False):
    fname_pic    =  pjoin(_folder_,'data_drag_Breuer.png')
    A            =  np.asarray(PIL_Image.open(fname_pic))[:,:,:3].transpose(1,0,2)
    A            =  A[:,::-1,:] 
    if do_plot:
        plt.pcolormesh(list(range(A.shape[0]+1)),list(range(A.shape[1]+1)),A[:,:,0].transpose())
        plt.show()
    rgb_axes     =  ( 34 , 177 , 76)
    rgb_line     =  (255 , 201 , 14)
    cempty       = 255
    phys_coords  =  dict(x0    =  1 , 
                         x1    = 50 ,
                         y0    =  2 ,
                         y1    = 50 ,
                         log_x = True,
                         log_y = True)
    img_coords   = match_anchors(find_clusters(A,rgb_axes,cempty))
    data_anchors =               find_clusters(A,rgb_line,cempty)

    fx,fy    =  [Convert_Axes(c, phys_coords, img_coords) for c in 'xy']
    data_xy  =  avg_repeated(sorted([list(xy) for all_xy in data_anchors for xy in all_xy]))
    data_xy  =  np.array([[fx(x),fy(y)] for x,y in data_xy])
    f_Re_Cd_Brueuer = interp1d(data_xy[:,0], data_xy[:,1])
    if do_plot:
        plt.plot(data_xy[:,0],data_xy[:,1],lw=2)
        xx_Re = get_uni(np.linspace(min(data_xy[:,0]),
                                    max(data_xy[:,0]),
                                    max(1_000,data_xy.shape[0])).tolist() + data_xy[:,0].tolist())
        plt.plot(xx_Re,f_Re_Cd_Brueuer(xx_Re),':',lw=2)
        plt.show()
    return f_Re_Cd_Brueuer

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------
if __name__ == '__main__':
    f_Re_Cd_Brueuer = get_curve_drag_Breuer(do_plot=True)
    