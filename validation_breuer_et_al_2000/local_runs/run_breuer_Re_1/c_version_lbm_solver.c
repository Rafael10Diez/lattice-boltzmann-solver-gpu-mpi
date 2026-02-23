#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <errno.h>
#include <string.h>

#define ind_ii_ijk(ii,i,j,k,ni,nj,nk) (ii*((ni+2)*(nj+2)*(nk+2)) + (i+1)*(nj+2)*(nk+2) + (j+1)*(nk+2) + k+1)
#define loc_factor_Feq(ca,ax,ax2)     ((ca*((double) lbm_obj->c_xyz[ax2*numel_ci + ii]) - ((double) ((ax==ax2)? (cs2):0.)))/(2*(pow(cs2,2))))
#define malloc_layers(num_layers)     ((double*) malloc(((lbm_obj.nx+2)*(lbm_obj.ny+2)*(lbm_obj.nz+2)*num_layers) * sizeof(double)))
#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

typedef struct{
	int     irank_mpi, nproc_mpi, divs_mpi_x, divs_mpi_y, divs_mpi_z, nx, ny, nz, nx_global, ny_global, nz_global, \
            num_iters, use_pressure_bc, pressure_ax, guo_Si, pos_mpi_x, pos_mpi_y, pos_mpi_z, i_global, j_global, k_global, *mpi_ranks, \
            is_per[3], numel_ci, *c_xyz, *pairs_x, *pairs_y, *pairs_z, pairs_x_shape[3], pairs_y_shape[3], pairs_z_shape[3], is_boundary[6], \
            pressure_bc_inds[2], pressure_bc_mode[2], size_buf_mpi, use_obstacle, obstacle_advancing, iters_full, obstacle_ax_active[3], \
            obstacle_inds_bbox[12], obstacle_ijk_defaults[6], obstacle_is_moving, rank_south[3], rank_north[3], snap_iter_start, snap_freq;
    double  size_elem_phys, dt_phys, nu_phys, scale_rho, scale_u, scale_nu, scale_body_F, nu_lbm, tau, A_model, cs, *wi, \
            pressure_bc_rho[2], obstacle_U[3], obstacle_ini_c0[3], obstacle_ini_c1[3];
} type_lbm_obj;

int int_ceil(double x){
    int y = x;
    if ((fabs(((double) y) - x)>1e-10)&&(x>y)){ y = y +1; }
    if ((fabs(((double) y) - x)>1e-10)&&(x>y)){ printf("Error int_ceil\n"); exit(EXIT_FAILURE); }
    return y;
}

void pack_halo(double* F, double* buffer_mpi, int ni, int nj, int nk, int sz_pack, int ax, int mode_fwd, \
               int* halo_m, int* halo_p, int numel_halo){
    if (ax == 0){
      if (mode_fwd ==1){
        #pragma omp target teams distribute parallel for  collapse(3)
          for     (int ii  = 0; ii < numel_halo ; ii++) {
            for   (int j  = -1; j < nj+1 ; j++) {
              for (int k  = -1; k < nk+1 ; k++) {
                buffer_mpi[ind_ii_ijk(ii,0,j,k,-1,nj,nk)        ]  = F[ind_ii_ijk(halo_m[ii],0   ,j,k,ni,nj,nk)];
                buffer_mpi[ind_ii_ijk(ii,0,j,k,-1,nj,nk)+sz_pack]  = F[ind_ii_ijk(halo_p[ii],ni-1,j,k,ni,nj,nk)];
              }}}}
      else {
        #pragma omp target teams distribute parallel for  collapse(3)
          for     (int ii  = 0; ii < numel_halo ; ii++) {
            for   (int j  = -1; j < nj+1 ; j++) {
              for (int k  = -1; k < nk+1 ; k++) {
                F[ind_ii_ijk(halo_p[ii],-1,j,k,ni,nj,nk)] = buffer_mpi[ind_ii_ijk(ii,0,j,k,-1,nj,nk)+2*sz_pack];
                F[ind_ii_ijk(halo_m[ii],ni,j,k,ni,nj,nk)] = buffer_mpi[ind_ii_ijk(ii,0,j,k,-1,nj,nk)+3*sz_pack];
              }}}}
    }
    else if (ax == 1){
      if (mode_fwd ==1){
        #pragma omp target teams distribute parallel for  collapse(3)
          for     (int ii  = 0; ii < numel_halo ; ii++) {
            for   (int i  = -1; i < ni+1 ; i++) {
              for (int k  = -1; k < nk+1 ; k++) {
                buffer_mpi[ind_ii_ijk(ii,i,0,k,ni,-1,nk)        ]  = F[ind_ii_ijk(halo_m[ii],i,0   ,k,ni,nj,nk)];
                buffer_mpi[ind_ii_ijk(ii,i,0,k,ni,-1,nk)+sz_pack]  = F[ind_ii_ijk(halo_p[ii],i,nj-1,k,ni,nj,nk)];
              }}}}
      else {
        #pragma omp target teams distribute parallel for  collapse(3)
          for     (int ii  = 0; ii < numel_halo ; ii++) {
            for   (int i  = -1; i < ni+1 ; i++) {
              for (int k  = -1; k < nk+1 ; k++) {
                F[ind_ii_ijk(halo_p[ii],i,-1,k,ni,nj,nk)] = buffer_mpi[ind_ii_ijk(ii,i,0,k,ni,-1,nk)+2*sz_pack];
                F[ind_ii_ijk(halo_m[ii],i,nj,k,ni,nj,nk)] = buffer_mpi[ind_ii_ijk(ii,i,0,k,ni,-1,nk)+3*sz_pack];
              }}}}
    }
    else if (ax == 2){
      if (mode_fwd ==1){
        #pragma omp target teams distribute parallel for  collapse(3)
          for     (int ii  = 0; ii < numel_halo ; ii++) {
            for   (int i  = -1; i < ni+1 ; i++) {
              for (int j  = -1; j < nj+1 ; j++) {
                buffer_mpi[ind_ii_ijk(ii,i,j,0,ni,nj,-1)        ]  = F[ind_ii_ijk(halo_m[ii],i,j,0   ,ni,nj,nk)];
                buffer_mpi[ind_ii_ijk(ii,i,j,0,ni,nj,-1)+sz_pack]  = F[ind_ii_ijk(halo_p[ii],i,j,nk-1,ni,nj,nk)];
              }}}}
      else {
        #pragma omp target teams distribute parallel for  collapse(3)
          for     (int ii  = 0; ii < numel_halo ; ii++) {
            for   (int i  = -1; i < ni+1 ; i++) {
              for (int j  = -1; j < nj+1 ; j++) {
                F[ind_ii_ijk(halo_p[ii],i,j,-1,ni,nj,nk)] = buffer_mpi[ind_ii_ijk(ii,i,j,0,ni,nj,-1)+2*sz_pack];
                F[ind_ii_ijk(halo_m[ii],i,j,nk,ni,nj,nk)] = buffer_mpi[ind_ii_ijk(ii,i,j,0,ni,nj,-1)+3*sz_pack];
              }}}}
    }
}

void copy_1d_buffer(double* buffer_mpi, int a, int b, int s){
  #pragma omp target teams distribute parallel for  collapse(1)
    for (int i  = 0; i < s ; i++) {
        buffer_mpi[i+b] = buffer_mpi[i+a];}
}

void mpi_buffer_sync(double* buffer_mpi, int sz_pack, int irank_mpi, int rank_south, int rank_north){
    // [0], [n-1], [-1], [n]
    if ((rank_south == irank_mpi)||(rank_north == irank_mpi)){
        if ((rank_south != irank_mpi)||(rank_north != irank_mpi)){printf("ERROR: rank alignment\n"); exit(EXIT_FAILURE);}
        copy_1d_buffer(buffer_mpi,       0, 3*sz_pack, sz_pack);
        copy_1d_buffer(buffer_mpi, sz_pack, 2*sz_pack, sz_pack);}
    else {
        MPI_Request requests[4];
        #pragma omp target data use_device_ptr(buffer_mpi)
        {
          MPI_Irecv(&(buffer_mpi[3*sz_pack]), sz_pack, MPI_DOUBLE, rank_north, 1, MPI_COMM_WORLD, &requests[0]);
          MPI_Irecv(&(buffer_mpi[2*sz_pack]), sz_pack, MPI_DOUBLE, rank_south, 2, MPI_COMM_WORLD, &requests[1]);
          MPI_Isend(&(buffer_mpi[0]        ), sz_pack, MPI_DOUBLE, rank_south, 1, MPI_COMM_WORLD, &requests[2]);
          MPI_Isend(&(buffer_mpi[  sz_pack]), sz_pack, MPI_DOUBLE, rank_north, 2, MPI_COMM_WORLD, &requests[3]);
          MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        }}
}

void update_halos(double* F, double* buffer_mpi, type_lbm_obj* lbm_obj,\
                  int numel_halo_x ,  int numel_halo_y ,  int numel_halo_z ,\
                  int*      halo_xm,  int*      halo_xp,\
                  int*      halo_ym,  int*      halo_yp,\
                  int*      halo_zm,  int*      halo_zp){
  int ni = lbm_obj->nx; int nj = lbm_obj->ny; int nk = lbm_obj->nz;
  int sz_pack, ax;

  ax = 0; sz_pack =        (nj+2)*(nk+2)*numel_halo_x;
  pack_halo(F, buffer_mpi, ni, nj, nk, sz_pack, ax, 1, halo_xm, halo_xp, numel_halo_x);
  mpi_buffer_sync(buffer_mpi, sz_pack, lbm_obj->irank_mpi, lbm_obj->rank_south[ax], lbm_obj->rank_north[ax]);
  pack_halo(F, buffer_mpi, ni, nj, nk, sz_pack, ax, 0, halo_xm, halo_xp, numel_halo_x);

  ax = 1; sz_pack = (ni+2)*       (nk+2)*numel_halo_y;
  pack_halo(F, buffer_mpi, ni, nj, nk, sz_pack, ax, 1, halo_ym, halo_yp, numel_halo_y);
  mpi_buffer_sync(buffer_mpi, sz_pack, lbm_obj->irank_mpi, lbm_obj->rank_south[ax], lbm_obj->rank_north[ax]);
  pack_halo(F, buffer_mpi, ni, nj, nk, sz_pack, ax, 0, halo_ym, halo_yp, numel_halo_y);

  ax = 2; sz_pack = (ni+2)*(nj+2)       *numel_halo_z;
  pack_halo(F, buffer_mpi, ni, nj, nk, sz_pack, ax, 1, halo_zm, halo_zp, numel_halo_z);
  mpi_buffer_sync(buffer_mpi, sz_pack, lbm_obj->irank_mpi, lbm_obj->rank_south[ax], lbm_obj->rank_north[ax]);
  pack_halo(F, buffer_mpi, ni, nj, nk, sz_pack, ax, 0, halo_zm, halo_zp, numel_halo_z);
}

void streaming(double* F, double* F2, type_lbm_obj* lbm_obj){
  int ni = lbm_obj->nx; int nj = lbm_obj->ny; int nk = lbm_obj->nz;
    for      (int ii  = 0; ii < lbm_obj->numel_ci ; ii++) {
    int di = lbm_obj->c_xyz[                      ii];
    int dj = lbm_obj->c_xyz[  lbm_obj->numel_ci + ii];
    int dk = lbm_obj->c_xyz[2*lbm_obj->numel_ci + ii];
    #pragma omp target teams distribute parallel for  collapse(3)
    for     (int i  = 0; i < ni       ; i++) {
      for   (int j  = 0; j < nj       ; j++) {
        for (int k  = 0; k < nk       ; k++) {
            F[ind_ii_ijk(ii,i,j,k,ni,nj,nk)] =  F2[ind_ii_ijk(ii,i-di,j-dj,k-dk,ni,nj,nk)];
        }}}
  }
}

void recalculate_BCs_obstacle(type_lbm_obj * lbm_obj){
  int n[3];
  n[0] = lbm_obj->nx_global;
  n[1] = lbm_obj->ny_global;
  n[2] = lbm_obj->nz_global;
  for      (int i  = 0; i < 3 ; i++) {
    for    (int j  = 0; j < 2 ; j++) {
      for  (int k  = 0; k < 2 ; k++) {
      // U_box[i]   =  (  lbm_obj->obstacle_U[i] )/(lbm_obj->scale_u);
      double aux_step = ((double) lbm_obj->iters_full        )*(lbm_obj->dt_phys)*\
                        ((double) lbm_obj->obstacle_advancing)*(lbm_obj->obstacle_U[i]);
      double coord;
      if (j==0){ coord  =  ( (lbm_obj->obstacle_ini_c0[i]) + aux_step )/(lbm_obj->size_elem_phys); }
      if (j==1){ coord  =  ( (lbm_obj->obstacle_ini_c1[i]) + aux_step )/(lbm_obj->size_elem_phys); }
      if (lbm_obj->obstacle_ax_active[i]==0){lbm_obj->obstacle_inds_bbox[i*4 + j*2 + k] = j*(n[i]-1)                                   ;}
      else                                  {lbm_obj->obstacle_inds_bbox[i*4 + j*2 + k] = ((int_ceil(coord) + (k-1))%n[i] + n[i])%n[i] ;}
      }
    lbm_obj->obstacle_ijk_defaults[i*2+j] =  lbm_obj->obstacle_inds_bbox[i*4 + j*2 + (1-j)];
  }}
}

void apply_BCs_obstacle(double* F, double* F2, double* rho, type_lbm_obj* lbm_obj){
  int ni = lbm_obj->nx; int nj = lbm_obj->ny; int nk = lbm_obj->nz;
  for      (int i  = 0; i < 3 ; i++) {
    if (lbm_obj->obstacle_ax_active[i]==1){
    int sp2,sp1,sp21;
    if      (i==0) {sp2 = lbm_obj->pairs_x_shape[2] ; sp1 = lbm_obj->pairs_x_shape[1];}
    else if (i==1) {sp2 = lbm_obj->pairs_y_shape[2] ; sp1 = lbm_obj->pairs_y_shape[1];}
    else if (i==2) {sp2 = lbm_obj->pairs_z_shape[2] ; sp1 = lbm_obj->pairs_z_shape[1];}
    sp21 = sp2*sp1;
    for    (int j  = 0; j < 2 ; j++) {
      for  (int k  = 0; k < 2 ; k++) {
            int i0_ref = lbm_obj->obstacle_ijk_defaults[0];
            int i1_ref = lbm_obj->obstacle_ijk_defaults[1];
            int j0_ref = lbm_obj->obstacle_ijk_defaults[2];
            int j1_ref = lbm_obj->obstacle_ijk_defaults[3];
            int k0_ref = lbm_obj->obstacle_ijk_defaults[4];
            int k1_ref = lbm_obj->obstacle_ijk_defaults[5];
            {int ind = lbm_obj->obstacle_inds_bbox[i*4+2*j+k];
             if      (i==0){i0_ref=ind; i1_ref=ind;}
             else if (i==1){j0_ref=ind; j1_ref=ind;}
             else if (i==2){k0_ref=ind; k1_ref=ind;}}
             int n_runs = (i0_ref<=i1_ref)? 1:2;
          for  (int i_check  = 0; i_check < n_runs ; i_check++) {
            int i0,i1;
            if (i0_ref<=i1_ref){ i0 = i0_ref; i1 = i1_ref; }
            else{
                if (i_check == 0) { i0 = 0     ; i1 = i1_ref              ; }
                else              { i0 = i0_ref; i1 = lbm_obj->nx_global-1; }}
                i0 = max(0   ,i0     - lbm_obj->i_global);
                i1 = min(ni-1,i1     - lbm_obj->i_global);
            int j0 = max(0   ,j0_ref - lbm_obj->j_global);
            int j1 = min(nj-1,j1_ref - lbm_obj->j_global);
            int k0 = max(0   ,k0_ref - lbm_obj->k_global);
            int k1 = min(nk-1,k1_ref - lbm_obj->k_global);
            if (((i0>=0)&&(i0<ni))&&((i1>=0)&&(i1<ni)&&(i0<=i1))&&\
                ((j0>=0)&&(j0<nj))&&((j1>=0)&&(j1<nj)&&(j0<=j1))&&\
                ((k0>=0)&&(k0<nk))&&((k1>=0)&&(k1<nk)&&(k0<=k1))){
            for  (int p  = 0; p < sp1 ; p++) {
                int a,r;
                if      (i==0){a = lbm_obj->pairs_x[k*sp21+p*sp2]; r = lbm_obj->pairs_x[k*sp21+p*sp2+1];}
                else if (i==1){a = lbm_obj->pairs_y[k*sp21+p*sp2]; r = lbm_obj->pairs_y[k*sp21+p*sp2+1];}
                else if (i==2){a = lbm_obj->pairs_z[k*sp21+p*sp2]; r = lbm_obj->pairs_z[k*sp21+p*sp2+1];}
                double loc_factor = (lbm_obj->c_xyz[                      r]*lbm_obj->obstacle_U[0] + \
                                     lbm_obj->c_xyz[  lbm_obj->numel_ci + r]*lbm_obj->obstacle_U[1] + \
                                     lbm_obj->c_xyz[2*lbm_obj->numel_ci + r]*lbm_obj->obstacle_U[2] )/lbm_obj->scale_u*\
                                     (2*lbm_obj->wi[r]/pow(lbm_obj->cs, 2.))*((double) lbm_obj->obstacle_is_moving);
                #pragma omp target teams distribute parallel for  collapse(3)
                for     (int i2  = i0; i2 <= i1 ; i2++) {
                  for   (int j2  = j0; j2 <= j1 ; j2++) {
                    for (int k2  = k0; k2 <= k1 ; k2++) {
                      F[ind_ii_ijk(r,i2,j2,k2,ni,nj,nk)] =  F2[ind_ii_ijk(a,i2,j2,k2,ni,nj,nk)] + loc_factor*rho[ind_ii_ijk(0,i2,j2,k2,ni,nj,nk)];
                    }}}
            }}}}}}}
}

int points_halo_ax_side(int ax, int ii, int numel_ci, int* c_xyz, int mode){
    int total = 0;
    if ((c_xyz[0]!=0)||(c_xyz[numel_ci]!=0)||(c_xyz[2*numel_ci]!=0)){printf("ERROR: c_xyz[0,:]!=0");exit(EXIT_FAILURE);}
    for  (int i  = 1; i < numel_ci ; i++) { // position 0 skipped
        if (mode==0){ if (c_xyz[ax*numel_ci + i] != (2*(1-ii)-1)){ total = total + 1;} }
        else        { if (c_xyz[ax*numel_ci + i] == (2*   ii -1)){ total = total + 1;} }} // ii=0: -1  ii=1: 1
    return total;
}

void fill_halo_pos(int* A, int ax, int ii, int numel_ci, int* c_xyz, int mode){
    int pos = 0;
    if ((c_xyz[0]!=0)||(c_xyz[numel_ci]!=0)||(c_xyz[2*numel_ci]!=0)){printf("ERROR: c_xyz[0,:]!=0");exit(EXIT_FAILURE);}
    for  (int i  = 1; i < numel_ci ; i++) { // position 0 skipped
        if (mode==0){ if (c_xyz[ax*numel_ci + i] != (2*(1-ii)-1)){ A[pos] = i; pos    = pos + 1;}}
        else        { if (c_xyz[ax*numel_ci + i] == (2*   ii -1)){ A[pos] = i; pos    = pos + 1;}}
        } // ii=0: -1  ii=1: 1
}

// int points_halo_ax_side(int ax, int ii, int numel_ci, int* c_xyz, int mode){
//     int total = numel_ci;
//     return total;}
// 
// void fill_halo_pos(int* A, int ax, int ii, int numel_ci, int* c_xyz, int mode){
//     int pos = 0;
//     for  (int i  = 0; i < numel_ci ; i++) {
//             A[pos] = i;
//             pos    = pos + 1;} // ii=0: -1  ii=1: 1
// }


void apply_BCs(double* F, double* F2, type_lbm_obj* lbm_obj){
  int ni = lbm_obj->nx; int nj = lbm_obj->ny; int nk = lbm_obj->nz;
  for      (int i  = 0; i < 3 ; i++) {
    int sp2,sp1,sp21;
    if      (i==0) {sp2 = lbm_obj->pairs_x_shape[2] ; sp1 = lbm_obj->pairs_x_shape[1];}
    else if (i==1) {sp2 = lbm_obj->pairs_y_shape[2] ; sp1 = lbm_obj->pairs_y_shape[1];}
    else if (i==2) {sp2 = lbm_obj->pairs_z_shape[2] ; sp1 = lbm_obj->pairs_z_shape[1];}
    sp21 = sp2*sp1;
    if (lbm_obj->is_per[i]==0){
      for  (int k  = 0; k < 2 ; k++) {
        if (lbm_obj->is_boundary[2*i+k]==1){
        int ijk[6];
        ijk[0] = 0; ijk[1] = ni-1;
        ijk[2] = 0; ijk[3] = nj-1;
        ijk[4] = 0; ijk[5] = nk-1;
        if (k==0) {ijk[2*i+1] = ijk[2*i  ];}
        if (k==1) {ijk[2*i  ] = ijk[2*i+1];}
        int i0 = ijk[0]; int i1 = ijk[1];
        int j0 = ijk[2]; int j1 = ijk[3];
        int k0 = ijk[4]; int k1 = ijk[5];
        for  (int p  = 0; p < sp1 ; p++) {
            int a,r;
            if      (i==0){a = lbm_obj->pairs_x[k*sp21+p*sp2]; r = lbm_obj->pairs_x[k*sp21+p*sp2+1];}
            else if (i==1){a = lbm_obj->pairs_y[k*sp21+p*sp2]; r = lbm_obj->pairs_y[k*sp21+p*sp2+1];}
            else if (i==2){a = lbm_obj->pairs_z[k*sp21+p*sp2]; r = lbm_obj->pairs_z[k*sp21+p*sp2+1];}
            #pragma omp target teams distribute parallel for  collapse(3)
            for     (int i2  = i0; i2 <= i1 ; i2++) {
              for   (int j2  = j0; j2 <= j1 ; j2++) {
                for (int k2  = k0; k2 <= k1 ; k2++) {
                  F[ind_ii_ijk(r,i2,j2,k2,ni,nj,nk)] =  F2[ind_ii_ijk(a,i2,j2,k2,ni,nj,nk)];
                }}}
        }}}}}
}


void init_mpi_info(type_lbm_obj* lbm_obj){
  lbm_obj->mpi_ranks = (int *) malloc((lbm_obj->nproc_mpi) * sizeof(int   ));
  {
  int nx;
  int i_global = 0;
  for      (int i  = 0; i < lbm_obj->divs_mpi_x ; i++) {
    int ny;
    int j_global = 0;
    for    (int j  = 0; j < lbm_obj->divs_mpi_y ; j++) {
      int nz;
      int k_global = 0;
      for  (int k  = 0; k < lbm_obj->divs_mpi_z ; k++) {
        int pos                  = i*lbm_obj->divs_mpi_y*lbm_obj->divs_mpi_z + j*lbm_obj->divs_mpi_z + k;
        lbm_obj->mpi_ranks[pos] = pos;
        nx    = (lbm_obj->nx_global)/lbm_obj->divs_mpi_x;
        ny    = (lbm_obj->ny_global)/lbm_obj->divs_mpi_y;
        nz    = (lbm_obj->nz_global)/lbm_obj->divs_mpi_z;
        int rem_x = lbm_obj->nx_global - nx*lbm_obj->divs_mpi_x;
        int rem_y = lbm_obj->ny_global - ny*lbm_obj->divs_mpi_y;
        int rem_z = lbm_obj->nz_global - nz*lbm_obj->divs_mpi_z;
        if (i<rem_x){ nx = nx+1; }
        if (j<rem_y){ ny = ny+1; }
        if (k<rem_z){ nz = nz+1; }
        if (pos == lbm_obj->irank_mpi){
            lbm_obj->pos_mpi_x = i;
            lbm_obj->pos_mpi_y = j;
            lbm_obj->pos_mpi_z = k;
            lbm_obj->i_global = i_global;
            lbm_obj->j_global = j_global;
            lbm_obj->k_global = k_global;
            lbm_obj->nx = nx;
            lbm_obj->ny = ny;
            lbm_obj->nz = nz;
        }
        k_global = k_global + nz;}
      j_global   = j_global + ny;}
    i_global     = i_global + nx;}
  }
  for   (int i  = 0; i < 3 ; i++) {
    for (int j  = 0; j < 2 ; j++) {
      int p[3],n[3];
      p[0]     = lbm_obj->pos_mpi_x;
      p[1]     = lbm_obj->pos_mpi_y;
      p[2]     = lbm_obj->pos_mpi_z;
      n[0]     = lbm_obj->divs_mpi_x;
      n[1]     = lbm_obj->divs_mpi_y;
      n[2]     = lbm_obj->divs_mpi_z;
      p[i]     = ((p[i] + (2*j-1))%n[i] + n[i])%n[i];
      int pos  =  p[0]*n[1]*n[2] + p[1]*n[2] + p[2];
      if (j==0) {lbm_obj->rank_south[i] = lbm_obj->mpi_ranks[pos];}
      if (j==1) {lbm_obj->rank_north[i] = lbm_obj->mpi_ranks[pos];}
  }}
  lbm_obj->is_boundary[0] = (lbm_obj->pos_mpi_x== 0                     )? 1:0;
  lbm_obj->is_boundary[1] = (lbm_obj->pos_mpi_x==(lbm_obj->divs_mpi_x-1))? 1:0;
  lbm_obj->is_boundary[2] = (lbm_obj->pos_mpi_y== 0                     )? 1:0;
  lbm_obj->is_boundary[3] = (lbm_obj->pos_mpi_y==(lbm_obj->divs_mpi_y-1))? 1:0;
  lbm_obj->is_boundary[4] = (lbm_obj->pos_mpi_z== 0                     )? 1:0;
  lbm_obj->is_boundary[5] = (lbm_obj->pos_mpi_z==(lbm_obj->divs_mpi_z-1))? 1:0;

  lbm_obj->scale_u       =  lbm_obj->size_elem_phys / lbm_obj->dt_phys;
  lbm_obj->scale_nu      =  pow(lbm_obj->size_elem_phys,2)/lbm_obj->dt_phys;
  lbm_obj->scale_body_F  =  lbm_obj->scale_nu*lbm_obj->scale_u/pow(lbm_obj->size_elem_phys,2);
  lbm_obj->nu_lbm        =   lbm_obj->nu_phys/lbm_obj->scale_nu;
  lbm_obj->tau           =   lbm_obj->nu_lbm/pow(lbm_obj->cs,2) + 0.5;
  lbm_obj->obstacle_is_moving = 0;
  for (int i  = 0; i < 3 ; i++) {
    if ((lbm_obj->use_obstacle) && (fabs(lbm_obj->obstacle_U[i])>1e-10)) {lbm_obj->obstacle_is_moving = 1;}
  }
}

void init_self_lbm(type_lbm_obj * lbm_obj){
      // "input.dat"
      FILE * fptr;
  if (lbm_obj->irank_mpi == 0){fptr = fopen("params.dat", "r");
      if (fptr == NULL) { printf("The file is not opened: %s\n","params.dat"); exit(EXIT_FAILURE); }}
      if (lbm_obj->irank_mpi == 0){
      fscanf(fptr, "%d %d %d %*[^\n]%*c", &(lbm_obj->nx_global), &(lbm_obj->ny_global), &(lbm_obj->nz_global));
      fscanf(fptr, "%d %d %d %*[^\n]%*c" , &(lbm_obj->num_iters), &(lbm_obj->snap_iter_start), &(lbm_obj->snap_freq));
      fscanf(fptr, "%d %d %d %*[^\n]%*c", &(lbm_obj->is_per[0]), &(lbm_obj->is_per[1]), &(lbm_obj->is_per[2]));
      fscanf(fptr, "%lf %lf %lf %lf %*[^\n]%*c", &(lbm_obj->size_elem_phys), &(lbm_obj->dt_phys), &(lbm_obj->nu_phys), &(lbm_obj->scale_rho));
      fscanf(fptr, "%lf %d %d %*[^\n]%*c", &(lbm_obj->A_model), &(lbm_obj->guo_Si), &(lbm_obj->use_pressure_bc));
      fscanf(fptr, "%lf %*[^\n]%*c" , &(lbm_obj->cs));
      fscanf(fptr, "%d %d %*[^\n]%*c", &(lbm_obj->use_obstacle), &(lbm_obj->obstacle_advancing));
      fscanf(fptr, "%lf %lf %lf %*[^\n]%*c", &(lbm_obj->obstacle_U[0])     , &(lbm_obj->obstacle_U[1])     , &(lbm_obj->obstacle_U[2]));
      fscanf(fptr, "%lf %lf %lf %*[^\n]%*c", &(lbm_obj->obstacle_ini_c0[0]), &(lbm_obj->obstacle_ini_c0[1]), &(lbm_obj->obstacle_ini_c0[2]));
      fscanf(fptr, "%lf %lf %lf %*[^\n]%*c", &(lbm_obj->obstacle_ini_c1[0]), &(lbm_obj->obstacle_ini_c1[1]), &(lbm_obj->obstacle_ini_c1[2]));
      fscanf(fptr, "%d %d %d %*[^\n]%*c"   , &(lbm_obj->obstacle_ax_active[0]) , &(lbm_obj->obstacle_ax_active[1]), &(lbm_obj->obstacle_ax_active[2]));
      fscanf(fptr, "%d %*[^\n]%*c"  , &(lbm_obj->numel_ci));
      }
      MPI_Bcast(&(lbm_obj->nx_global          ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->ny_global          ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->nz_global          ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->num_iters          ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->snap_iter_start    ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->snap_freq          ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->is_per             ), 3, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->size_elem_phys     ), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->dt_phys            ), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->nu_phys            ), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->scale_rho          ), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->A_model            ), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->guo_Si             ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->use_pressure_bc    ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->cs                 ), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      MPI_Bcast(&(lbm_obj->use_obstacle       ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->obstacle_advancing ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->obstacle_U         ), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->obstacle_ini_c0    ), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->obstacle_ini_c1    ), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->obstacle_ax_active ), 3, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->numel_ci           ), 1, MPI_INT   , 0, MPI_COMM_WORLD);

      lbm_obj->c_xyz = (int   *) malloc(3*(lbm_obj->numel_ci) * sizeof(int   ));
      lbm_obj->wi    = (double*) malloc(  (lbm_obj->numel_ci) * sizeof(double));
      for  (int i  = 0; i < (lbm_obj->numel_ci)       ; i++) {
          int     temp_cx, temp_cy, temp_cz;
          double  temp_wi;
  if (lbm_obj->irank_mpi == 0){fscanf(fptr, "%d %d %d %lf%*c" , &temp_cx, &temp_cy, &temp_cz, &temp_wi);}
      MPI_Bcast(&temp_cx, 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&temp_cy, 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&temp_cz, 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&temp_wi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          lbm_obj->c_xyz[i                       ] = temp_cx;
          lbm_obj->c_xyz[i +   lbm_obj->numel_ci] = temp_cy;
          lbm_obj->c_xyz[i + 2*lbm_obj->numel_ci] = temp_cz;
          lbm_obj->wi   [i                       ] = temp_wi;
      }

      { int temp_i,temp_j,temp_k;
  if (lbm_obj->irank_mpi == 0){fscanf(fptr, "%d %d %d %*[^\n]%*c" , &temp_i, &temp_j, &temp_k);}
        MPI_Bcast(&temp_i, 1, MPI_INT   , 0, MPI_COMM_WORLD);
        MPI_Bcast(&temp_j, 1, MPI_INT   , 0, MPI_COMM_WORLD);
        MPI_Bcast(&temp_k, 1, MPI_INT   , 0, MPI_COMM_WORLD);
        lbm_obj->pairs_x = (int   *) malloc((temp_i*temp_j*temp_k) * sizeof(int   ));
        lbm_obj->pairs_x_shape[0] = temp_i;
        lbm_obj->pairs_x_shape[1] = temp_j;
        lbm_obj->pairs_x_shape[2] = temp_k;
        for        (int i  = 0; i < (temp_i*temp_j*temp_k) ; i++) {
          int     temp_int;
          if (lbm_obj->irank_mpi == 0){fscanf(fptr, "%d%*c" , &temp_int);}
          MPI_Bcast(&temp_int, 1, MPI_INT   , 0, MPI_COMM_WORLD);
          lbm_obj->pairs_x[i] = temp_int;
        }}

      { int temp_i,temp_j,temp_k;
  if (lbm_obj->irank_mpi == 0){fscanf(fptr, "%d %d %d %*[^\n]%*c" , &temp_i, &temp_j, &temp_k);}
        MPI_Bcast(&temp_i, 1, MPI_INT   , 0, MPI_COMM_WORLD);
        MPI_Bcast(&temp_j, 1, MPI_INT   , 0, MPI_COMM_WORLD);
        MPI_Bcast(&temp_k, 1, MPI_INT   , 0, MPI_COMM_WORLD);
        lbm_obj->pairs_y = (int   *) malloc((temp_i*temp_j*temp_k) * sizeof(int   ));
        lbm_obj->pairs_y_shape[0] = temp_i;
        lbm_obj->pairs_y_shape[1] = temp_j;
        lbm_obj->pairs_y_shape[2] = temp_k;
        for        (int i  = 0; i < (temp_i*temp_j*temp_k) ; i++) {
          int     temp_int;
          if (lbm_obj->irank_mpi == 0){fscanf(fptr, "%d%*c" , &temp_int);}
          MPI_Bcast(&temp_int, 1, MPI_INT   , 0, MPI_COMM_WORLD);
          lbm_obj->pairs_y[i] = temp_int;
        }}

      { int temp_i,temp_j,temp_k;
  if (lbm_obj->irank_mpi == 0){fscanf(fptr, "%d %d %d %*[^\n]%*c" , &temp_i, &temp_j, &temp_k);}
        MPI_Bcast(&temp_i, 1, MPI_INT   , 0, MPI_COMM_WORLD);
        MPI_Bcast(&temp_j, 1, MPI_INT   , 0, MPI_COMM_WORLD);
        MPI_Bcast(&temp_k, 1, MPI_INT   , 0, MPI_COMM_WORLD);
        lbm_obj->pairs_z = (int   *) malloc((temp_i*temp_j*temp_k) * sizeof(int   ));
        lbm_obj->pairs_z_shape[0] = temp_i;
        lbm_obj->pairs_z_shape[1] = temp_j;
        lbm_obj->pairs_z_shape[2] = temp_k;
        for        (int i  = 0; i < (temp_i*temp_j*temp_k) ; i++) {
          int     temp_int;
          if (lbm_obj->irank_mpi == 0){fscanf(fptr, "%d%*c" , &temp_int);}
          MPI_Bcast(&temp_int, 1, MPI_INT   , 0, MPI_COMM_WORLD);
          lbm_obj->pairs_z[i] = temp_int;
        }}
        if (lbm_obj->use_pressure_bc==1){
  if (lbm_obj->irank_mpi == 0){
          fscanf(fptr, "%d %*[^\n]%*c"      , &(lbm_obj->pressure_ax));
          fscanf(fptr, "%d %d %*[^\n]%*c"   , &(lbm_obj->pressure_bc_inds[0]), &(lbm_obj->pressure_bc_inds[1]));
          fscanf(fptr, "%d %d %*[^\n]%*c"   , &(lbm_obj->pressure_bc_mode[0]), &(lbm_obj->pressure_bc_mode[1]));
          fscanf(fptr, "%lf %lf %*[^\n]%*c" , &(lbm_obj->pressure_bc_rho[0]) , &(lbm_obj->pressure_bc_rho[1]));}
      MPI_Bcast(&(lbm_obj->pressure_ax     ), 1, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->pressure_bc_inds), 2, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->pressure_bc_mode), 2, MPI_INT   , 0, MPI_COMM_WORLD);
      MPI_Bcast(&(lbm_obj->pressure_bc_rho ), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
  if (lbm_obj->irank_mpi == 0){fclose(fptr);}
}

void update_macroscopic_fields(double* F, double* rho, double* U, double* body_F, double* c_xyz, type_lbm_obj* lbm_obj){
  int ni = lbm_obj->nx; int nj = lbm_obj->ny; int nk = lbm_obj->nz;
  int    numel_ci = lbm_obj->numel_ci;
  double A_model  = lbm_obj->A_model;
  #pragma omp target teams distribute parallel for  collapse(3)
  for     (int i  = 0; i < ni       ; i++) {
    for   (int j  = 0; j < nj       ; j++) {
      for (int k  = 0; k < nk       ; k++) {
          double loc_total = 0;
          for (int ii  = 0; ii < numel_ci ; ii++) {
            loc_total = loc_total + F[ind_ii_ijk(ii,i,j,k,ni,nj,nk)];
          }
          rho[ind_ii_ijk(0,i,j,k,ni,nj,nk)] = loc_total;
      }}}
  #pragma omp target teams distribute parallel for  collapse(4)
  for     (int ax = 0; ax< 3        ; ax++) {
  for     (int i  = 0; i < ni       ;  i++) {
    for   (int j  = 0; j < nj       ;  j++) {
      for (int k  = 0; k < nk       ;  k++) {
          double loc_total = A_model * body_F[ind_ii_ijk(ax,i,j,k,ni,nj,nk)];
          for (int ii  = 0; ii < numel_ci ; ii++) {
            loc_total = loc_total + F[ind_ii_ijk(ii,i,j,k,ni,nj,nk)]*c_xyz[ii+ax*numel_ci];
          }
          U[ind_ii_ijk(ax,i,j,k,ni,nj,nk)] = loc_total/rho[ind_ii_ijk(0,i,j,k,ni,nj,nk)];
      }}}}
}

void zeros_arr(double* A, int numel){
  for  (int i  = 0; i < numel ; i++) { A[i] = 0. ; }
}

void read_or_write_arr(char* fname, double * A, int ax, double conv_factor, int mode_read, type_lbm_obj* lbm_obj){
  int ni = lbm_obj->nx; int nj = lbm_obj->ny; int nk = lbm_obj->nz;
  for     (int i_proc  = 0; i_proc < lbm_obj->nproc_mpi ; i_proc++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (i_proc == lbm_obj->irank_mpi){
      FILE * fptr;
      if (mode_read==1){fptr = fopen(fname, "rb");}
      else             {if (i_proc == 0){ fptr = fopen(fname, "wb");
                                          fwrite(&lbm_obj->nx_global, sizeof(int), 1, fptr);
                                          fwrite(&lbm_obj->ny_global, sizeof(int), 1, fptr);
                                          fwrite(&lbm_obj->nz_global, sizeof(int), 1, fptr);
                                          for     (int i  = 0; i < lbm_obj->nx_global ; i++) {
                                            for   (int j  = 0; j < lbm_obj->ny_global ; j++) {
                                              for (int k  = 0; k < lbm_obj->nz_global ; k++) {
                                                double temp_zero = 0;
                                                fwrite(&temp_zero, sizeof(double), 1, fptr);
                                              }}};
                                          fclose(fptr);}
                        fptr = fopen(fname, "rb+");}
      if (fptr == NULL) { printf("The file is not opened: %s\n",fname); exit(EXIT_FAILURE); }
      for     (int i  = 0; i < ni       ; i++) {
        for   (int j  = 0; j < nj       ; j++) {
          for (int k  = 0; k < nk       ; k++) {
          int pos_global = ((lbm_obj->i_global + i)*(lbm_obj->ny_global)*(lbm_obj->nz_global) + \
                            (lbm_obj->j_global + j)                     *(lbm_obj->nz_global) + \
                            (lbm_obj->k_global + k))*8 + 12;
          int pos_loc =  ind_ii_ijk(ax,i,j,k,lbm_obj->nx,lbm_obj->ny,lbm_obj->nz);
          if (k==0){ fseek(fptr, pos_global, SEEK_SET);
                     if (mode_read==1){fread(&A[pos_loc], sizeof(double), nk, fptr);}}
          if (mode_read==1){A[pos_loc] = A[pos_loc]*conv_factor;}
          else             {double temp_dbl = A[pos_loc]*conv_factor;
                            fwrite(&temp_dbl, sizeof(double), nk, fptr);}
          }}}
      fclose(fptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}


void get_equilibrum(double* Feq, double* rho, double* U, double* Ua, int use_Ua_special, double rho_scalar, int use_rho_scalar, type_lbm_obj* lbm_obj){
    double cs2   =  pow(lbm_obj->cs, 2.);
    int ni = lbm_obj->nx; int nj = lbm_obj->ny; int nk = lbm_obj->nz;
    int numel_ci =  lbm_obj->numel_ci;

    for (int ii = 0; ii < numel_ci; ii++) {
      double  val_ini = (use_Ua_special==1)? 0. : 1. ;
      double my_w, my_w_rho;
      if (use_rho_scalar==1){
        my_w     = 0;
        my_w_rho = (lbm_obj->wi[ii])*rho_scalar;}
      else{
        my_w     = lbm_obj->wi[ii];
        my_w_rho = 0;}
      double  ca_0 = lbm_obj->c_xyz[             ii]/cs2;
      double  ca_1 = lbm_obj->c_xyz[  numel_ci + ii]/cs2;
      double  ca_2 = lbm_obj->c_xyz[2*numel_ci + ii]/cs2;
      // (above) #define loc_factor_Feq(ca,ax,ax2) ...
      double  loc_factor_00 = loc_factor_Feq(ca_0*cs2,0,0);
      double  loc_factor_01 = loc_factor_Feq(ca_0*cs2,0,1);
      double  loc_factor_02 = loc_factor_Feq(ca_0*cs2,0,2);
      double  loc_factor_10 = loc_factor_Feq(ca_1*cs2,1,0);
      double  loc_factor_11 = loc_factor_Feq(ca_1*cs2,1,1);
      double  loc_factor_12 = loc_factor_Feq(ca_1*cs2,1,2);
      double  loc_factor_20 = loc_factor_Feq(ca_2*cs2,2,0);
      double  loc_factor_21 = loc_factor_Feq(ca_2*cs2,2,1);
      double  loc_factor_22 = loc_factor_Feq(ca_2*cs2,2,2);
      #pragma omp target teams distribute parallel for  collapse(3)
      for       (int i  = 0; i < ni       ; i++) {
        for     (int j  = 0; j < nj       ; j++) {
            for (int k  = 0; k < nk       ; k++) {
               Feq[ind_ii_ijk(ii,i,j,k,ni,nj,nk)]  =  (val_ini + Ua[ind_ii_ijk(0,i,j,k,ni,nj,nk)]*(ca_0 + \
                                                                                                   U[ind_ii_ijk(0,i,j,k,ni,nj,nk)]*loc_factor_00  + \
                                                                                                   U[ind_ii_ijk(1,i,j,k,ni,nj,nk)]*loc_factor_01  + \
                                                                                                   U[ind_ii_ijk(2,i,j,k,ni,nj,nk)]*loc_factor_02  ) + \
                                                                 Ua[ind_ii_ijk(1,i,j,k,ni,nj,nk)]*(ca_1 + \
                                                                                                   U[ind_ii_ijk(0,i,j,k,ni,nj,nk)]*loc_factor_10  + \
                                                                                                   U[ind_ii_ijk(1,i,j,k,ni,nj,nk)]*loc_factor_11  + \
                                                                                                   U[ind_ii_ijk(2,i,j,k,ni,nj,nk)]*loc_factor_12  ) + \
                                                                 Ua[ind_ii_ijk(2,i,j,k,ni,nj,nk)]*(ca_2 + \
                                                                                                   U[ind_ii_ijk(0,i,j,k,ni,nj,nk)]*loc_factor_20  + \
                                                                                                   U[ind_ii_ijk(1,i,j,k,ni,nj,nk)]*loc_factor_21  + \
                                                                                                   U[ind_ii_ijk(2,i,j,k,ni,nj,nk)]*loc_factor_22  ) ) \
                                                      *(my_w_rho + my_w*rho[ind_ii_ijk(0,i,j,k,ni,nj,nk)]) + (1-val_ini)*Feq[ind_ii_ijk(ii,i,j,k,ni,nj,nk)];
      }}}}
}

void set_ini_data(double* U, double* rho, double* body_F, type_lbm_obj * lbm_obj){
    {int numel_grid =  (lbm_obj->nx+2)*(lbm_obj->ny+2)*(lbm_obj->nz+2);
     zeros_arr(rho   , numel_grid                   );
     zeros_arr(U     , numel_grid*3                 );
     zeros_arr(body_F, numel_grid*3                 );

    read_or_write_arr("array_ini_rho.dat"    , rho   , 0, pow(lbm_obj->scale_rho,-1), 1, lbm_obj);

    read_or_write_arr("array_ini_U.dat"      , U     , 0, pow(lbm_obj->scale_u,-1)  , 1, lbm_obj);
    read_or_write_arr("array_ini_V.dat"      , U     , 1, pow(lbm_obj->scale_u,-1)  , 1, lbm_obj);
    read_or_write_arr("array_ini_W.dat"      , U     , 2, pow(lbm_obj->scale_u,-1)  , 1, lbm_obj);
    if (lbm_obj->use_pressure_bc == 0){
      read_or_write_arr("array_ini_body_Fx.dat", body_F, 0, pow(lbm_obj->scale_body_F,-1), 1, lbm_obj);
      read_or_write_arr("array_ini_body_Fy.dat", body_F, 1, pow(lbm_obj->scale_body_F,-1), 1, lbm_obj);
      read_or_write_arr("array_ini_body_Fz.dat", body_F, 2, pow(lbm_obj->scale_body_F,-1), 1, lbm_obj);
      }
}}


void sub_apply_pressure_BC(int ax, int ii, int d, int ni, int nj, int nk, double* F, double* body, double* U_ax_ini, \
                        double A_model, double rho_inlet, double rho_outlet, int mode_inlet, int mode_outlet){
    if ((ax<0)||(ax>2)||(ii<0)||(ii>1)){
      printf("Mismatch in inputs [ax = %d] and [ii = (%d)].\n", ax, ii);
      exit(EXIT_FAILURE);
    }
    double rho_bc    =  rho_inlet*(1-ii) + ii*rho_outlet;
    double is_mode_P = mode_inlet*(1-ii) + ii*mode_outlet;
    if (ax==0){
    // ------------------------------ (ax = 0, ii = 0) ------------------------------
      if(ii==0){
        #pragma omp target teams distribute parallel for  collapse(2)
      //for     (int i  = 0; i < ni       ; i++) {
          for   (int j  = 0; j < nj       ; j++) {
            for (int k  = 0; k < nk       ; k++) {
              double U_ax =  (A_model*body[ind_ii_ijk(0,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(0,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] + 2*F[ind_ii_ijk(13,d,j,k,ni,nj,nk)] + 2*F[ind_ii_ijk(15,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] + 2*F[ind_ii_ijk(1,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(3,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(4,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(5,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(6,d,j,k,ni,nj,nk)] + 2*F[ind_ii_ijk(7,d,j,k,ni,nj,nk)] + 2*F[ind_ii_ijk(9,d,j,k,ni,nj,nk)] - rho_bc)/rho_bc;
              U_ax = is_mode_P*U_ax + (1-is_mode_P)*U_ax_ini[ind_ii_ijk(0,d,j,k,ni,nj,nk)];
              double rho_calc =  (A_model*body[ind_ii_ijk(0,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(0,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] + 2*F[ind_ii_ijk(13,d,j,k,ni,nj,nk)] + 2*F[ind_ii_ijk(15,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] + 2*F[ind_ii_ijk(1,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(3,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(4,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(5,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(6,d,j,k,ni,nj,nk)] + 2*F[ind_ii_ijk(7,d,j,k,ni,nj,nk)] + 2*F[ind_ii_ijk(9,d,j,k,ni,nj,nk)])/(U_ax + 1);
              rho_calc = is_mode_P*rho_bc + (1-is_mode_P)*rho_calc;
              F[ind_ii_ijk(2,d,j,k,ni,nj,nk)] =  F[ind_ii_ijk(1,d,j,k,ni,nj,nk)] - 0.333333333333333*U_ax*rho_calc;
              F[ind_ii_ijk(8,d,j,k,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(1,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(3,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(4,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(7,d,j,k,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(10,d,j,k,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(2,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(5,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(6,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(9,d,j,k,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(14,d,j,k,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(1,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(13,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(3,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(4,d,j,k,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(16,d,j,k,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(2,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(15,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(5,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(6,d,j,k,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
        }}}
    // ------------------------------ (ax = 0, ii = 1) ------------------------------
      else {
        #pragma omp target teams distribute parallel for  collapse(2)
      //for     (int i  = 0; i < ni       ; i++) {
          for   (int j  = 0; j < nj       ; j++) {
            for (int k  = 0; k < nk       ; k++) {
              double U_ax =  (A_model*body[ind_ii_ijk(0,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(0,d,j,k,ni,nj,nk)] - 2*F[ind_ii_ijk(10,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] - 2*F[ind_ii_ijk(14,d,j,k,ni,nj,nk)] - 2*F[ind_ii_ijk(16,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] - 2*F[ind_ii_ijk(2,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(3,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(4,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(5,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(6,d,j,k,ni,nj,nk)] - 2*F[ind_ii_ijk(8,d,j,k,ni,nj,nk)] + rho_bc)/rho_bc;
              U_ax = is_mode_P*U_ax + (1-is_mode_P)*U_ax_ini[ind_ii_ijk(0,d,j,k,ni,nj,nk)];
              double rho_calc =  (A_model*body[ind_ii_ijk(0,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(0,d,j,k,ni,nj,nk)] - 2*F[ind_ii_ijk(10,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] - 2*F[ind_ii_ijk(14,d,j,k,ni,nj,nk)] - 2*F[ind_ii_ijk(16,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] - 2*F[ind_ii_ijk(2,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(3,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(4,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(5,d,j,k,ni,nj,nk)] - F[ind_ii_ijk(6,d,j,k,ni,nj,nk)] - 2*F[ind_ii_ijk(8,d,j,k,ni,nj,nk)])/(U_ax - 1);
              rho_calc = is_mode_P*rho_bc + (1-is_mode_P)*rho_calc;
              F[ind_ii_ijk(1,d,j,k,ni,nj,nk)] =  F[ind_ii_ijk(2,d,j,k,ni,nj,nk)] + 0.333333333333333*U_ax*rho_calc;
              F[ind_ii_ijk(7,d,j,k,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(1,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(3,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(4,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(8,d,j,k,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(9,d,j,k,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(2,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(10,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(5,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(6,d,j,k,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(13,d,j,k,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(1,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(14,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(3,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(4,d,j,k,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(15,d,j,k,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(2,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(11,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(12,d,j,k,ni,nj,nk)] + F[ind_ii_ijk(16,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(17,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(18,d,j,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(5,d,j,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(6,d,j,k,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
        }}}}

    else if (ax==1){
      // ------------------------------ (ax = 1, ii = 0) ------------------------------
      if (ii==0){
        #pragma omp target teams distribute parallel for  collapse(2)
        for     (int i  = 0; i < ni       ; i++) {
      //  for   (int j  = 0; j < nj       ; j++) {
            for (int k  = 0; k < nk       ; k++) {
              double U_ax =  (A_model*body[ind_ii_ijk(1,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(0,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] + 2*F[ind_ii_ijk(11,i,d,k,ni,nj,nk)] + 2*F[ind_ii_ijk(14,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] + 2*F[ind_ii_ijk(17,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(1,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(2,i,d,k,ni,nj,nk)] + 2*F[ind_ii_ijk(3,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(5,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(6,i,d,k,ni,nj,nk)] + 2*F[ind_ii_ijk(7,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(9,i,d,k,ni,nj,nk)] - rho_bc)/rho_bc;
              U_ax = is_mode_P*U_ax + (1-is_mode_P)*U_ax_ini[ind_ii_ijk(0,i,d,k,ni,nj,nk)];
              double rho_calc =  (A_model*body[ind_ii_ijk(1,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(0,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] + 2*F[ind_ii_ijk(11,i,d,k,ni,nj,nk)] + 2*F[ind_ii_ijk(14,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] + 2*F[ind_ii_ijk(17,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(1,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(2,i,d,k,ni,nj,nk)] + 2*F[ind_ii_ijk(3,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(5,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(6,i,d,k,ni,nj,nk)] + 2*F[ind_ii_ijk(7,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(9,i,d,k,ni,nj,nk)])/(U_ax + 1);
              rho_calc = is_mode_P*rho_bc + (1-is_mode_P)*rho_calc;
              F[ind_ii_ijk(4,i,d,k,ni,nj,nk)] =  F[ind_ii_ijk(3,i,d,k,ni,nj,nk)] - 0.333333333333333*U_ax*rho_calc;
              F[ind_ii_ijk(8,i,d,k,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(0,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(1,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(2,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(7,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(9,i,d,k,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(12,i,d,k,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(2,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(11,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(5,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(6,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(9,i,d,k,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(13,i,d,k,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(0,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(14,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(1,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(2,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(9,i,d,k,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(18,i,d,k,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(2,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(17,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(5,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(6,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(9,i,d,k,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
        }}}

      // ------------------------------ (ax = 1, ii = 1) ------------------------------
      else{
        #pragma omp target teams distribute parallel for  collapse(2)
        for     (int i  = 0; i < ni       ; i++) {
      //  for   (int j  = 0; j < nj       ; j++) {
            for (int k  = 0; k < nk       ; k++) {
              double U_ax =  (A_model*body[ind_ii_ijk(1,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(0,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] - 2*F[ind_ii_ijk(12,i,d,k,ni,nj,nk)] - 2*F[ind_ii_ijk(13,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] - 2*F[ind_ii_ijk(18,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(1,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(2,i,d,k,ni,nj,nk)] - 2*F[ind_ii_ijk(4,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(5,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(6,i,d,k,ni,nj,nk)] - 2*F[ind_ii_ijk(8,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(9,i,d,k,ni,nj,nk)] + rho_bc)/rho_bc;
              U_ax = is_mode_P*U_ax + (1-is_mode_P)*U_ax_ini[ind_ii_ijk(0,i,d,k,ni,nj,nk)];
              double rho_calc =  (A_model*body[ind_ii_ijk(1,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(0,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] - 2*F[ind_ii_ijk(12,i,d,k,ni,nj,nk)] - 2*F[ind_ii_ijk(13,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] - 2*F[ind_ii_ijk(18,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(1,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(2,i,d,k,ni,nj,nk)] - 2*F[ind_ii_ijk(4,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(5,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(6,i,d,k,ni,nj,nk)] - 2*F[ind_ii_ijk(8,i,d,k,ni,nj,nk)] - F[ind_ii_ijk(9,i,d,k,ni,nj,nk)])/(U_ax - 1);
              rho_calc = is_mode_P*rho_bc + (1-is_mode_P)*rho_calc;
              F[ind_ii_ijk(3,i,d,k,ni,nj,nk)] =  F[ind_ii_ijk(4,i,d,k,ni,nj,nk)] + 0.333333333333333*U_ax*rho_calc;
              F[ind_ii_ijk(7,i,d,k,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(0,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(1,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(2,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(8,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(9,i,d,k,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(11,i,d,k,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(2,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(12,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(5,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(6,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(9,i,d,k,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(14,i,d,k,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(0,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(13,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(1,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(2,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(9,i,d,k,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(17,i,d,k,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(2,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(10,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(15,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(16,i,d,k,ni,nj,nk)] + F[ind_ii_ijk(18,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(5,i,d,k,ni,nj,nk)] - 0.5*F[ind_ii_ijk(6,i,d,k,ni,nj,nk)] + 0.5*F[ind_ii_ijk(9,i,d,k,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
        }}}}

    else if (ax==2){
      // ------------------------------ (ax = 2, ii = 0) ------------------------------
      if (ii==0){
        #pragma omp target teams distribute parallel for  collapse(2)
        for     (int i  = 0; i < ni       ; i++) {
          for   (int j  = 0; j < nj       ; j++) {
      //    for (int k  = 0; k < nk       ; k++) {
              double U_ax =  (A_model*body[ind_ii_ijk(2,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(0,i,j,d,ni,nj,nk)] + 2*F[ind_ii_ijk(11,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] + 2*F[ind_ii_ijk(16,i,j,d,ni,nj,nk)] + 2*F[ind_ii_ijk(18,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(1,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(2,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(3,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(4,i,j,d,ni,nj,nk)] + 2*F[ind_ii_ijk(5,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] + 2*F[ind_ii_ijk(9,i,j,d,ni,nj,nk)] - rho_bc)/rho_bc;
              U_ax = is_mode_P*U_ax + (1-is_mode_P)*U_ax_ini[ind_ii_ijk(0,i,j,d,ni,nj,nk)];
              double rho_calc =  (A_model*body[ind_ii_ijk(2,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(0,i,j,d,ni,nj,nk)] + 2*F[ind_ii_ijk(11,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] + 2*F[ind_ii_ijk(16,i,j,d,ni,nj,nk)] + 2*F[ind_ii_ijk(18,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(1,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(2,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(3,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(4,i,j,d,ni,nj,nk)] + 2*F[ind_ii_ijk(5,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] + 2*F[ind_ii_ijk(9,i,j,d,ni,nj,nk)])/(U_ax + 1);
              rho_calc = is_mode_P*rho_bc + (1-is_mode_P)*rho_calc;
              F[ind_ii_ijk(6,i,j,d,ni,nj,nk)] =  F[ind_ii_ijk(5,i,j,d,ni,nj,nk)] - 0.333333333333333*U_ax*rho_calc;
              F[ind_ii_ijk(10,i,j,d,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(0,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(1,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(2,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(9,i,j,d,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(12,i,j,d,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(1,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(11,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(3,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(4,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(15,i,j,d,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(0,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(16,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(1,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(2,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(17,i,j,d,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(1,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(18,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(3,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(4,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] - 0.166666666666667*U_ax*rho_calc;
        }}}

      // ------------------------------ (ax = 2, ii = 1) ------------------------------
      else {
      #pragma omp target teams distribute parallel for  collapse(2)
        for     (int i  = 0; i < ni       ; i++) {
          for   (int j  = 0; j < nj       ; j++) {
      //    for (int k  = 0; k < nk       ; k++) {
              double U_ax =  (A_model*body[ind_ii_ijk(2,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(0,i,j,d,ni,nj,nk)] - 2*F[ind_ii_ijk(10,i,j,d,ni,nj,nk)] - 2*F[ind_ii_ijk(12,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] - 2*F[ind_ii_ijk(15,i,j,d,ni,nj,nk)] - 2*F[ind_ii_ijk(17,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(1,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(2,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(3,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(4,i,j,d,ni,nj,nk)] - 2*F[ind_ii_ijk(6,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] + rho_bc)/rho_bc;
              U_ax = is_mode_P*U_ax + (1-is_mode_P)*U_ax_ini[ind_ii_ijk(0,i,j,d,ni,nj,nk)];
              double rho_calc =  (A_model*body[ind_ii_ijk(2,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(0,i,j,d,ni,nj,nk)] - 2*F[ind_ii_ijk(10,i,j,d,ni,nj,nk)] - 2*F[ind_ii_ijk(12,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] - 2*F[ind_ii_ijk(15,i,j,d,ni,nj,nk)] - 2*F[ind_ii_ijk(17,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(1,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(2,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(3,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(4,i,j,d,ni,nj,nk)] - 2*F[ind_ii_ijk(6,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] - F[ind_ii_ijk(8,i,j,d,ni,nj,nk)])/(U_ax - 1);
              rho_calc = is_mode_P*rho_bc + (1-is_mode_P)*rho_calc;
              F[ind_ii_ijk(5,i,j,d,ni,nj,nk)] =  F[ind_ii_ijk(6,i,j,d,ni,nj,nk)] + 0.333333333333333*U_ax*rho_calc;
              F[ind_ii_ijk(9,i,j,d,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(0,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(10,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(1,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(2,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(11,i,j,d,ni,nj,nk)] =  -0.5*A_model*body[ind_ii_ijk(1,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(12,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(3,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(4,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(16,i,j,d,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(0,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(15,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(1,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(2,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
              F[ind_ii_ijk(18,i,j,d,ni,nj,nk)] =  0.5*A_model*body[ind_ii_ijk(1,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(13,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(14,i,j,d,ni,nj,nk)] + F[ind_ii_ijk(17,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(3,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(4,i,j,d,ni,nj,nk)] + 0.5*F[ind_ii_ijk(7,i,j,d,ni,nj,nk)] - 0.5*F[ind_ii_ijk(8,i,j,d,ni,nj,nk)] + 0.166666666666667*U_ax*rho_calc;
        }}}}

}

void apply_pressure_BCs(double* F, double* U, double* body_F, double* U_ax_ini, type_lbm_obj* lbm_obj){
  int ni = lbm_obj->nx; int nj = lbm_obj->ny; int nk = lbm_obj->nz;
  int ax = lbm_obj->pressure_ax;

  int ind_global, n;

  switch (ax) {
    case 0: ; n = ni; ind_global=lbm_obj->i_global; break;
    case 1: ; n = nj; ind_global=lbm_obj->j_global; break;
    case 2: ; n = nk; ind_global=lbm_obj->k_global; break;
  }

  if (lbm_obj->use_pressure_bc==1){
    for     (int i  = 0; i < 2 ; i++) {
    int ind = lbm_obj->pressure_bc_inds[i] - ind_global;
    if ((0<=ind)&&(ind<n)){
      sub_apply_pressure_BC(ax, i, ind, ni, nj, nk, F, body_F, U_ax_ini, lbm_obj->A_model, lbm_obj->pressure_bc_rho[0], lbm_obj->pressure_bc_rho[1], \
                                                                                        lbm_obj->pressure_bc_mode[0], lbm_obj->pressure_bc_mode[1]);
      }}
  }}

double compare_with_arr(char* fname, double* A, int numel_shape_file, int ax_A, int ax_file, double conv_factor, type_lbm_obj* lbm_obj){
  double max_error = 0.;
  int  read_error  = 0 ;
  for     (int i_proc  = 0; i_proc < lbm_obj->nproc_mpi ; i_proc++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (i_proc == lbm_obj->irank_mpi){
      FILE * fptr;
      fptr  =  fopen(fname, "rb");
      if (fptr == NULL) { read_error=1; }
      else {
      for     (int i  = 0; i < lbm_obj->nx ; i++) {
        for   (int j  = 0; j < lbm_obj->ny ; j++) {
          for (int k  = 0; k < lbm_obj->nz ; k++) {
          int pos_global = (ax_file*(lbm_obj->nx_global   )*(lbm_obj->ny_global)*(lbm_obj->nz_global) + \
                                    (lbm_obj->i_global + i)*(lbm_obj->ny_global)*(lbm_obj->nz_global) + \
                                    (lbm_obj->j_global + j)                     *(lbm_obj->nz_global) + \
                                    (lbm_obj->k_global + k))*8 + 4*numel_shape_file;
          int pos_loc =  ind_ii_ijk(ax_A,i,j,k,lbm_obj->nx,lbm_obj->ny,lbm_obj->nz);
          if (k==0){
            fseek(fptr, pos_global, SEEK_SET);
            for (int p  = 0; p < lbm_obj->nz ; p++) {
              double val_read;
              fread(&val_read, sizeof(double), 1, fptr);
              max_error = max(max_error, fabs(val_read - A[pos_loc]*conv_factor));
              }}
          }}}
      fclose(fptr);}
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&read_error, 1, MPI_INT, i_proc, MPI_COMM_WORLD);
    if (read_error != 0) { printf("The file is not opened: %s\n",fname); exit(EXIT_FAILURE); }
    MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&max_error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return max_error;
}

void update_host(double* F, double* F2, double* U, double* body_F, double* rho, double* buffer_mpi, double* c_xyz_gpu, type_lbm_obj* lbm_obj){
  // update from device
  int sz_F, sz_U, sz_r, sz_b, sz_c, base_grid;
  base_grid = (lbm_obj->nx+2)*(lbm_obj->ny+2)*(lbm_obj->nz+2);
  sz_F      = base_grid*lbm_obj->numel_ci;
  sz_U      = base_grid*3;
  sz_r      = base_grid;
  sz_b      = lbm_obj->size_buf_mpi;
  sz_c      = lbm_obj->numel_ci * 3;
  #pragma omp target update from(         F[0:sz_F])
  #pragma omp target update from(        F2[0:sz_F])
  #pragma omp target update from(         U[0:sz_U])
  #pragma omp target update from(    body_F[0:sz_U])
  #pragma omp target update from(       rho[0:sz_r])
  #pragma omp target update from(buffer_mpi[0:sz_b])
  #pragma omp target update from( c_xyz_gpu[0:sz_c])
}

int main(int argc, char *argv[]){

    /* ---------- Begin: MPI Initialization ---------- */

    MPI_Init(&argc, &argv);
    int irank_mpi, nproc_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &irank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc_mpi);

#ifdef _OPENMP
    {
    int count_gpu_mpi, id_device_gpu_mpi;
    count_gpu_mpi      =  omp_get_num_devices();
    omp_set_default_device(irank_mpi % count_gpu_mpi);
    id_device_gpu_mpi  =  omp_get_default_device();
    for  (int i = 0; i < nproc_mpi; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i == irank_mpi){
      printf("Information: MPI rank %d/%d with GPU %d/%d\n", irank_mpi, nproc_mpi, id_device_gpu_mpi, count_gpu_mpi);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      }
    }
#else
    for  (int i = 0; i < nproc_mpi; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i == irank_mpi){
      printf("Information: MPI rank %d/%d (GPU disabled)\n", irank_mpi, nproc_mpi);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
    /* ---------- End: MPI Initialization ---------- */

    // ---------- Initialize "self" object ----------
    type_lbm_obj  lbm_obj;
    { // MPI layout
       lbm_obj.irank_mpi = irank_mpi;
       lbm_obj.nproc_mpi = nproc_mpi;
       lbm_obj.divs_mpi_x = atoi(argv[1]);
       lbm_obj.divs_mpi_y = atoi(argv[2]);
       lbm_obj.divs_mpi_z = atoi(argv[3]);
       if (lbm_obj.nproc_mpi != (lbm_obj.divs_mpi_x*lbm_obj.divs_mpi_y*lbm_obj.divs_mpi_z)){
         printf("Mismatch between [nproc_mpi = %d] and [divs_mpi = (%d,%d,%d)].\n",lbm_obj.nproc_mpi, lbm_obj.divs_mpi_x, lbm_obj.divs_mpi_y, lbm_obj.divs_mpi_z);
         exit(EXIT_FAILURE);
       }
    }
    init_self_lbm(&lbm_obj); // params, scheme, obstacle, etc.
    init_mpi_info(&lbm_obj);
    int numel_halo_x, numel_halo_y, numel_halo_z;
    {
    int numel_halo_xp  =  points_halo_ax_side(0, 1, lbm_obj.numel_ci, lbm_obj.c_xyz, 0);
    int numel_halo_xm  =  points_halo_ax_side(0, 0, lbm_obj.numel_ci, lbm_obj.c_xyz, 0);
    int numel_halo_yp  =  points_halo_ax_side(1, 1, lbm_obj.numel_ci, lbm_obj.c_xyz, 1);
    int numel_halo_ym  =  points_halo_ax_side(1, 0, lbm_obj.numel_ci, lbm_obj.c_xyz, 1);
    int numel_halo_zp  =  points_halo_ax_side(2, 1, lbm_obj.numel_ci, lbm_obj.c_xyz, 1);
    int numel_halo_zm  =  points_halo_ax_side(2, 0, lbm_obj.numel_ci, lbm_obj.c_xyz, 1);
    if (numel_halo_xm!=numel_halo_xp){printf("ERROR: numel_halo_xm!=numel_halo_xp (%d, %d)", numel_halo_xm, numel_halo_xp); exit(EXIT_FAILURE);}
    if (numel_halo_ym!=numel_halo_yp){printf("ERROR: numel_halo_ym!=numel_halo_yp (%d, %d)", numel_halo_ym, numel_halo_yp); exit(EXIT_FAILURE);}
    if (numel_halo_zm!=numel_halo_zp){printf("ERROR: numel_halo_zm!=numel_halo_zp (%d, %d)", numel_halo_zm, numel_halo_zp); exit(EXIT_FAILURE);}
    numel_halo_x = numel_halo_xp;
    numel_halo_y = numel_halo_yp;
    numel_halo_z = numel_halo_zp;
    }

    int*      halo_xp  =  (int*) malloc((numel_halo_x)*sizeof(int)); fill_halo_pos(halo_xp, 0, 1, lbm_obj.numel_ci, lbm_obj.c_xyz, 0);
    int*      halo_xm  =  (int*) malloc((numel_halo_x)*sizeof(int)); fill_halo_pos(halo_xm, 0, 0, lbm_obj.numel_ci, lbm_obj.c_xyz, 0);
    int*      halo_yp  =  (int*) malloc((numel_halo_y)*sizeof(int)); fill_halo_pos(halo_yp, 1, 1, lbm_obj.numel_ci, lbm_obj.c_xyz, 1);
    int*      halo_ym  =  (int*) malloc((numel_halo_y)*sizeof(int)); fill_halo_pos(halo_ym, 1, 0, lbm_obj.numel_ci, lbm_obj.c_xyz, 1);
    int*      halo_zp  =  (int*) malloc((numel_halo_z)*sizeof(int)); fill_halo_pos(halo_zp, 2, 1, lbm_obj.numel_ci, lbm_obj.c_xyz, 1);
    int*      halo_zm  =  (int*) malloc((numel_halo_z)*sizeof(int)); fill_halo_pos(halo_zm, 2, 0, lbm_obj.numel_ci, lbm_obj.c_xyz, 1);

    lbm_obj.size_buf_mpi = 4*max((lbm_obj.nx+2)*(lbm_obj.ny+2)               *numel_halo_z, \
                             max((lbm_obj.nx+2)               *(lbm_obj.nz+2)*numel_halo_y, \
                                                (lbm_obj.ny+2)*(lbm_obj.nz+2)*numel_halo_x));

    printf("numel_halos: %d %d %d \n", numel_halo_x, numel_halo_y, numel_halo_z);
    
    #pragma omp target enter data map(to: halo_xm[0:numel_halo_x])
    #pragma omp target enter data map(to: halo_xp[0:numel_halo_x])
    #pragma omp target enter data map(to: halo_ym[0:numel_halo_y])
    #pragma omp target enter data map(to: halo_yp[0:numel_halo_y])
    #pragma omp target enter data map(to: halo_zm[0:numel_halo_z])
    #pragma omp target enter data map(to: halo_zp[0:numel_halo_z])

    // ---------- Initialize arrays ----------
    double *F           =     malloc_layers(lbm_obj.numel_ci);
    double *F2          =     malloc_layers(lbm_obj.numel_ci);
    double *U           =     malloc_layers(               3);
    double *U_ax_ini    =     malloc_layers(               1);
    double *body_F      =     malloc_layers(               3);
    double *rho         =     malloc_layers(               1);
    double *buffer_mpi  =  (double*) malloc(  lbm_obj.size_buf_mpi * sizeof(double));
    double *c_xyz_gpu   =  (double*) malloc(3*lbm_obj.numel_ci     * sizeof(double));

    for  (int i  = 0; i < 3*lbm_obj.numel_ci; i++) {
        c_xyz_gpu[i] = lbm_obj.c_xyz[i];
    }

    set_ini_data(U, rho, body_F, &lbm_obj);
    
    {   int ni = lbm_obj.nx; int nj = lbm_obj.ny; int nk = lbm_obj.nz;
        int ax = (lbm_obj.use_pressure_bc)? lbm_obj.pressure_ax:0;
        for     (int i  = 0; i < ni       ;  i++) {
          for   (int j  = 0; j < nj       ;  j++) {
            for (int k  = 0; k < nk       ;  k++) {
              U_ax_ini[ind_ii_ijk(0,i,j,k,ni,nj,nk)] = U[ind_ii_ijk(ax,i,j,k,ni,nj,nk)];
        }}}
    }

    { int sz_F, sz_U, sz_r, sz_b, sz_c, base_grid;
      base_grid = (lbm_obj.nx+2)*(lbm_obj.ny+2)*(lbm_obj.nz+2);
      sz_F      = base_grid*lbm_obj.numel_ci;
      sz_U      = base_grid*3;
      sz_r      = base_grid;
      sz_b      = lbm_obj.size_buf_mpi;
      sz_c      = lbm_obj.numel_ci * 3;
      zeros_arr(F  , sz_F);
      zeros_arr(F2 , sz_F);

      #pragma omp target enter data map(to:          F[0:sz_F])
      #pragma omp target enter data map(to:         F2[0:sz_F])
      #pragma omp target enter data map(to:          U[0:sz_U])
      #pragma omp target enter data map(to:     body_F[0:sz_U])
      #pragma omp target enter data map(to:        rho[0:sz_r])
      #pragma omp target enter data map(to:   U_ax_ini[0:sz_r])
      #pragma omp target enter data map(to: buffer_mpi[0:sz_b])
      #pragma omp target enter data map(to:  c_xyz_gpu[0:sz_c])
    }

    // ---------- Initialize F,F2 ----------
    get_equilibrum(F , rho, U, U, 0, 0., 0, &lbm_obj);
    get_equilibrum(F2, rho, U, U, 0, 0., 0, &lbm_obj);

    // ---------- Initialize macroscopic fields and obstacle ----------
    update_macroscopic_fields(F, rho, U, body_F, c_xyz_gpu, &lbm_obj);
    lbm_obj.iters_full = 0;
    if (lbm_obj.use_obstacle==1) {recalculate_BCs_obstacle(&lbm_obj);}

    // ---------- Iterate ----------
    int iter_next_check = 1;
    int last_iter       = 0;
    double last_t0      = MPI_Wtime() ;
    double t0_start     = MPI_Wtime() ;
    
    for (int iters_full = 1; iters_full <= lbm_obj.num_iters ; iters_full++) {
      lbm_obj.iters_full = iters_full;

                              get_equilibrum(F2, rho, U, U     , 0, 0.             , 0, &lbm_obj);
      if (lbm_obj.guo_Si==1){ get_equilibrum(F2, rho, U, body_F, 1, lbm_obj.tau-0.5, 1, &lbm_obj); }

      {
        int ni = lbm_obj.nx; int nj = lbm_obj.ny; int nk = lbm_obj.nz;
        int numel_ci   = lbm_obj.numel_ci;
        double inv_tau = pow(lbm_obj.tau,-1);
        #pragma omp target teams distribute parallel for  collapse(4)
        for         (int ii = 0; ii < numel_ci ; ii++) {
          for       (int i  = 0; i < ni        ;  i++) {
            for     (int j  = 0; j < nj        ;  j++) {
                for (int k  = 0; k < nk        ;  k++) {
                   F[ind_ii_ijk(ii,i,j,k,ni,nj,nk)]  =  F[ind_ii_ijk(ii,i,j,k,ni,nj,nk)] * (1 - inv_tau) + F2[ind_ii_ijk(ii,i,j,k,ni,nj,nk)] * inv_tau;
                   F2[ind_ii_ijk(ii,i,j,k,ni,nj,nk)] =  F[ind_ii_ijk(ii,i,j,k,ni,nj,nk)];
        }}}}
      }
      update_halos(F2, buffer_mpi, &lbm_obj,\
                   numel_halo_x, numel_halo_y, numel_halo_z,\
                         halo_xm, halo_xp, halo_ym, halo_yp, halo_zm, halo_zp);

      streaming(F, F2, &lbm_obj);
      apply_BCs(F, F2, &lbm_obj);
      if (lbm_obj.use_obstacle==1) {apply_BCs_obstacle(F, F2, rho, &lbm_obj);}
      if (lbm_obj.use_pressure_bc==1) {apply_pressure_BCs(F, U, body_F, U_ax_ini, &lbm_obj);}
      update_macroscopic_fields(F, rho, U, body_F, c_xyz_gpu, &lbm_obj);
      if ((lbm_obj.use_obstacle==1) && (lbm_obj.obstacle_advancing==1)) {recalculate_BCs_obstacle(&lbm_obj);}

      if (((iters_full>=lbm_obj.snap_iter_start)&&(iters_full%lbm_obj.snap_freq == 0))||\
          (iters_full==lbm_obj.num_iters)) {
         char str_U[27], str_V[27], str_W[27], str_rho[29];
         snprintf(str_U  , sizeof(str_U  ), "%11s%09d%6s", "array_iter_", iters_full, "_U.dat"  );
         snprintf(str_V  , sizeof(str_V  ), "%11s%09d%6s", "array_iter_", iters_full, "_V.dat"  );
         snprintf(str_W  , sizeof(str_W  ), "%11s%09d%6s", "array_iter_", iters_full, "_W.dat"  );
         snprintf(str_rho, sizeof(str_rho), "%11s%09d%8s", "array_iter_", iters_full, "_rho.dat");
         update_host(F, F2, U, body_F, rho, buffer_mpi, c_xyz_gpu, &lbm_obj);
         read_or_write_arr(str_U   , U     , 0, lbm_obj.scale_u  , 0, &lbm_obj);
         read_or_write_arr(str_V   , U     , 1, lbm_obj.scale_u  , 0, &lbm_obj);
         read_or_write_arr(str_W   , U     , 2, lbm_obj.scale_u  , 0, &lbm_obj);
         read_or_write_arr(str_rho , rho   , 0, lbm_obj.scale_rho, 0, &lbm_obj);
       }
       
       if ((lbm_obj.irank_mpi == 0)&&((iters_full>=iter_next_check)||(iters_full==lbm_obj.num_iters))) {
         double elapsed_dt =  MPI_Wtime() - t0_start;
         double iters_sec  =  ((double) (iters_full - last_iter))/(MPI_Wtime() - last_t0);
         int elapsed_h = elapsed_dt/3600;
         int elapsed_m = (elapsed_dt-elapsed_h*3600)/60;
         int elapsed_s = elapsed_dt-elapsed_h*3600 - elapsed_m*60;
         printf("Iteration: %6d (iters/s: %8.2f) (elapsed time: %02d:%02d:%02d)\n", iters_full, iters_sec, elapsed_h, elapsed_m, elapsed_s);
         iter_next_check   =  (int) (iters_full + iters_sec*10);
         last_iter         =  iters_full;
         last_t0           =  MPI_Wtime() ;
       }
    }
    update_host(F, F2, U, body_F, rho, buffer_mpi, c_xyz_gpu, &lbm_obj);
    {
    double max_error = 0;
    double loc_error;
    FILE * fptr;
    fptr = fopen("report_errors.txt", "w");
    loc_error = compare_with_arr("array_end_U_phys.dat"  , U  , 3, 0, 0, lbm_obj.scale_u  , &lbm_obj);max_error=max(loc_error,max_error);
    fprintf(fptr,"Max. error (U end): %.6e %.6e\n",loc_error,max_error);
    loc_error = compare_with_arr("array_end_V_phys.dat"  , U  , 3, 1, 0, lbm_obj.scale_u  , &lbm_obj);max_error=max(loc_error,max_error);
    fprintf(fptr,"Max. error (V end): %.6e %.6e\n",loc_error,max_error);
    loc_error = compare_with_arr("array_end_W_phys.dat"  , U  , 3, 2, 0, lbm_obj.scale_u  , &lbm_obj);max_error=max(loc_error,max_error);
    fprintf(fptr,"Max. error (W end): %.6e %.6e\n",loc_error,max_error);
    loc_error = compare_with_arr("array_end_rho_phys.dat", rho, 3, 0, 0, lbm_obj.scale_rho, &lbm_obj);max_error=max(loc_error,max_error);
    fprintf(fptr,"Max. error (rho end): %.6e %.6e\n",loc_error,max_error);
    for         (int ii = 0; ii < lbm_obj.numel_ci ; ii++) {
        loc_error = compare_with_arr("array_end_F.dat", F, 4, ii, ii, (double) 1, &lbm_obj);max_error=max(loc_error,max_error);
     fprintf(fptr,"Max. error (F[%d] end): %.6e %.6e\n",ii,loc_error,max_error);}
    fprintf(fptr,"Max. error (all): %.6e\n",max_error);
    if (max_error>1e-10){ exit(EXIT_FAILURE);}
    fclose(fptr);
    }
}
