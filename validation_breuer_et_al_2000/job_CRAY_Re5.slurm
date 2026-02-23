#!/bin/bash -l
#SBATCH -J dns
#SBATCH -o %x-j%j.out
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8 
#SBATCH -N 1
#SBATCH --account=project_???
#SBATCH --partition=standard-g
#SBATCH --time 10:00:00
#SBATCH --exclusive

# https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/
echo "Starting job $SLURM_JOB_ID at `date`"
echo "USER: ${USER}"
echo "PWD: ${PWD}"


# --------------- environment variables ---------------
export ROCM_GPU=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'`
if [ -z "$ROCM_GPU" ]; then       echo "WARNING: ROCM_GPU=$ROCM_GPU was empty. Assigning arbitrary value. (ROCM_GPU=gfx90a is the right answer at the moment)";       ROCM_GPU=gfx90a; else       echo "SUCCESS: ROCM_GPU=$ROCM_GPU was non-empty."; fi
echo "Final value of ROCM_GPU: " $ROCM_GPU


export ROCM_HOME=/opt/rocm-6.3.4
export ROCM_PATH=/opt/rocm-6.3.4
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export OMP_PLACES=threads  
export OMP_PROC_BIND=close  

# --------------- compile hipfort ---------------
export CRAY_ACC_DEBUG=0
export SBATCH_ACCOUNT=project_???
export ENABLE_AMD_GPU=1

export HSA_ENABLE_SDMA=0
export MPICH_GPU_IPC_THRESHOLD=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_NET_GDR_LEVEL=3
export NCCL_NCHANNELS_PER_PEER=32 

# --------------- application directory ---------------

clear;clear;clear

app_name=c_version_lbm_solver
mypwd=/scratch/project_???/???/breuer_tests/breuer_runs
lumi_folder=$mypwd/..

subfolder=/scratch/project_???/???/breuer_tests/breuer_runs/run_breuer_Re_5

echo "Subfolder: ${subfolder}"

cd $subfolder
rm -f ${app_name}.c
rm -f ../${app_name}.c
rm -f report_errors.txt
cp -f $lumi_folder/${app_name}.c ${app_name}.c

cc  -O3 -fopenmp -lmpi -o ${app_name}_gpu  $app_name.c

cat << EOF > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF
chmod +x ./select_gpu
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
echo "Begin run: ${subfolder}"

srun -n 8 --cpu-bind=${CPU_BIND} --distribution=block:block ./select_gpu ./${app_name}_gpu 4 2 1
mytoc=$(python3 -c "import time;from sys import argv;print(time.time()-float(argv[1]))" $mytic)




