# Formal validation
  - Successful replication of results from Breuer et al. (2000) regarding the changes in the drag coefficients (as a function of the Reynolds number) for the flow past a square cylinder within a channel:
      - M. Breuer, J. Bernsdorf, T. Zeiser, F. Durst (2000). Accurate computations of the laminar flow past a square cylinder based on two different methods: lattice-Boltzmann and finite-volume. International Journal of Heat and Fluid Flow, vol. 21, pp. 186-196.


<img src="post_breuer/drag_coeff_comparison.png" alt="drawing" width="290"/>

<img src="post_breuer/table_drag_coeff_comparison.png" alt="drawing" width="490"/>


## General information:
  1) The files `python_lbm_validation_breuer_init.py` and `python_lbm_validation_breuer_post.py` contain the pre- and post-processing of the validation study.
  2) Folder `data_Cd_breuer` contains the Python code to scan the flow data from Breuer et al. (2000). (The colors in `data_drag_Breuer.png` are to identify the data points.)
  3) The script `commands_ubuntu_breuer.txt` is intended to run the case files in an Ubuntu desktop.
  4) To run the validation cases in a cluster with a slurm-based system, the file `commands_lumi_breuer_Re_1_5_10.txt` can be considered as an example. This file was originally used to run the cases in the LUMI supercomputer.
     - The `slurm` files `job_CRAY_Re[1-5-10].slurm` were pre-configured to submit batch jobs in LUMI.

## How to use:
  1) Run `python_lbm_validation_breuer_init.py` to generate `local_runs` subfolder.
  2) Run the CFD cases from `local_runs` in your computer/cluster. 
     - The pre-configured files `commands_[...]_breuer_[...].txt` can be used as a reference for local workstations or clusters (adapting the environment variables).
  3) Run `python_lbm_validation_breuer_post.py` to compute the resulting curve for the drag coefficient.