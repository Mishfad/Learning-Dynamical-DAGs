run_parallelized_fixed_freq_correlated_noise: Runs the simulation with correlated noise.

conti_samp=0      : restart and record sampling
conti_samp=1      : continuous sampling 
correlated_noise=0: i.i.d. noise
correlated_noise=1: temporally-correlated noise

running the Python file will store the G=G_hat for each network to a CSV file. This file is opened in a notebook in the jupyter notebook below to plot errors.


The plots are generated using the jupyter notebook "plots_upload.ipynb"