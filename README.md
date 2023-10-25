# Learning-Dynamical-DAGs
Learning DAGs with time dependencies from time series data

Use of code allowed. Please cite Veedu, M.S., Deka, D. and Salapaka, M.V., 2023. Information Theoretically Optimal Sample Complexity of Learning Dynamical Directed Acyclic Graphs. arXiv preprint arXiv:2308.16859.

run_parallelized_fixed_freq_correlated_noise: Runs the simulation with correlated noise.

different settings (see the paper for details)
conti_samp=0      : restart and record sampling
conti_samp=1      : continuous sampling 
correlated_noise=0: i.i.d. exogenous noise
correlated_noise=1: temporally-correlated (wide sense stationary) exogenous noise

running the Python file will store the G=G_hat, where G_hat is the estimated graph, for each network to a CSV file. This file is opened in a notebook in the jupyter notebook below to plot errors.


The plots are generated using the jupyter notebook "plots_upload.ipynb"
