# import networkx as nx
# import random
# You can visualize the DAG using NetworkX's built-in drawing functions
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg
import numpy as np
# import sympy as sym
import os
import sys 
import time
import csv
from multiprocessing import Pool 
from itertools import repeat

directory = os.getcwd()
parent_path = os.path.dirname(directory) # # Getting the parent directory name where the current directory is present.
sys.path.append(parent_path)  # # setting path... # equialent to  sys.path.append('../')

# The following imports load the functions from the user defined functions list. The folder src contains sample_complexity_functions.py, which contains all
# the necessary python functions

from src.sample_complexity_functions import find_topological_order
from src.sample_complexity_functions import theoretical_analysis
from src.sample_complexity_functions import find_parents
from src.sample_complexity_functions import restart_record_node_wise
from src.sample_complexity_functions import restart_record_data_generation
from src.sample_complexity_functions import continuous_data_generation
from src.sample_complexity_functions import continuous_data_generation_correlated_noise
from src.sample_complexity_functions import restart_record_data_generation_correlated_noise
from src.sample_complexity_functions import compute_fft
from src.sample_complexity_functions import generate_graph
from src.sample_complexity_functions import compute_coeffs_lag_1
from src.sample_complexity_functions import plot_digraph
from src.sample_complexity_functions import plot_digraph_from_adj
from src.sample_complexity_functions import estimate_adj_from_parents
from src.sample_complexity_functions import run_theoretical
from src.plot_functions import plot_directed_graph_6nodes

###################################################################################################################
# Initialization of parameters
###################################################################################################################

max_in_degree_list=[2,3]
# node_list = [10,20,30]
node_list = [5,6]
nfft=64
Samples=[10**ind for ind in range(1,3)]
# Samples=[10,100,1000,5*10**3,10**4,2*10**4,5*10**4,7*10**4]
nTraj=Samples[len(Samples)-1]
nNetworks=50 # number of random networks where the algorithm is tested, for a fixed num_nodes and max_in_degree. 
# Used to approximate P(G=G_hat) by #(G=G_hat)/nNetworks. We used 50 for the simulations in the paper.
ind_f=17 # must be between 0 and nfft-1. The frequency at which the algorithm is tested
conti_samp=0 # conti_samp=1(0) denote continuous sampling (restart and record sampling) see the paper for details
correlated_noise=1 # correlated_noise=1(0) denote temporally-correlated(i.i.d.) noise
file_loc=directory

# parallel_run_function performs the major part of the simulation




###################################################################################################################
def parallel_run_function(ind_net,num_nodes,max_in_degree):
    # the function generates a random network, generates parameters, data for some nSamples/nTrajectories
    # From data, estimates PSD for the given ind_f frequency using the estimator in the paper.
    # using the estimated PSD, performs the algorithm and 
    # returns G==G_hat, relative errror, and Adjecency matrix
    ###################################################################################################################
    print("ind_net: ",ind_net)
    random_dag,Adj=generate_graph(num_nodes,max_in_degree)
    # plot_digraph(random_dag)
    # plot_digraph_from_adj(Adj)
    # Compute A, B coeffs with a given adjacency matrix and delay
    A1,B1=compute_coeffs_lag_1(Adj)
    B1=B1/1.002/np.linalg.norm(B1[:,:,1],ord=2)
    ########### Theoretical #############
    psd_theo,H_min=theoretical_analysis(B1,omega=np.asarray(range(nfft))*np.pi/nfft,noise_var=noise_pow1)
    # reconstruction_error,psd_theo,H_min,ordered_set,parents=run_theoretical(B1,max_in_degree,nfft,noise_pow1,Adj=Adj)
    Delta=noise_pow1[0]*H_min**2
    # print(ordered_set)
    # print(parents)
    ################################################################## 

    ########### ########### Generate data ############################################  
    if conti_samp==0 and correlated_noise==0: # RR
        # file_name=file_loc+"/RR_sampling_"+str(nTraj)+"_"+str(nfft)+"-FFT_"+str(num_nodes)+"nodes"+".csv"
        data1=restart_record_data_generation(nTrajectories=nTraj,nSamples=nfft,B=B1,noise_pow=noise_pow1)
    elif conti_samp==0 and correlated_noise==1:
        data1=restart_record_data_generation_correlated_noise(nTrajectories=nTraj,nSamples=nfft,B=B1,noise_pow=noise_pow1)
    elif conti_samp==1 and correlated_noise==0:
        # file_name=file_loc+"/conti_sampling_"+str(nTraj)+"_"+str(nfft)+"-FFT_"+str(num_nodes)+"nodes"+".csv"
        data2=continuous_data_generation(nTraj*nfft,B=B1,noise_pow=noise_pow1)
        # print("data_generated")
        data1=compute_fft(data2,nfft)
        # print("Computed FFT")
    elif conti_samp==1 and correlated_noise==1:
        # file_name=file_loc+"/conti_sampling_"+str(nTraj)+"_"+str(nfft)+"-FFT_"+str(num_nodes)+"nodes"+".csv"
        data2=continuous_data_generation_correlated_noise(nTraj*nfft,B=B1,noise_pow=noise_pow1)
        # print("data_generated")
        data1=compute_fft(data2,nfft)
        # print("Computed FFT")
########### ########### ########### ########### ########### ########### ########### 
    graph_error_samples=np.zeros([len(Samples)])
    error_samples=np.zeros([len(Samples)])
    # det_samples=np.zeros([len(Samples),nfft])
    for ind_samples in range(len(Samples)):
        # print(Samples[ind_samples])
# reads specific number of samples from data1
        x=data1[:Samples[ind_samples],:,:]
        x_shape=x.shape
        rec_error=np.zeros(nfft)
        # det=np.zeros(nfft)
        # graph_error=np.zeros(nfft)
########### ########### PSD estimation ########### ###########  
        psd_est=np.zeros([x_shape[1],x_shape[1]],dtype=complex)
        X=x[:,:,ind_f]
        psd_est=np.dot(X.T,np.conj(X))/(X.shape[0])
        # psd_error=abs(psd_est-psd_theo[:,:,ind_f])
        # print("psd error: ",np.max(np.max(abs(psd_est-psd_theo),0),0))
########### ########### ########### ########### ###########   
# Run Algorithm 1 from estimated psd
        ordered_set_est,C_est=find_topological_order(psd_est,max_in_degree)
        parents_est=find_parents(psd_est,ordered_set_est,C_est,gamma=Delta/2)
        # print("Estimated ordered set : ", ordered_set_est)
        # print("Estimated parents: ", parents_est)
        Adj_data=estimate_adj_from_parents(ordered_set_est,parents_est)
########### ########### ########### ########### ########### ########### 
        reconstruction_error_est=np.sum(np.abs(Adj_data-Adj))
        graph_err=np.sum(np.abs(Adj_data-Adj))==0

        rec_error[ind_f]=reconstruction_error_est
        graph_error_samples[ind_samples]=graph_err
        error_samples[ind_samples]=reconstruction_error_est
        # det_samples[ind_samples,:]=det
        # print("I(G=G_hat): ",graph_error_samples)
        # print("Detection: ",det_samples, "\nnum_edges:",np.sum(Adj))
    return graph_error_samples,error_samples,Adj



###################################################################################################################
# Code to perform simulation
###################################################################################################################

for ind_q in range(len(max_in_degree_list)): # iterate through the list of p and q
    for ind_p in range(len(node_list)):
        num_nodes=node_list[ind_p]
        max_in_degree = max_in_degree_list[ind_q]
        noise_pow1=.5*np.ones(num_nodes)

    ########### ########### ########### ########### ########### ########### ########### ########### 
        # main part of the file
    ####################################################################################### 
        avg_errors=[]
        prob_correc=[]
        start_time=time.localtime()
        print("Start time: ",start_time)
        if __name__ == '__main__':
            # start 8 worker processes    
            pool= Pool(6)
            # out1=list()
            # for ind in range(nNetworks):
            #     out=parallel_run_function(ind)
            #     out1.append(out)
            out1=pool.starmap(parallel_run_function, zip(range(nNetworks),repeat(num_nodes), repeat(max_in_degree)))
            pool.close()
            for ind in range(len(out1)):
                prob_correc.append(out1[ind][0])
                avg_errors.append(out1[ind][1]/(np.sum(out1[ind][2])))
            
            end_time=time.localtime()
            print("Start time: ",start_time)
            print("End time: ",end_time)

            print("done 2")

                # print(avg_errors, prob_correc)
            omega=[2*i*np.pi/nfft for i in range(nfft)]
            ################################################################################################
            # Writing prob_error to file
            ################################################################################################
            if conti_samp==0 and correlated_noise==0:
                file_name=file_loc+"/error_data/prob_error_RR_p="+str(num_nodes)+"_q="+str(max_in_degree)+"_nfft="+str(nfft)+"_nTraj="+str(nTraj)+".csv"
            elif conti_samp==1 and correlated_noise==0:
                file_name=file_loc+"/error_data/prob_error_conti_p="+str(num_nodes)+"_q="+str(max_in_degree)+"_nfft="+str(nfft)+"_nTraj="+str(nTraj)+".csv"
            elif conti_samp==0 and correlated_noise==1:
                file_name=file_loc+"/error_data/prob_error_corre_RR_p="+str(num_nodes)+"_q="+str(max_in_degree)+"_nfft="+str(nfft)+"_nTraj="+str(nTraj)+".csv"
            elif conti_samp==1 and correlated_noise==1:
                file_name=file_loc+"/error_data/prob_error_corre_conti_p="+str(num_nodes)+"_q="+str(max_in_degree)+"_nfft="+str(nfft)+"_nTraj="+str(nTraj)+".csv"


            with open(file_name, 'w', newline='\n') as csvfile:
                csvwriter=csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(omega)
                csvwriter.writerow(["nNetworks: ",nNetworks])
                csvwriter.writerow(["Prob(G=G_hat)"])
                
                for ind1 in range(len(prob_correc)):
                    csvwriter.writerow(prob_correc[ind1])
                csvwriter.writerow(["time: start:",start_time])
                csvwriter.writerow(["time: stop:",end_time])
                csvwriter.writerow(["number of Samples"])
                csvwriter.writerow(Samples)



            ################################################################################################
            # Writing rec_error to file
            ################################################################################################
            if conti_samp==0 and correlated_noise==0:
                file_name=file_loc+"/error_data/rec_error_RR_p="+str(num_nodes)+"_q="+str(max_in_degree)+"_nfft="+str(nfft)+"_nTraj="+str(nTraj)+".csv"
            elif conti_samp==1 and correlated_noise==0:
                file_name=file_loc+"/error_data/rec_error_conti_p="+str(num_nodes)+"_q="+str(max_in_degree)+"_nfft="+str(nfft)+"_nTraj="+str(nTraj)+".csv"
            elif conti_samp==0 and correlated_noise==1:
                file_name=file_loc+"/error_data/rec_error_corre_RR_p="+str(num_nodes)+"_q="+str(max_in_degree)+"_nfft="+str(nfft)+"_nTraj="+str(nTraj)+".csv"
            elif conti_samp==1 and correlated_noise==1:
                file_name=file_loc+"/error_data/rec_error_corre_conti_p="+str(num_nodes)+"_q="+str(max_in_degree)+"_nfft="+str(nfft)+"_nTraj="+str(nTraj)+".csv"

            with open(file_name, 'w', newline='\n') as csvfile:
                csvwriter=csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(omega)
                csvwriter.writerow(["nNetworks: ",nNetworks])
                # csvwriter.writerow(["Number of wrong edges"])
                # for ind in range(len(Samples)):
                    # csvwriter.writerow(avg_errors[ind,:])
                csvwriter.writerow(["Reconstruction errors"])
                
                for ind1 in range(len(avg_errors)):
                    # csvwriter.writerow(["ind:", ind1])
                    # for ind2 in range(len(Samples)):
                    csvwriter.writerow(avg_errors[ind1])

                csvwriter.writerow(["number of Samples"])
                csvwriter.writerow(Samples)
                csvwriter.writerow(["ind_f",ind_f])
                csvwriter.writerow(["time: start:",start_time])
                csvwriter.writerow(["time: stop:",end_time])
                csvwriter.writerow(Samples)

                # for ind in range(nfft):
                #     plt.plot(Samples,prob_correc[:,ind],label="q="+str(max_in_degree)+" "+str(ind))
                # plt.show()
                # plt.legend()
                # print("done 1")

