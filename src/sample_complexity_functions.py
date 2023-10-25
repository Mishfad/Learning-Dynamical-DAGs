import networkx as nx
import random
# You can visualize the DAG using NetworkX's built-in drawing functions
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg
import netgraph
import numpy as np
import sympy as sym
from itertools import combinations

def generate_random_dag_with_max_in_degree(num_nodes, max_in_degree):
# generates a random dag with maximum in degree equal to max_in_degree
    if num_nodes <= 0:
        raise ValueError("Number of nodes should be greater than 0.")    
    if max_in_degree <= 0:
        raise ValueError("Maximum in-degree should be greater than 0.")
    if max_in_degree >= num_nodes:
        raise ValueError("Maximum in-degree should be less than the number of nodes.")
    
    G = nx.DiGraph()
    nodes = list(range(num_nodes))
    G.add_nodes_from(node for node in nodes)
    for node in nodes:
        in_degree = max_in_degree #random.randint(1, max_in_degree)
        if node - in_degree < 0:
            in_degree = node  # Avoid in-degrees exceeding node indices
        in_neighbors = random.sample(nodes[:node], in_degree)
        G.add_edges_from((neighbor, node) for neighbor in in_neighbors)
    
    return G



def generate_graph(num_nodes,max_in_degree,file_loc=None):
    # random_dag = generate_random_dag(num_nodes, avg_degree)
    random_dag = generate_random_dag_with_max_in_degree(num_nodes, max_in_degree)
    Adj = nx.adjacency_matrix(random_dag).toarray().T

    # Print some basic information about the DAG
    print("Number of nodes:", random_dag.number_of_nodes())
    print("Number of edges:", random_dag.number_of_edges())
    # print("Adjacency matrix:", Adj.shape," \n", Adj)
    return random_dag,Adj

def plot_digraph(random_dag):
    # plots the directed graph
    pos = nx.spring_layout(random_dag, seed=42)
    # nx.draw(random_dag, pos, with_labels=True, node_size=500, font_size=10, font_color="black", arrowsize=20)

    # Allow for interactive node movement
    plt.figure()
    # nx.draw_networkx_nodes(random_dag, pos, node_size=500, node_color="blue")  # Draw nodes with a different color
    plt.title("Drag nodes to adjust their positions")

    plot_instance = netgraph.InteractiveGraph(random_dag,pos
                , node_size=5
                , node_color="lightblue"
                , node_labels=True
                # , node_label_offset=0.1
                # , node_label_fontdict=dict(size=20)
                , edge_color="green"
                , edge_width=1
                , arrows=True
        # , ax=ax
        )
    # Show the plot and allow for interaction
    plt.show()

def plot_digraph_from_adj(adj):
# plot the directed graph from the Adjacency matrix
    nNodes=adj.shape[0]
    G=nx.Graph()
    G.add_nodes_from([str(ind) for ind in range(nNodes)])
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            if adj[ind1,ind2]:
                G.add_edge(str(ind1),str(ind2))
    plot_digraph(G)


##############################################################################################################################
# Creates a function to generate data according to the restart and record format
# Parameters are 
# 1) nTrajectories: Number of independent trajectories, 
# 2) nSamples: Number of samples per trajectory
# 3) B-p x p x DELAY-1
# 4) noise_pow: Noise power (noise is i.i.d. Gausian)
def restart_record_data_generation(nTrajectories,nSamples,B,noise_pow=-1):
    print("generating restart and record data",nTrajectories,nSamples)
    nNodes=B.shape[0]
    if noise_pow.all()==-1:
        noise_pow=np.ones(B.shape[0]) # if no value is passed, then generated standard normal noise
    y=np.zeros((nTrajectories,nNodes,nSamples),dtype=complex)
    for ind_Traj in range(nTrajectories): # start an independent trajectory
        x=np.zeros((nSamples,nNodes))
        x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
        x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
        # evolution according to the linear dynamical system
        for ind_Samp in range(2,nSamples):
            x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)#-A@x[ind_Samp-1,:] # @ performs matrix multiplication
            x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,1],x[ind_Samp-1,:])+np.dot(B[:,:,2],x[ind_Samp-2,:]) #+np.dot(B[:,:,2],x[ind_Samp-3,:])
# computes the fft with proper scaling.
        X=np.fft.fft(x,axis=0)/np.sqrt(nSamples)
        y[ind_Traj,:,:]=np.transpose(X) #X is (nFreq,nSamples)
    return y
##############################################################################################################################        

def restart_record_data_generation_correlated_noise(nTrajectories,nSamples,B,noise_pow=-1):
# **same as "restart_record_data_generation". Here, the noise is temporally correlated noise
# Creates a function to generate data according to the restart and record format
# Parameters are 
# 1) nTrajectories: Number of independent trajectories, 
# 2) nSamples: Number of samples per trajectory
# 3) B-p x p x DELAY-1
# 4) noise_pow: Noise power (noise is i.i.d. Gausian)
    print("generating restart and record data",nTrajectories,nSamples)
    nNodes=B.shape[0]
    if noise_pow.all()==-1:
        noise_pow=np.ones(B.shape[0])
    y=np.zeros((nTrajectories,nNodes,nSamples),dtype=complex)
    for ind_Traj in range(nTrajectories):
        x=np.zeros((nSamples,nNodes))
        w=np.random.randn(nSamples,nNodes)*np.sqrt(noise_pow[0])
        e=np.zeros([nSamples,nNodes])
        e[0,:]=w[0,:]
        for ind in range(1,nSamples):
            e[ind,:]=.5*e[ind-1,:]+w[ind,:]
        x[0,:]=e[0,:]
        x[1,:]=e[1,:] #-A[:,0]*x[0,:]
        for ind_Samp in range(2,nSamples):
            x[ind_Samp,:]=e[ind_Samp,:] #-A[:,0]*x[ind_Samp-1,:]-A[:,1]*x[ind_Samp-2,:]-A[:,2]*x[ind_Samp-3,:]
            x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,1],x[ind_Samp-1,:])+np.dot(B[:,:,2],x[ind_Samp-2,:]) #+np.dot(B[:,:,2],x[ind_Samp-3,:])
        X=np.fft.fft(x,axis=0)/np.sqrt(nSamples)
        y[ind_Traj,:,:]=np.transpose(X) #X is (nFreq,nSamples)
    return y
##############################################################################################################################        

def continuous_data_generation_correlated_noise(nSamples,B,C=None,noise_pow=-1):
    nNodes=B.shape[0]
    if noise_pow.all()==-1:
        noise_pow=np.ones(B.shape[0])
    # if C.all()==None:
        # noise_pow=np.ones(B.shape[0])
    print("Generating continous data.. nSamples: ",nSamples," for nNodes :",nNodes)
    x=np.zeros((nSamples,nNodes))
    # e=create_WSS_noise(nSamples=nSamples,num_nodes=nNodes,noise_pow)
    w=np.random.randn(nSamples,nNodes)*np.sqrt(noise_pow[0])
    e=np.zeros([nSamples,nNodes])
    e[0,:]=w[0,:]
    for ind in range(1,nSamples):
        e[ind,:]=.5*e[ind-1,:]+w[ind,:]
    x[0,:]=e[0,:]
    x[1,:]=e[1,:] #-A[:,0]*x[0,:]
    # x[2,:]=np.random.randn(nNodes)*np.sqrt(noise_pow) #-A[:,0]*x[1,:]-A[:,1]*x[0,:]
    for ind_Samp in range(2,nSamples):
        x[ind_Samp,:]=e[ind_Samp,:] #-A[:,0]*x[ind_Samp-1,:]-A[:,1]*x[ind_Samp-2,:]-A[:,2]*x[ind_Samp-3,:]
        x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,1],x[ind_Samp-1,:])+np.dot(B[:,:,2],x[ind_Samp-2,:])
    return x


def continuous_data_generation(nSamples,B,A=None,noise_pow=None):
# Similar to "restart_record_data_generation". Here, the noise is iid
# Creates a function to generate data according to the continuous sampling, i.e,., one single trajectory of length nSamples
# Parameters are 
# 2) nSamples: Number of samples per trajectory
# 3) B-p x p x DELAY-1
# 4) noise_pow: Noise power (noise is i.i.d. Gausian)
    nNodes=B.shape[0]
    if noise_pow.all()==-1:
        noise_pow=np.ones(B.shape[0])
    print("Generating continous data.. nSamples: ",nSamples," for nNodes :",nNodes)
    x=np.zeros((nSamples,nNodes))
    x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
    x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow) #-A[:,0]*x[0,:]
    x[2,:]=np.random.randn(nNodes)*np.sqrt(noise_pow) #-A[:,0]*x[1,:]-A[:,1]*x[0,:]
# evolution according to the linear dynamical system
    for ind_Samp in range(3,nSamples):
        x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow) #-A[:,0]*x[ind_Samp-1,:]-A[:,1]*x[ind_Samp-2,:]-A[:,2]*x[ind_Samp-3,:]
        x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,1],x[ind_Samp-1,:])+np.dot(B[:,:,2],x[ind_Samp-2,:])
    return x
##############################################################################################################################        

def compute_fft(data,nfft):
    # compute fft without overlap
    nSamples,nNodes=data.shape
    nTrajectories=np.int32(nSamples/nfft) 
    y=np.zeros((nTrajectories,nNodes,nfft),dtype=complex)
    for ind in range(nTrajectories-1): # discards final few residual samples
        x=data[ind*(nfft):(ind+1)*nfft,:]
        X=np.fft.fft(x,axis=0)/np.sqrt(nfft)
        y[ind,:,:]=np.transpose(X) #X is (nfft,nNodes)
    return y

 
def compute_coeffs(Adj,DELAY=2,FILTER_LIST=None):
    # compute coefficient matrices for a given network. The input parameter is the adjacency matrix
    # *** This function is not optimized. Since it wasn't causing any issues, I haven't optimized this function ***
    nNodes=Adj.shape[0]
    B=np.ones([nNodes,nNodes,3])
    delta=.5 # lower-limit of the uniform random variable
    # genrate a Rademacher random variable
    B_scale1=(np.random.uniform(-1,1,size=[nNodes,nNodes]))
    B_scale1=2*(B_scale1>0)-1
# generates a matrix with each entry rad*unif(delta,1) , where rad is the Rademacher rv {+1,-1} with prob 1/2 each
    B_scale=np.random.uniform(delta,1,size=[nNodes,nNodes])*B_scale1
    B_scale=np.repeat(B_scale[:nNodes, :, np.newaxis], 3, axis=2)
    Adj1=np.repeat(Adj[:, :, np.newaxis], 3, axis=2)
    B=B*Adj1
    # makes all the entries without edges zero
    B=B*B_scale
    # B has only the entries corresponsing to lag 1. Set others to zero
    B[:,:,2:3]=0
    B[:,:,0]=0
    A=np.diag(np.random.rand(nNodes))
    return A,B

def compute_coeffs_lag_1(Adj):
    # compute coefficient matrices for a given network with lag 1. The input parameter is the adjacency matrix
    nNodes=Adj.shape[0]
    B=np.ones([nNodes,nNodes])
    B=B*Adj
    # genrate a Rademacher random variable
    Rademacher_rv=2*((np.random.uniform(-1,1,size=[nNodes,nNodes]))>0)-1
    delta=.5  # lower-limit of the uniform random variable
# generates a matrix with each entry rad*unif(delta,1) , where rad is the Rademacher rv {+1,-1} with prob 1/2 each
    B_scale=np.random.uniform(delta,1,size=[nNodes,nNodes])*Rademacher_rv
    B1=B*B_scale
    B=np.zeros([nNodes,nNodes,3])
    B[:,:,1]=B1
    A=np.diag(np.random.rand(nNodes))
    return A,B

def compute_CPSD(psd_est,ind,C):
    phi_CC_inv=np.linalg.inv(psd_est(C,C))
    temp1=np.dot(psd_est(ind,C),phi_CC_inv)
    f_est=psd_est(ind,ind)-np.dot(np.dot(temp1,psd_est(C,ind)))
    return f_est

def compute_f_metric(psd,i,c):
    # computes the metric f(i,C,omega). Instead of omega, we pass psd(omega) as a parameter to the function
    # inputs: i, psd(omega), c
    if c==None or c==[]: 
        return psd[i,i]
    if len(c)==1: # np.linalg.inv worked only if dimension > 1
        phi_C_inv=1/psd[c,c]
        temp=psd[i,c] * phi_C_inv * psd[c,i]
    else:
        phi_C_inv=np.linalg.inv(psd[c,:][:,c])
        temp=psd[i,c] @ phi_C_inv @ psd[c,i]
    # print(phi_C_inv,temp)
    return psd[i,i]-temp

def find_topological_order(psd,q):
    #  inputs : psd. q (max in degree)
    # returns topological ordering
    num_nodes=psd.shape[0]
    V=[ind for ind in range(num_nodes)]
    S=[]
    C=[]
    # for ind in range(num_nodes):
    psd_diag=psd.diagonal()
    # min_val = np.min(psd_diag)
    min_row = np.argmin(psd_diag)
    # print(min_val,min_row)
    S.append(min_row)
    C.append([])
    V.remove(min_row)
    for count in range(num_nodes-1):
        # print(count, V)
        combination_set=[]
        for ind_combi in range(1,min(q,len(S))+1):
            [combination_set.append(i) for i in combinations(S,ind_combi)]
        # print(combination_set)

        fvals=10**5*np.ones([num_nodes,len(combination_set)],dtype=np.complex128)
        for c_ind in range(len(combination_set)):
            # print("\n combination set: ",combination_set[c_ind])
            for ind in V:
                temp=compute_f_metric(psd,ind,combination_set[c_ind])
                fvals[ind,c_ind]=temp
        min_val = np.min(fvals)
        min_index = np.argmin(fvals)
        num_rows, num_cols = fvals.shape
        min_row, min_col = divmod(min_index, num_cols)
        # print(min_val,(min_row,min_col))
        S.append(min_row)
        C.append(combination_set[min_col])
        # print(fvals)
        V.remove(min_row)

    # print(S,"\n",C)
    return S,C
        # i_star,C_star=find_minimizer(psd_est,C)



def theoretical_analysis(B,omega=0*np.pi/64,noise_var=1):
    num_nodes=B.shape[0]
    psd_theo=np.zeros([num_nodes,num_nodes,len(omega)],dtype=complex)
    H_min=10
    for ind in range(len(omega)):
        omega1=omega[ind]
        z = sym.Symbol('z')
        A=sym.eye(B.shape[0])
        num=sym.Matrix(B[:,:,0])+sym.Matrix(B[:,:,1])/z+sym.Matrix(B[:,:,2])/z**2
        
        H = sym.Matrix(num) #num/den[:,None]
        H1=H.subs(z,sym.exp(1j*omega1))
        H_abs=np.array(abs(H1)).astype(np.complex128)
        H_min1=min(H_abs[H_abs>.00000001])
        H_min=min(H_min1,H_min)
        I_H=(sym.eye(A.shape[0])-H1)
        I_H = np.array(I_H.expand(complex=True)).astype(np.complex128)
        I_H_inv=np.linalg.inv(I_H)
        I_H_inv_star = np.transpose(np.conjugate(I_H_inv))
        psd_theo1 = I_H_inv @ I_H_inv_star*noise_var[0]
        psd_theo[:,:,ind]=psd_theo1
    return psd_theo,H_min
    
def find_parents(psd,ordered_set,C,gamma):
    num_nodes=len(ordered_set)
    Pa=[]
    diff_j_list=[]
    # diff_j_list_remove=[]
    for ind in range(num_nodes):
        node=ordered_set[ind]
        parents=[]
        for j in C[ind]:
            P=list(C[ind])
            P.remove(j)
            diff_j=compute_f_metric(psd,node,P)-compute_f_metric(psd,node,C[ind])
            diff_j_list.append([diff_j,node,P,j])
            if diff_j>gamma:
                parents.append(j)
        Pa.append(parents)
        # print("Node ",node,", Parents: ",Pa[ind])
    return Pa

def estimate_adj_from_parents(ordered_set,parents):
    num_nodes=len(ordered_set)
    Adj=np.zeros([num_nodes,num_nodes],dtype=bool)
    for ind in range(num_nodes):
        Adj[ordered_set[ind],parents[ind]]=True
    return Adj

##############################################################################################################################
# Creates a function to generate data according to the restart and record format
# Parameters are 
# 1) Number of independent trajectories, 
# 2) NUmber of samples per trajectory
# 3) B-p x p x DELAY-1 and A- diagonal matrix
# 4) Noise power (noise is i.i.d. Gausian)
def restart_record_node_wise(nTrajectories,nSamples,B,noise_pow=None):
    print("generating restart and record data",nTrajectories,nSamples)
    nNodes=B.shape[0]
    if noise_pow.any()==None:
        noise_pow=np.ones(B.shape[0])
    y=np.zeros((nTrajectories,nNodes,nSamples),dtype=complex)
    for ind_Traj in range(nTrajectories):
        x=np.zeros((nSamples,nNodes))
        x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
        x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
        for ind_Samp in range(1,nSamples):
            x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)#-A@x[ind_Samp-1,:] # @ performs matrix multiplication
            for ind_row in range(nNodes):
                for ind_col in range(ind_row):
                    x[ind_Samp,ind_row]=x[ind_Samp,ind_row]+B[ind_row,ind_col,0]*x[ind_Samp,ind_col]+B[ind_row,ind_col,1]*x[ind_Samp-1,ind_col]+B[ind_row,ind_col,2]*x[ind_Samp-2,ind_col]
        X=np.fft.fft(x,axis=0)
        y[ind_Traj,:,:]=np.transpose(X) #X is (nFreq,nSamples)
    return y
##############################################################################################################################        



########### Theoretical #############
def run_theoretical(B1,max_in_degree, nfft,noise_pow,Adj):
    psd_theo,H_min=theoretical_analysis(B1,omega=np.asarray(range(nfft))*np.pi/nfft,noise_var=noise_pow)
    Delta=noise_pow[0]*H_min**2
    print("H_min=",H_min, "Delta: ",Delta)# print(psd_theo)

    # print("H_min=",H_min)
    reconstruction_error=[]
    # print("Num edges: ",np.sum(Adj))
    for ind in range(nfft):
        ordered_set,C=find_topological_order(psd_theo[:,:,ind],max_in_degree)
        parents=find_parents(psd_theo[:,:,ind],ordered_set,C,gamma=Delta/2)
        # print(ind, "Theoretical ordered set : ", ordered_set)
        # print("Theoretical parents: ", parents)
        Adj_theory=estimate_adj_from_parents(ordered_set,parents)
        reconstruction_error.append(np.sum(Adj_theory!=Adj))
        # if reconstruction_error>0:
    # print("Errors: theory ",(reconstruction_error))
    return reconstruction_error,psd_theo,H_min,ordered_set,parents
#######################################################


###################################################################################################################
def parallel_run_function(ind_net,num_nodes,max_in_degree,nfft,ind_f,nTraj,Samples,noise_pow1,conti_samp,correlated_noise):
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


