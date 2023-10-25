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

def generate_random_dag(num_nodes, avg_degree):
    if num_nodes <= 0:
        raise ValueError("Number of nodes should be greater than 0.")
    
    if avg_degree <= 0:
        raise ValueError("Average degree should be greater than 0.")
    
    if avg_degree >= num_nodes:
        raise ValueError("Average degree should be less than the number of nodes.")
    
    G = nx.DiGraph()
    nodes = list(range(num_nodes))
    
    for node in nodes:
        # Add directed edges to the current node from nodes that come later in the list
        out_degree = random.randint(0, avg_degree * 2)
        if node + out_degree >= num_nodes:
            out_degree = num_nodes - node - 1
        out_neighbors = random.sample(nodes[node + 1:], out_degree)
        G.add_edges_from((node, neighbor) for neighbor in out_neighbors)
    
    return G

def generate_random_dag_with_max_in_degree(num_nodes, max_in_degree):
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
        in_degree = random.randint(1, max_in_degree)
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
    print("Adjacency matrix:", Adj.shape," \n", Adj)
    return random_dag,Adj

def plot_digraph(random_dag):

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


##############################################################################################################################
# Creates a function to generate data according to the restart and record format
# Parameters are 
# 1) Number of independent trajectories, 
# 2) NUmber of samples per trajectory
# 3) B-p x p x DELAY-1 and A- diagonal matrix
# 4) Noise power (noise is i.i.d. Gausian)
def restart_record_data_generation(nTrajectories,nSamples,B,A,noise_pow=None):
    print("generating restart and record data",nTrajectories,nSamples)
    nNodes=B.shape[0]
    if noise_pow==None:
        noise_pow=np.ones(A.shape[0])
    # print(B)
    y=np.zeros((nTrajectories,nNodes,nSamples),dtype=complex)
    for ind_Traj in range(nTrajectories):
        # print(ind_Traj)
        x=np.zeros((nSamples,nNodes))
        x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
        # x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A*x[0,:]
        for ind_Samp in range(1,nSamples):
            x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)#-A@x[ind_Samp-1,:] # @ performs matrix multiplication
            # print(ind_Samp, "Previous sample: ",x[ind_Samp-1,:])
            # print(ind_Samp, "Present sample: ",x[ind_Samp,:])
            x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,0],x[ind_Samp-1,:])+np.dot(B[:,:,1],x[ind_Samp-2,:])+np.dot(B[:,:,2],x[ind_Samp-3,:])
            # print("B*X(t-1)",np.dot(B,x[ind_Samp-1,:]))
            # print(x[ind_Samp,:])
        # print(x)
        # y=np.tile(x[:2,0],[3,1])
        # print(y)
        X=np.fft.fft(x,axis=0)
        # print("Shape: ",X.shape)
        # print("\n\nfft:",X)
        y[ind_Traj,:,:]=np.transpose(X) #X is (nFreq,nSamples)
        # print("y:",y[ind_Traj,:,:])
    return y
##############################################################################################################################        

def continuous_data_generation(nSamples,B,A,noise_pow):
    nNodes=B.shape[0]
    print("Generating continous data.. nSamples: ",nSamples," for nNodes :",nNodes)
    x=np.zeros((nSamples,nNodes))
    x[0,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)
    x[1,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[0,:]
    x[2,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[1,:]-A[:,1]*x[0,:]
    for ind_Samp in range(3,nSamples):
        x[ind_Samp,:]=np.random.randn(nNodes)*np.sqrt(noise_pow)-A[:,0]*x[ind_Samp-1,:]-A[:,1]*x[ind_Samp-2,:]-A[:,2]*x[ind_Samp-3,:]
        # print(ind_Samp, "Previous sample: ",x[ind_Samp-1,:])
        # print(ind_Samp, "Present sample: ",x[ind_Samp,:])
        x[ind_Samp,:]=x[ind_Samp,:]+np.dot(B[:,:,0],x[ind_Samp-1,:])+np.dot(B[:,:,1],x[ind_Samp-2,:])+np.dot(B[:,:,2],x[ind_Samp-3,:])
        # print("B*X(t-1)",np.dot(B,x[ind_Samp-1,:]))
        # print(x[ind_Samp,:])
        # print(x)
        # X=np.fft.fft(x,axis=0)
        # y=np.tile(x[:2,0],[3,1])
        # print(y)
        # print("Shape: ",X.shape)
        # print("\n\nfft:",X)
        # y[ind_traj,:,:]=np.transpose(X) #X is (nFreq,nNodes)
        # print("y:",y[ind_Traj,:,:])
    return x
##############################################################################################################################        

def compute_fft(data,nfft):
    nSamples,nNodes=data.shape
    nTrajectories=np.int32(nSamples/nfft) 
    y=np.zeros((nTrajectories,nNodes,nfft),dtype=complex)
    for ind in range(nTrajectories-1): # discard final few residual samples
        x=data[ind*(nfft):(ind+1)*nfft,:]
        X=np.fft.fft(x,axis=0)
        y[ind,:,:]=np.transpose(X) #X is (nfft,nNodes)
    return y


def generate_var_coeffs(Adj,DELAY=2,FILTER_LIST=None):
    nNodes=Adj.shape[0]
    if FILTER_LIST==None:
        FILTER_LIST=np.array([[1, .6,.3],[1, .5,.2],[1, .7,.4],[1, .5,.1],[1, .6,.2],[1, .7,.3],[1, .4,.1],[1, .7,.2] ])
    num_rows_to_select=(nNodes,nNodes)
    selected_indices = np.random.choice(FILTER_LIST.shape[0], num_rows_to_select, replace=True)
    B=FILTER_LIST[selected_indices]
    delta=0.5
    B_scale1=(np.random.uniform(-1,1,size=[nNodes,nNodes]))
    B_scale1=2*(B_scale1>0)-1
    B_scale=np.random.uniform(delta,1,size=[nNodes,nNodes])*B_scale1
    # Combine the two ranges to get the final result
    # B_scale = np.concatenate((B_scale1,B_scale2))
    np.random.shuffle(B_scale)
    B_scale=np.repeat(B_scale[:nNodes, :, np.newaxis], 3, axis=2)
    Adj1=np.repeat(Adj[:, :, np.newaxis], 3, axis=2)
    B=B*Adj1
    B=B*B_scale
    B[:,:,1:3]=0
    A=np.diag(np.random.rand(nNodes))
    return A,B

def compute_CPSD(psd_est,ind,C):
    phi_CC_inv=np.linalg.inv(psd_est(C,C))
    temp1=np.dot(psd_est(ind,C),phi_CC_inv)
    f_est=psd_est(ind,ind)-np.dot(np.dot(temp1,psd_est(C,ind)))
    return f_est

def compute_f_metric(psd,i,c):
    if c==None or c==[]:
        return psd[i,i]
    if len(c)==1:
        phi_C_inv=1/psd[c,c]
        temp=psd[i,c] * phi_C_inv * psd[c,i]
    else:
        phi_C_inv=np.linalg.inv(psd[c,:][:,c])
        temp=psd[i,c] @ phi_C_inv @ psd[c,i]
    # print(phi_C_inv,temp)
    return psd[i,i]-temp

def find_topological_order(psd,q):
    num_nodes=psd.shape[0]
    V=[ind for ind in range(num_nodes)]
    S=[]
    C=[]
    # for ind in range(num_nodes):
    psd_diag=psd.diagonal()
    min_val = np.min(psd_diag)
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

        fvals=10*np.ones([num_nodes,len(combination_set)],dtype=np.complex128)
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

def compute_coeffs(Adj,DELAY=2,file_AB=None):
    A1,B1=generate_var_coeffs(Adj,DELAY=DELAY)
    # np.savez(file_AB,A=A1,B=B1.reshape(B1.shape[0], -1))
    return A1,B1


def theoretical_analysis(B,omega=0*np.pi/64):
    num_nodes=B.shape[0]
    psd_theo=np.zeros([num_nodes,num_nodes,len(omega)])
    for ind in range(len(omega)):
        omega1=omega[ind]
        z = sym.Symbol('z')
        A=np.eye(B.shape[0])
        den=sym.Matrix(A) # A[:,0]+A[:,1]/z+A[:,2]/z**2
        num=sym.Matrix(B[:,:,0])+sym.Matrix(B[:,:,1])/z+sym.Matrix(B[:,:,2])/z**2
        
        # k=1
        # num=num.subs(z,np.exp(k*1j*np.pi/64))
        # den=den.subs(z,np.exp(k*1j*np.pi/64))

        H = sym.Matrix(num) #num/den[:,None]
        # print(H.shape)
        # print(H)
        # H1=H.subs(z,np.exp(0*1j*np.pi/64))
        # I_H=sym.Matrix(sym.eye(A.shape[0])-H)
        # I_H_star = np.transpose(np.conjugate(I_H))
        # psd_theo = I_H_star @ I_H
        H1=H.subs(z,sym.exp(1j*omega1))
        I_H=(sym.eye(A.shape[0])-H1)
        I_H = np.array(I_H.expand(complex=True)).astype(np.complex128)
        I_H_inv=np.linalg.inv(I_H)
        I_H_inv_star = np.transpose(np.conjugate(I_H_inv))
        psd_theo1 = I_H_inv @ I_H_inv_star
    # psd_theo=psd_theo.expand(complex=True)
    # psd_theo_val=psd_theo.subs(z,np.exp(1j*np.pi/64))
    # psd_mat = sym.matrices.dense.matrix2numpy(psd_theo)
    # print(psd_theo.diagonal())
    # print(sym.matrix2numpy(psd_theo[0,2]))
        psd_theo[:,:,ind]=psd_theo1
    return psd_theo
    
def find_parents(psd,ordered_set,C,gamma):
    num_nodes=len(ordered_set)
    Pa=[]
    for ind in range(num_nodes):
        node=ordered_set[ind]
        # if node==2:
        #     print("node :2")
        parents=[]
        for j in C[ind]:
            P=list(C[ind])
            P.remove(j)
            diff_j=compute_f_metric(psd,node,P)-compute_f_metric(psd,node,C[ind])
            if diff_j>gamma:
                parents.append(j)
            # else:
            #     parents.remove(j)
        Pa.append(parents)
        # print("Node ",node,", Parents: ",Pa[ind])
    return Pa

def estimate_adj_from_parents(ordered_set,parents):
    num_nodes=len(ordered_set)
    Adj=np.zeros([num_nodes,num_nodes],dtype=bool)
    for ind in range(num_nodes):
        Adj[ordered_set[ind],parents[ind]]=True
    return Adj
