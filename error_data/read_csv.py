import csv 
import numpy as np
import matplotlib.pyplot as plt


def plot_fixed_num_nodes_variable_q(num_nodes,max_in_degree_list):
    avg_error_list=list()
    prob_error_list=list()
    for ind in range(len(max_in_degree_list)):
        max_in_degree=max_in_degree_list[ind]
        file_loc="/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/sample_complexity"
        file_name=file_loc+"/reconstruction_error_p="+str(num_nodes)+"_q="+str(max_in_degree)+".csv"
        
        with open(file_name, 'r', newline='\n') as csvfile:
            csvreader=csv.reader(csvfile, delimiter=',')
            rows=list()
            for lines in csvreader:
                rows.append(lines)
        omega=[float(i) for i in rows[0]]
        Samples=[int(i) for i in rows[len(rows)-1]]
        avg_error=np.zeros([len(Samples),len(omega)])
        prob_error=np.zeros([len(Samples),len(omega)])
        for ind1 in range(2,2+len(Samples)):
            avg_error[ind1-2,:]=[float(i) for i in rows[ind1]]
        for ind1 in range(9,9+len(Samples)):
            prob_error[ind1-9,:]=[float(i) for i in rows[ind1]]
        avg_error_list.append(avg_error[:,4])
        prob_error_list.append(prob_error[:,4])
    return Samples,avg_error_list,prob_error_list


def plot_fixed_q_variable_num_nodes(num_nodes_list,max_in_degree):
    avg_error_list=list()
    prob_error_list=list()
    for ind in range(len(num_nodes_list)):
        num_nodes=num_nodes_list[ind]
        file_loc="/Users/mishfad/Google Drive/My Drive/UMN/PhD/Simulations/simulation-python/sample_complexity"
        file_name=file_loc+"/reconstruction_error_p="+str(num_nodes)+"_q="+str(max_in_degree)+".csv"
        
        with open(file_name, 'r', newline='\n') as csvfile:
            csvreader=csv.reader(csvfile, delimiter=',')
            rows=list()
            for lines in csvreader:
                rows.append(lines)
        omega=[float(i) for i in rows[0]]
        Samples=[int(i) for i in rows[len(rows)-1]]
        avg_error=np.zeros([len(Samples),len(omega)])
        prob_error=np.zeros([len(Samples),len(omega)])
        for ind1 in range(2,2+len(Samples)):
            avg_error[ind1-2,:]=[float(i) for i in rows[ind1]]
        for ind1 in range(9,9+len(Samples)):
            prob_error[ind1-9,:]=[float(i) for i in rows[ind1]]
        avg_error_list.append(avg_error[:,4])
        prob_error_list.append(prob_error[:,4])
    return Samples,avg_error_list,prob_error_list




num_nodes_list=[10,15]
max_in_degree_list=[3]

# Samples,avg_error_list,prob_error_list=plot_fixed_num_nodes_variable_q(num_nodes_list[0],max_in_degree_list)
# plt.figure()
# for ind in range(len(max_in_degree_list)):
#     plt.plot(Samples,avg_error_list[ind],label="q="+str(max_in_degree_list[ind]))
# plt.legend(loc="upper right")
# plt.title("Comparison for p=10")
# plt.xlabel("Number of samples")
# plt.ylabel("Reconstruction error (FA+MD)")


Samples,avg_error_list,prob_error_list=plot_fixed_q_variable_num_nodes(num_nodes_list,max_in_degree_list[1])
plt.figure()
for ind in range(len(num_nodes_list)):
    plt.plot(Samples,avg_error_list[ind],label="p="+str(num_nodes_list[ind]))
plt.legend(loc="upper right")
plt.title("Comparison for q=3")
plt.xlabel("Number of samples")
plt.ylabel("Reconstruction error (FA+MD)")


plt.show()





# for ind in range(len(max_in_degree_list)):
#     plt.plot(Samples,avg_error_list[ind],label="q="+str(max_in_degree_list[ind]))
