import numpy as np
import networkx as nx
import netgraph

# went to settings.json in command+shift+P and added the line ,
#          "python.defaultInterpreterPath": "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"


import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def plot_topology_6nodes(adjacency_matrix):
    # import pygraphviz as pgv
    nNodes=adjacency_matrix.shape[0]
    G=nx.Graph()
    G.add_nodes_from(['X'+str(ind) for ind in range(1,nNodes+1)])
    # G.add_nodes_from([(2,{"color":"red"}),(3,{"color":"green"})])
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            if adjacency_matrix[ind1,ind2]:
                # print(ind1,ind2)
                G.add_edge('X'+str(ind1+1),'X'+str(ind2+1))
# explicitly set positions
    pos = {'X'+str(1): (-1, 0.3), 'X'+str(2): (0, 0.05), 
           'X'+str(3): (1.7, 0.22), 'X'+str(4): (5, 0.285), 
           'X'+str(5): (4.5, 0.03), 'X'+str(6): (8, 0.00)}
    options = {
    "font_size": 26,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
    "arrowsize" : 35
    }
    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.001)

    # nx.draw_networkx(G, with_labels=True, 
    #                     font_weight='bold', 
    #                     font_size=30, node_size=4000, arrowsize=14)


def plot_directed_graph_6nodes(adjacency_matrix):
    # import pygraphviz as pgv
    nNodes=adjacency_matrix.shape[0]
    G=nx.DiGraph()
    G.add_nodes_from(['X'+str(ind) for ind in range(1,nNodes+1)])
    # G.add_nodes_from([(2,{"color":"red"}),(3,{"color":"green"})])
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            if adjacency_matrix[ind1,ind2]:
                # print(ind1,ind2)
                G.add_edge('X'+str(ind2+1),'X'+str(ind1+1))
    # explicitly set positions
    pos = {'X'+str(1): (-1, 0.3), 'X'+str(2): (0, 0), 
           'X'+str(3): (1.7, 0.22), 'X'+str(4): (4, 0.255), 
           'X'+str(5): (4.5, 0.03), 'X'+str(6): (8, 0.13)}
    options = {
    "font_size": 26,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
    "arrowsize" : 35
        }

    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.001)

def phase_shift(w):
    # Shifts the phase from [-pi,pi] to [0,2pi]
    # receives a vector of phases and returns the modified phase
    for ind in range(w.shape[0]):
        # print(w.shape)
        # w_mod=w[ind]
        if w[ind]<0:
            w[ind]=2*np.pi+w[ind]
    return w
    # print(fig.get_axes())
    # ax.plot(w_mod)
    # ax.axis([0, nSamples, 0, 6.3])


def plot_phase_response(W,file=None):
# plot the phase response of complex matrix W
    n1=W.shape[0];  n2=W.shape[2]
    nSamples=W.shape[1]
    fig1,ax = plt.subplots(n1,n2-1)
    fig1.tight_layout()
    yvals=np.arange(0,2,.5)
    phase_val=yvals*np.pi
    idx=[ind for ind in range(0,np.size(yvals))]
    y_N=[ r""+eval("\"$"+str(yvals[ind])+"\pi$"+"\"") for ind in range(0,np.size(yvals))]
    y_N[0]=0; y_N[1]=r""+eval("\"$\pi/2$"+"\""); y_N[2]=r""+eval("\"$\pi$"+"\""); y_N[3]=r""+eval("\"$3\pi/2$"+"\"")
    
    xvals=np.arange(nSamples)
    omega_N=[ r""+eval("\"$f_{"+str(xvals[ind])+"}$"+"\"") for ind in range(0,np.size(xvals),15)]
    xidx=[ind for ind in range(0,np.size(xvals),15)]

    for ind1 in range(n1):
        indc=0
        for ind2 in range(n2):
            if ind1!=ind2:
                ax[ind1,indc].plot(phase_shift(np.angle(W[ind1,:,ind2])))
                if (ind1==1 and ind2==3) or (ind1==2 and ind2==3) or (ind1==3 and ind2==1) or (ind1==3 and ind2==2):
                    ax[ind1,indc].plot(phase_shift(np.angle(W[ind1,:,ind2])),color="red")
                ax[ind1,indc].axis([0, nSamples, 0, 6.3])
                # ax[ind1,indc].set_title(r'$\\angle$'+eval('\'$\angle W_'+str(ind1+1)+'$\'')+"["+str(ind2+1)+"]", fontsize=12, ha='center')
                # ax[ind1,indc].set_title(r' '+str(ind1+1)+"["+str(ind2+1)+"]", fontsize=12, ha='center')
                # ax[ind1,indc].set_title(r"$ \angle$"+eval("\"$W_"+str(ind1+1)+"$"+"\"")+"["+str(ind2+1)+"]", fontsize=12, ha='center')
                ax[ind1,indc].set_title(r"$ \angle$"+eval("\"$W_"+str(ind1+1)+"$"+"\"")+"["+str(ind2+1)+"]", fontsize=12, ha='center')
                plt.setp(ax,xticks=xvals[xidx], xticklabels=omega_N,yticks=phase_val[idx],yticklabels=y_N)
                ax[ind1,indc].set_xlabel('DFT points')
                sigma='%.2f'%(np.std(phase_shift(np.angle(W[ind1,:,ind2]))))
                print("\n",np.mean(phase_shift(np.angle(W[ind1,:,ind2]))))
                ax[ind1,indc].text(45, 7.3, r'($\sigma:$'+str(sigma)+')',color='red', fontsize=12)
                indc=indc+1
            # plt.figtext(0.15,0.95,'Phase response: (X1,X2)', fontsize=18, ha='center')
    plt.get_current_fig_manager().resize(2700,1700)
    # if file!=None:
        # plt.savefig(file,format='pdf')
    plt.show(block=False)

def plot_histogram(d,colr='#0504aa',bin='auto',title='My Very Own Histogram'):
    # h, xedges, yedges,image = plt.hist2d(x=np.real(d),y=np.imag(d), bins=bin, color=colr)
    n, bins, patches = plt.hist(x=d, bins=bin, color=colr,
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title(title, fontsize=10)
    # plt.text(23, 45, r'$\mu=15, b=3$')
    # maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    # print(np.amin(d))
    # plt.xlim(left=-25,right=np.maximum(30,np.amax(abs(d))))
    # plt.ylim(left=-25,right=np.maximum(30,np.amax(abs(d))))



def plot_directed_interactive_graph(adjacency_matrix):
    # import pygraphviz as pgv
    nNodes=adjacency_matrix.shape[0]
    G=nx.DiGraph()
    # G.add_nodes_from(['X'+str(ind) for ind in range(1,nNodes+1)])
    nodes=[ind for ind in range(1,nNodes+1)]
    # pos = np.array([[0, 0], [-2, 1], [2, 1], [-3, 2], [-1, 2], [1, 2], [3, 2]])
    G.add_nodes_from(nodes)
    # G.add_nodes_from([(2,{"color":"red"}),(3,{"color":"green"})])
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            if adjacency_matrix[ind1,ind2]:
                # print(ind1,ind2)
                G.add_edge(ind2+1,ind1+1)
    I = netgraph.InteractiveGraph(G,
                            #   node_positions=dict(zip(nodes, pos)),
                            #   node_labels=dict(zip(nodes,nodes)),
                            #   node_label_bbox=dict(fc="lightgreen", ec="black", boxstyle="square", lw=3),
                              node_size=4,node_labels=True,arrows=True, edge_width=1
    )
    # Set margins for the axes so that nodes aren't clipped
    plt.get_current_fig_manager().resize(2700,1500)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show(block=True)
    

def plot_topology(adjacency_matrix):
    # import pygraphviz as pgv
    nNodes=adjacency_matrix.shape[0]
    G=nx.Graph()
    G.add_nodes_from([(ind) for ind in range(1,nNodes+1)])
    # G.add_nodes_from([(2,{"color":"red"}),(3,{"color":"green"})])
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            if adjacency_matrix[ind1,ind2]:
                # print(ind1,ind2)
                G.add_edge((ind2+1),(ind1+1))
    # explicitly set positions
    # pos = {'X'+str(1): (-1, 0.3), 'X'+str(2): (0, 0), 
    #        'X'+str(3): (1.7, 0.22), 'X'+str(4): (4, 0.255), 
    #        'X'+str(5): (4.5, 0.03), 'X'+str(6): (8, 0.13)}
    options = {
    "font_size": 10,
    "node_size": 500,
    # "node_color": "white",
    # "edgecolors": "black",
    # "linewidths": 5,
    # "width": 5,
    # "arrowsize" : 15
        }

    nx.draw_networkx(G, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    # plt.show(block=False)
    # plt.pause(0.001)
    # nx.draw_networkx(G, with_labels=True, 
    #                     font_weight='bold', 
    #                     font_size=30, node_size=4000, arrowsize=14)


def plot_directed_graph_32nodes(adjacency_matrix):
    nNodes=adjacency_matrix.shape[0]
    G=nx.DiGraph()
    G.add_nodes_from([ind for ind in range(1,nNodes+1)])
    # G.add_nodes_from([(2,{"color":"red"}),(3,{"color":"green"})])
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            if adjacency_matrix[ind1,ind2]:
                # print(ind1,ind2)
                G.add_edge((ind2+1),(ind1+1))
    # explicitly set positions
    pos = {(1): (-1, 0.3), (2): (0, 0), 
           (3): (1.7, 0.22), (4): (4, 0.255), 
           (5): (4.5, 0.03), (6): (8, 0.13)}
    options = {
    "font_size": 26,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
    "arrowsize" : 35
        }

    nx.draw_networkx(G, pos, **options)
    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.001)
