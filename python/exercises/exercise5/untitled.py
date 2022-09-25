import numpy as np
from scipy import linalg as spla
from matplotlib import pyplot as plt
#%% define elements
def bar_element(E,A,rho,x1,x2):
    L = ((x2-x1) @ (x2-x1))**0.5
        
    k_local = E*A/L*np.array([[1, 0, -1, 0],[0, 0, 0, 0],[-1, 0, 1, 0],[0, 0, 0, 0]])
    m_local = rho*A*L/6*np.array([[2, 0, 1, 0],[0, 2, 0, 1],[1, 0, 2, 0],[0, 1, 0, 2]])
    
    e1 = (x2-x1)/L    
    e2 = np.cross(np.array([0, 0, 1]),np.append(e1,0))
    e2 = e2[0:-1]
    

    T_glob2loc = np.vstack((e1,e2))
    T_glob2loc_element = spla.block_diag(T_glob2loc,T_glob2loc)
    
     
    k_global = T_glob2loc_element.T @ k_local @ T_glob2loc_element
    m_global = T_glob2loc_element.T @ m_local @ T_glob2loc_element
    
    return L, k_global, m_global


#%% define nodes and elements

nodes = np.array([[1,	0,	0],
[2,	2,	0],
[3,	4,	0],
[4,	6,	0],
[5,	8,	0],
[6,	10,	0],
[7,	12,	0],
[8,	14,	0],
[9,	1,	1],
[10,	3,	1],
[11,	5,	1],
[12,	7,	1],
[13,	9,	1],
[14,	11,	1],
[15,	13,	1]])

elements = np.array([[1,	1,	2,	1],
[2,	2,	3,	1],
[3,	3,	4,	1],
[4,	4,	5,	1],
[5,	5,	6,	1],
[6,	6,	7,	1],
[7,	7,	8,	1],
[8,	9,	10,	1],
[9,	10,	11,	1],
[10,	11,	12,	1],
[11,	12,	13,	1],
[12,	13,	14,	1],
[13,	14,	15,	1],
[14,	1,	9,	1],
[15,	9,	2,	1],
[16,	2,	10,	1],
[17,	10,	3,	1],
[18,	3,	11,	1],
[19,	11,	4,	1],
[20,	4,	12,	1],
[21,	12,	5,	1],
[22,	5,	13,	1],
[23,	13,	6,	1],
[24,	6,	14,	1],
[25,	14,	7,	1],
[26,	7,	15,	1],
[27,	15,	8,	1]])

#%% plot nodes and elements
plt.figure(figsize=(10,10))
plt.show()
plt.plot(nodes[:,1],nodes[:,2],"o")

for k in range(elements.shape[0]):
    print(k)
    x1 = [nodes[nodes[:,0]==elements[k,1],1],nodes[nodes[:,0]==elements[k,2],1] ]
    x2 = [nodes[nodes[:,0]==elements[k,1],2],nodes[nodes[:,0]==elements[k,2],2] ]
    
    plt.plot(x1,x2)
sk = 8    
plt.xlim([0,2*sk])
plt.ylim([-1*sk,1*sk])
plt.grid()

#%% assembly bar element model
E = 1.0e10
A = 0.2*0.2
rho = 750
mass_matrix = np.zeros((nodes.shape[0]*2,nodes.shape[0]*2))
stiffness_matrix = np.zeros((nodes.shape[0]*2,nodes.shape[0]*2)) 

for k in range(elements.shape[0]):
    print(k)
    node_index1 = np.where(nodes[:,0]==elements[k,1])[0][0]
    node_index2 = np.where(nodes[:,0]==elements[k,2])[0][0]
    
    
    
    x1 = nodes[node_index1,1:]
    x2 = nodes[node_index2,1:]    
    
    L, k_global, m_global = bar_element(E,A,rho,x1,x2)
    
    stiffness_matrix[2*node_index1:2*(node_index1+1),2*node_index1:2*(node_index1+1)] = stiffness_matrix[2*node_index1:2*(node_index1+1),2*node_index1:2*(node_index1+1)] + k_global[0:2,0:2]
    stiffness_matrix[2*node_index1:2*(node_index1+1),2*node_index2:2*(node_index2+1)] = stiffness_matrix[2*node_index1:2*(node_index1+1),2*node_index2:2*(node_index2+1)] + k_global[0:2,2:] 
    stiffness_matrix[2*node_index2:2*(node_index2+1),2*node_index1:2*(node_index1+1)] = stiffness_matrix[2*node_index2:2*(node_index2+1),2*node_index1:2*(node_index1+1)] + k_global[2:,0:2] 
    stiffness_matrix[2*node_index2:2*(node_index2+1),2*node_index2:2*(node_index2+1)] = stiffness_matrix[2*node_index2:2*(node_index2+1),2*node_index2:2*(node_index2+1)] + k_global[2:,2:] 
    
    mass_matrix[2*node_index1:2*(node_index1+1),2*node_index1:2*(node_index1+1)] = mass_matrix[2*node_index1:2*(node_index1+1),2*node_index1:2*(node_index1+1)] + m_global[0:2,0:2]
    mass_matrix[2*node_index1:2*(node_index1+1),2*node_index2:2*(node_index2+1)] = mass_matrix[2*node_index1:2*(node_index1+1),2*node_index2:2*(node_index2+1)] + m_global[0:2,2:] 
    mass_matrix[2*node_index2:2*(node_index2+1),2*node_index1:2*(node_index1+1)] = mass_matrix[2*node_index2:2*(node_index2+1),2*node_index1:2*(node_index1+1)] + m_global[2:,0:2] 
    mass_matrix[2*node_index2:2*(node_index2+1),2*node_index2:2*(node_index2+1)] = mass_matrix[2*node_index2:2*(node_index2+1),2*node_index2:2*(node_index2+1)] + m_global[2:,2:] 
    
   
#%% boundary conditions
transform = np.eye(nodes.shape[0]*2)
transform = np.delete(transform,[0, 1 , 2*7+1],axis=1)

stiffness_matrix_bc = transform.T @ stiffness_matrix @ transform
mass_matrix_bc = transform.T @ mass_matrix @ transform
#%% natural frequencies and modes
lam,vec = spla.eig(stiffness_matrix_bc,mass_matrix_bc)
indx = np.argsort(lam)
lam = lam[indx]

vec = vec[:,indx]

f = np.real(lam**0.5)/2/np.pi

#%% plot modes

u = transform @ vec[:,0]

skd = 2.0

plt.figure()
plt.plot(nodes[:,1]+skd*u[0::2],nodes[:,2]+skd*u[1::2],"o")

sk = 8    
plt.xlim([0,2*sk])
plt.ylim([-1*sk,1*sk])
plt.grid()





