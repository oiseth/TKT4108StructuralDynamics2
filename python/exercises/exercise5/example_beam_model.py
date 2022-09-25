# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:44:53 2022

@author: oiseth
"""

import numpy as np
from scipy import linalg as spla
from matplotlib import pyplot as plt
#%%

def beam_element(E,A,I,rho,x1,x2):
    """
    

    Parameters
    ----------
    E : float
        Modulus of elasticity
    A : flat
        section area
    I : float
        second moment of area
    rho : float
        material density
    x1 : float
        x,y coordinates of node 1
    x2 : float
        a,y coordinates of node 2

    Returns
    -------
    k_global : float
        element stiffness matrix is global coordinates
    m_global : float
        DESCRIPTION.
        element mass matrix in global coordinates

    """
    L = ((x2-x1) @ (x2-x1))**0.5
    
    EA = E*A
    EI = E*I
    
    k_local = np.array([[EA/L, 0, 0, -EA/L, 0, 0],
                        [0, 12*EI/L**3, -6*EI/L**2, 0, -12*EI/L**3, -6*EI/L**2 ],
                        [0, -6*EI/L**2, 4*EI/L, 0, 6*EI/L**2, 2*EI/L],
                        [-EA/L, 0, 0, EA/L, 0, 0],
                        [0, -12*EI/L**3, 6*EI/L**2, 0, 12*EI/L**3, 6*EI/L**2],
                        [0, -6*EI/L**2, 2*EI/L, 0, 6*EI/L**2, 4*EI/L]])
    
    m_local = rho*A*L/420*np.array([[140, 0, 0, 70, 0, 0],
                        [0, 156, -22*L, 0, 54, 13*L],
                        [0, -22*L, 4*L**2, 0, -13*L, -3*L**2],
                        [70, 0, 0, 140, 0, 0],
                        [0, 54, -13*L, 0, 156, 22*L],
                        [0, 13*L, -3*L**2, 0, 22*L, 4*L**2]])
    
    e1 = (x2-x1)/L    
    e2 = np.cross(np.array([0, 0, 1]),np.append(e1,0))
    e2 = e2[0:-1]
    

    T_glob2loc = np.vstack((e1,e2))
    T_glob2loc = spla.block_diag(T_glob2loc,1.0)
    T_glob2loc_element = spla.block_diag(T_glob2loc,T_glob2loc)
    
     
    k_global = T_glob2loc_element.T @ k_local @ T_glob2loc_element
    m_global = T_glob2loc_element.T @ m_local @ T_glob2loc_element
    
    return k_global, m_global



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
[15,	13,	1]],dtype=float)

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

def plot_fe_model(nodes,elements):
    
    hor_size = np.max(nodes[:,1]) - np.min(nodes[:,1])
    hor_mid = 1/2*(np.max(nodes[:,1]) + np.min(nodes[:,1]))
    
    vert_size = np.max(nodes[:,2]) - np.min(nodes[:,2])
    vert_mid = 1/2*(np.max(nodes[:,2]) + np.min(nodes[:,2]))
    
    max_dim = np.max([hor_size,vert_size])*1.1
    
    plt.figure()
    plt.show()
    plt.plot(nodes[:,1],nodes[:,2],"o")
    
    for k in range(elements.shape[0]):
        x1 = [nodes[nodes[:,0]==elements[k,1],1],nodes[nodes[:,0]==elements[k,2],1] ]
        x2 = [nodes[nodes[:,0]==elements[k,1],2],nodes[nodes[:,0]==elements[k,2],2] ]
        
        plt.plot(x1,x2)
    plt.xlim([hor_mid-max_dim/2, hor_mid+max_dim/2])
    plt.ylim([vert_mid-max_dim/2, vert_mid+max_dim/2])
    plt.grid()

plot_fe_model(nodes,elements)
#%% assembly element model
E = 1.0e10
A = 0.2*0.2
I = 1/12*0.2*0.2**3
rho = 3000

dofs_in_nodes = 3

def assembly(nodes,elements,dofs_in_nodes,E,A,I,rho):
    mass_matrix = np.zeros((nodes.shape[0]*dofs_in_nodes,nodes.shape[0]*dofs_in_nodes))
    stiffness_matrix = np.zeros((nodes.shape[0]*dofs_in_nodes,nodes.shape[0]*dofs_in_nodes)) 
    
    for k in range(elements.shape[0]):
        node_index1 = np.where(nodes[:,0]==elements[k,1])[0][0]
        node_index2 = np.where(nodes[:,0]==elements[k,2])[0][0]   
        
        x1 = nodes[node_index1,1:]
        x2 = nodes[node_index2,1:]    
        
        k_global, m_global = beam_element(E,A,I,rho,x1,x2)
        
        stiffness_matrix[dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1),dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1)] = stiffness_matrix[dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1),dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1)] + k_global[0:dofs_in_nodes,0:dofs_in_nodes]
        stiffness_matrix[dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1),dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1)] = stiffness_matrix[dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1),dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1)] + k_global[0:dofs_in_nodes,dofs_in_nodes:] 
        stiffness_matrix[dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1),dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1)] = stiffness_matrix[dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1),dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1)] + k_global[dofs_in_nodes:,0:dofs_in_nodes] 
        stiffness_matrix[dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1),dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1)] = stiffness_matrix[dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1),dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1)] + k_global[dofs_in_nodes:,dofs_in_nodes:] 
        
        mass_matrix[dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1),dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1)] = mass_matrix[dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1),dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1)] + m_global[0:dofs_in_nodes,0:dofs_in_nodes]
        mass_matrix[dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1),dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1)] = mass_matrix[dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1),dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1)] + m_global[0:dofs_in_nodes,dofs_in_nodes:] 
        mass_matrix[dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1),dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1)] = mass_matrix[dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1),dofs_in_nodes*node_index1:dofs_in_nodes*(node_index1+1)] + m_global[dofs_in_nodes:,0:dofs_in_nodes] 
        mass_matrix[dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1),dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1)] = mass_matrix[dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1),dofs_in_nodes*node_index2:dofs_in_nodes*(node_index2+1)] + m_global[dofs_in_nodes:,dofs_in_nodes:] 
        
    return stiffness_matrix, mass_matrix



stiffness_matrix, mass_matrix = assembly(nodes,elements,dofs_in_nodes,E,A,I,rho)
#%% boundary conditions
T_BC = np.eye(nodes.shape[0]*dofs_in_nodes)
T_BC = np.delete(T_BC,[0, 1 , dofs_in_nodes*7+1],axis=1)

stiffness_matrix_bc = T_BC.T @ stiffness_matrix @ T_BC
mass_matrix_bc = T_BC.T @ mass_matrix @ T_BC

#%% natural frequencies and modes
lam,vec = spla.eig(stiffness_matrix_bc,mass_matrix_bc)
indx = np.argsort(lam)
lam = lam[indx]

vec = vec[:,indx]

f = np.real(lam**0.5)/2/np.pi

#%% plot modes

u = T_BC @ vec[:,1]

skd = 2

def plot_deformed_model(nodes,elements,u,skd):
    
    hor_size = np.max(nodes[:,1]) - np.min(nodes[:,1])
    hor_mid = 1/2*(np.max(nodes[:,1]) + np.min(nodes[:,1]))
    
    vert_size = np.max(nodes[:,2]) - np.min(nodes[:,2])
    vert_mid = 1/2*(np.max(nodes[:,2]) + np.min(nodes[:,2]))
    
    max_dim = np.max([hor_size,vert_size])*1.1

    nodes_deformed = np.copy(nodes)
    nodes_deformed[:,1] = nodes_deformed[:,1]+skd*u[0::3]
    nodes_deformed[:,2] = nodes_deformed[:,2]+skd*u[1::3]
    plt.figure()
    for k in range(elements.shape[0]):
        print(k)
        x1 = [nodes_deformed[nodes_deformed[:,0]==elements[k,1],1],nodes_deformed[nodes_deformed[:,0]==elements[k,2],1] ]
        x2 = [nodes_deformed[nodes_deformed[:,0]==elements[k,1],2],nodes_deformed[nodes_deformed[:,0]==elements[k,2],2] ]
        
        plt.plot(x1,x2)
    plt.plot(nodes[:,1]+skd*u[0::3],nodes[:,2]+skd*u[1::3],"o")
    plt.plot(nodes_deformed[:,1],nodes_deformed[:,2],"o")
    
    plt.xlim([hor_mid-max_dim/2, hor_mid+max_dim/2])
    plt.ylim([vert_mid-max_dim/2, vert_mid+max_dim/2])
    plt.grid()

plot_deformed_model(nodes,elements,u,skd)

#%%

element_refine_factor = 10

def refine_mesh(nodes,elements,T_BC,element_refine_factor,dofs_in_nodes):

    nodes_add = np.zeros(((element_refine_factor-1)*elements.shape[0],3))    
    nodes_add[:,0] = np.arange(0,nodes_add.shape[0])+ 10000
    new_nodes_element = element_refine_factor-1
    
    refined_T_BC = spla.block_diag(T_BC,np.eye(nodes_add.shape[0]*dofs_in_nodes))
        
    refined_elements = np.zeros((element_refine_factor * elements.shape[0],3))    
    refined_elements[:,0] = np.arange(1,elements.shape[0]*element_refine_factor+1)    
    
    for k in range(elements.shape[0]):
        
        refined_elements[k*element_refine_factor:(k+1)*element_refine_factor,1] = np.hstack((elements[k,1], np.arange(k*new_nodes_element,(k+1)*new_nodes_element) +10000))
        refined_elements[k*element_refine_factor:(k+1)*element_refine_factor,2] = np.hstack(( np.arange(k*new_nodes_element,(k+1)*new_nodes_element) +10000, elements[k,2]))
        
        
        node_index1 = np.where(nodes[:,0]==elements[k,1])[0][0]
        node_index2 = np.where(nodes[:,0]==elements[k,2])[0][0] 
        
        x1 = nodes[node_index1,1:]
        x2 = nodes[node_index2,1:] 
        
        nodes_add[k*new_nodes_element:(k+1)*new_nodes_element,1] = np.linspace(x1[0],x2[0],element_refine_factor+1)[1:-1]
        nodes_add[k*new_nodes_element:(k+1)*new_nodes_element,2] = np.linspace(x1[1],x2[1],element_refine_factor+1)[1:-1]
    
    refined_nodes = np.vstack((nodes,nodes_add))
    
    return refined_nodes, refined_elements, refined_T_BC

refined_nodes, refined_elements, refined_T_BC = refine_mesh(nodes,elements,T_BC,element_refine_factor,dofs_in_nodes)
#%% Plot refined model

plot_fe_model(refined_nodes,refined_elements)

#%% Assembly new model
stiffness_matrix, mass_matrix = assembly(refined_nodes,refined_elements,dofs_in_nodes,E,A,I,rho)

stiffness_matrix_bc = refined_T_BC.T @ stiffness_matrix @ refined_T_BC
mass_matrix_bc = refined_T_BC.T @ mass_matrix @ refined_T_BC

#%% natural frequencies and modes new model
lam,vec = spla.eig(stiffness_matrix_bc,mass_matrix_bc)
indx = np.argsort(lam)
lam = lam[indx]

vec = vec[:,indx]

f = np.real(lam**0.5)/2/np.pi

#%% plot modes new model

u = refined_T_BC @ vec[:,0]

skd = 10

plot_deformed_model(refined_nodes,refined_elements,u,skd)
