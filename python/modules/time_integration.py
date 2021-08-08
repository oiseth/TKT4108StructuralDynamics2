import numpy as np

def linear_newmark_krenk(M,C,K,f,u0,udot0,h,gamma,beta):
    """
    Return the solution of a system of second order differential equations obtained by Newmarks beta method  
    
    Arguments
    ---------------------------
    M : double
         Mass matrix
    C : double
         Damping matrix
    K : double
         Stiffness matrix
    f : double
         Load vectors
    u0 : double
         initial displacements
    udot0 : double
         initial velocity
    h : double
         time step size
    gamma : double
         parameter in the algorithm
    beta  : double
         parameter in the algorithm
         
        
    Returns
    ---------------------------
    u : double
        displacements
    udot : double
        velocities
    u2dot : double
        accelerations
        
    Reference: 
    Krenk, S. (2009). Non-linear modeling and analysis of solids and structures. Cambridge University Press.   
        
    """
    
    # Initialize variables
    u = np.zeros((M.shape[0],f.shape[1]))
    
    udot = np.zeros((M.shape[0],f.shape[1]))
    
    u2dot = np.zeros((M.shape[0],f.shape[1]))
    
    # Insert initial conditions in response vectors
    u[:,0] = u0[:,0]
    
    udot[:,0] = udot0[:,0]
    
    # Calculate "modified mass"
    Mstar = M + gamma*h*C + beta*h**2*K;
    
    # Calculate initial accelerations
    u2dot[:,0] = np.linalg.solve(M, f[:,0]-np.dot(C,udot[:,0])-np.dot(K,u[:,0])) 
    
    for n in range(0,f.shape[1]-1):
        
        #Predicion step
        
        udotstar_np1 = udot[:,n] + (1-gamma)*h*u2dot[:,n];
        
        ustar_np1 = u[:,n] + h*udot[:,n] + (1/2-beta)*h**2*u2dot[:,n];
        
        # Correction step
        
        u2dot[:,n+1] = np.linalg.solve(M, f[:,n+1]-np.dot(C,udotstar_np1)-np.dot(K,ustar_np1)) 
        
        udot[:,n+1] = udotstar_np1 + gamma*h*u2dot[:,n+1];
        
        u[:,n+1] = ustar_np1 + beta*h**2*u2dot[:,n+1];
    
    return u, udot, u2dot