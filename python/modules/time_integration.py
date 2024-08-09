import numpy as np

def linear_newmark_krenk(M,C,K,f,u0,udot0,h,gamma=0.5,beta=0.25):
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
        
    gamma=0.5 and beta=0.25 set as default values
    This leads to the "average constant acceleration" integration shceme, which is unconditionally stable
    for all time steps. However, a time step (h) lower than T/20 is recommended, where T is the period
    of the highest natural frequency.
        
        
    """
    
    # If one dimensional, convert to matrix
    f=np.atleast_2d(f)
        
    # If scalar, convert to matrix
    M=np.atleast_2d(M)
    C=np.atleast_2d(C)
    K=np.atleast_2d(K)   
    
    u0=np.atleast_2d(u0)
    udot0=np.atleast_2d(udot0)
    
    # Check dimensions
    do_err=False

    if M.shape[0] != C.shape[0]:
        print('Mass and damping matrix not same size')
        do_err=True
        
    if M.shape[0] != K.shape[0]:
        print('Mass and stiffness matrix not same size')
        do_err=True
        
    if M.shape[0] != u0.shape[0]:
        print('Mass matrix and initial displacment not same size')
        do_err=True
        
    elif M.shape[0] != udot0.shape[0]:
        print('Mass matrix and initial velocity not same size')
        do_err=True
        
    elif M.shape[0] != f.shape[0]:
        print('Mass matrix and force not same size')
        do_err=True
        
    if do_err==True:
        print('Mass matrix dimension is ' + str(M.shape[0]))
        print('Damping matrix dimension is ' + str(C.shape[0]))
        print('Stiffness matrix dimension is ' + str(K.shape[0]))
        print('Initial displacment dimension is ' + str(u0.shape[0]))     
        print('Initial velocity dimension is ' + str(udot0.shape[0]))     
        print('Force dimension is ' + str(f.shape[0]))     
        raise Exception('Dimension error')


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
        
        u2dot[:,n+1] = np.linalg.solve(Mstar, f[:,n+1]-np.dot(C,udotstar_np1)-np.dot(K,ustar_np1)) 
        
        udot[:,n+1] = udotstar_np1 + gamma*h*u2dot[:,n+1];
        
        u[:,n+1] = ustar_np1 + beta*h**2*u2dot[:,n+1];
    
    return u, udot, u2dot

def linear_newmark_chopra(M,C,K,f,u0,udot0,h,gamma,beta):
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
    Chopra A. (2007). Dynamics of Structure. Table 5.4.2
    """
    
    # Initialize variables
    u = np.zeros((M.shape[0],f.shape[1]))
    
    udot = np.zeros((M.shape[0],f.shape[1]))
    
    u2dot = np.zeros((M.shape[0],f.shape[1]))
    
    # Insert initial conditions in response vectors
    u[:,0] = u0[:,0]
    
    udot[:,0] = udot0[:,0]
    
    # Calculate "modified mass"
    Khat = K + gamma/(beta*h)*C + M*1/(beta*h**2)
    
    # Calculate initial accelerations
    u2dot[:,0] = np.linalg.solve(M, f[:,0]-np.dot(C,udot[:,0])-np.dot(K,u[:,0])) 
    
    a = M/(beta*h) + gamma/beta*C;                      
    b = 0.5*M/beta + h*(0.5*gamma/beta - 1)*C;
    P=f
    
    for n in range(0,f.shape[1]-1):
        
        #Predicion step
        
        dP=P[:,n+1]-P[:,n]+np.dot(a,udot[:,n])+np.dot(b,u2dot[:,n])
        
        du_n=np.linalg.solve(Khat,dP)
        
        dudot_n = gamma/(beta*h)*du_n - gamma/beta*udot[:,n] + h*(1-0.5*gamma/beta)*u2dot[:,n]
        du2dot_n = 1/(beta*h**2)*du_n - 1/(beta*h)*udot[:,n] - 0.5/beta*u2dot[:,n]
        u[:,n+1] = du_n + u[:,n]
        udot[:,n+1] = dudot_n + udot[:,n]
        u2dot[:,n+1] = du2dot_n + u2dot[:,n]
        
    return u, udot, u2dot