import numpy as np

def loadspectrum(load_type):
    """
    This function gives three different load spectra, depending on the input
    
    Arguments
    ---------------------------
    load_type : string
         'a' or 'b' or 'c'

    Returns
    ---------------------------
    omega : double
        frequency axis
    Spp : double
        spectra of load 

        
    Reference: 
        
    """
    
    d_omega=.1
    omega=np.arange(0,130+d_omega,d_omega)
    sigma_load=3000
    
    Spp=np.zeros((len(omega),3,3))
    
    # Limits 
    if load_type=='a':
        omega_cutoff_low=0
        omega_cutoff_high=80
    elif load_type=='b':
        omega_cutoff_low=10
        omega_cutoff_high=40
    
    # Uniform band limited white noise
    if load_type=='a' or load_type=='b':

        S0=sigma_load**2/(omega_cutoff_high-omega_cutoff_low)
        
        for k in np.arange(0,len(omega)):
            
            Spp[k,:,:]=np.diag([S0,S0,S0])
    

    # Narrow banded load, bell shape from Guassian distribution
    if load_type=='c':
        
        mu=18
        sig=2
        
        for k in np.arange(0,len(omega)):
            
            S0=sigma_load**2*1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*(omega[k]-mu)**2/sig**2)
            Spp[k,:,:]=np.diag([S0,S0,S0])
    
        
    
    return omega, Spp