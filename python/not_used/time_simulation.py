import numpy as np

def MCCholesky(omegaaxisinput,S,Nsim=1,domegasim=1e-3):
    """
    This function simulates realizations (time series) using the CPSD matrix S(N_var,N_var,N_omega)
    (NOT FULLY TESTED YET)
    
    Arguments
    ---------------------------
    omegaaxisinput : double
         frequency axis, positive frequencies only
    S : double
         CPSD matrix
    Nsim : double
         Number of simulations
    domegasim : double
         interpolation density

    Returns
    ---------------------------
    t : double
        time axis
    x : double
        time series with last simulation
    X : list
        list with all time series
        
    Reference: 
      
    """
    
    # Empty Cholesky matrix
    N_var=np.shape(S)
    N_var=N_var[2]
    G=np.zeros(np.shape(S),dtype=complex)

    # Cholesky decomposition, lower
    for n in np.arange(len(omegaaxisinput)):
        if np.any(S[n,:,:]):
            G[n,:,:]=np.linalg.cholesky(S[n,:,:])
            
    # Interpolate to fine omega axis
    omegaaxissim=np.arange(domegasim,np.max(omegaaxisinput)+domegasim,domegasim)
    
    # Next power of 2 for FFT
    NFFT=int(2**np.ceil(np.log2(2*len(omegaaxissim)))) 
    t=np.linspace(0,2*np.pi/domegasim,NFFT)
    
    # Precalculate the interpolated Cholesky matrix
    c_precalc=[None]*N_var
    
    for m in np.arange(0,N_var):
        c_precalc[m]=np.zeros([m+1,len(omegaaxissim)],dtype=complex)
        for n in np.arange(0,m+1):
            
            G_slice=G[:,m,n]
            
            # c_precalc is a list. In each list element, the number of rows are varying
            # because G is lower diagonal
            # one row for the first, two for the second, and so on
            c_precalc[m][n,:]=np.interp(omegaaxissim,omegaaxisinput,G_slice)
            
    # Simulate
    
    # List with all simulations
    X=[None]*Nsim        
    
    # Loop over all simulations
    for z in np.arange(0,Nsim):
        
        # Random numbers with size [N_var,N_freq]
        phi=2*np.pi*np.random.uniform(0,1,[N_var,len(omegaaxissim)])
        
        x=np.zeros([N_var,NFFT])
        
        for m in np.arange(0,N_var):
            
            c_all=np.multiply(c_precalc[m],np.exp(complex(0,1)*phi[np.arange(0,m+1),:]))
            
            # Sum over all [1 -> m] components (without loop)
            x[m,:]=np.sum( np.real(np.fft.ifft(c_all,NFFT,1)) ,0)*NFFT*np.sqrt(2*domegasim)
            
        # Save simulation to list
        X[z]=x
                
    return t,x,X

def fft_function(x,dt,axis=0):
    """
    This function gives the DFT
    
    Arguments
    ---------------------------
    x : double
         time series
    dt : double
         resolution of the time axis
    axis : double
         the axis on which the DFT operates 
         
    Returns
    ---------------------------
    f : double
        frequency axis
    G : double
        DFT (complex Fourier coefficients)

        
    Reference: 
      
    """
    
    n=np.shape(x)
    n_points=n[axis]
    
    G=np.fft.fftshift( np.fft.fft(x,None,axis) )/n_points
   
    # Sampling frequency
    Fs=1/np.double(dt)
    
    # Frequency axis
    f=Fs/2*np.linspace(-1,1,n_points)
    
    return f,G

def ifft_function(G,Fs,axis=0):
    """
    This function gives the IDFT
    
    Arguments
    ---------------------------
    G : double
         DFT (complex Fourier coefficients)
    Fs : double
         sample rate, maximum frequency of G times 2 (=F_nyquist*2)
    axis : double
         the axis on which the IDFT operates 
         
    Returns
    ---------------------------
    t : double
        time axis
    x : double
        time series

    Reference: 
      
    """
    
    G=np.atleast_2d(G)
    n=np.shape(G)
    
    n_points=n[axis]
    G=np.fft.ifft( np.fft.ifftshift(G,None,axis) )/n_points
   
    dt=1/np.double(Fs)
    t=np.arange(0,dt*n_points,dt)
    
    return t,x