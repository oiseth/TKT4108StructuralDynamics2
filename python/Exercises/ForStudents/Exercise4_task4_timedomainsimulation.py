# -*- coding: utf-8 -*-
"""

"""

#%% Import necessary modules and packages
import numpy as np
from scipy import linalg
from scipy import signal
from matplotlib import pyplot as plt

import time_integration as ti
import time_simulation as ts

#%% Establish system

plt.close('all')

L=3
I=60e-6
E=210e9
k=12*E*I/L**3
m=20e3

M=np.diag([1.2*m,m])
K=np.array([ [2.2,-1.2] , [-1.2, 1.2] ])*k

C=0.1155*M+0.0024*K

(lambd,phi)=linalg.eigh(K,M)

phi=np.array(phi)
lambd=np.array(lambd)
omega_n=np.sqrt(lambd)

#%% Monte carlo simulation

# Frequency axis for load
d_omega=.1
omega_load=np.arange(0,80+d_omega,d_omega)

# Spectrum for load
Spp=np.zeros((len(omega_load),2,2))
for k in np.arange(0,len(omega_load)):
            
    S_p2p2=1e4*(np.exp(-(omega_load[k]+1)**0.5)+1e-3)
    S_p1p1=S_p2p2*np.exp(-0.8)
    Co=0.5*(1-1/(1+np.exp(-0.1*(omega_load[k]-15))))
    S_p1p2=np.sqrt(S_p1p1*S_p2p2*Co)
    
    Spp[k,:,:]=np.array([ [S_p1p1,S_p1p2] , [S_p1p2, S_p2p2] ])
            
 
# Simulate the load
Nsim=1    
(t,p,P)=ts.MCCholesky(omega_load,Spp,Nsim,0.001)

# Cut to one hour duration
index_cut=np.argmin(abs(t-3600))
t=t[0:index_cut]
p=p[:,0:index_cut]

dt=t[1]-t[0]
T=t[-1]

# Plot time series
plt.figure()
plt.plot(t,p[0,:])
plt.plot(t,p[1,:])
plt.show()
plt.xlabel('Time [s]')
plt.ylabel('Load [N]')
plt.grid()

#%% Spectrum of load

plt.close('all')

(f,Gp)=ts.fft_function(p[0,:],dt)

S_p=np.multiply(Gp,np.conj(Gp))*T

plt.figure()
plt.plot(f,S_p)
plt.show()
plt.xlabel('f [Hz]')
plt.ylabel('S(omega) [N^2/Hz]')
plt.grid()
plt.title('PSD of Load')
plt.xlim(-25,25)

#plt.yscale('log')

#%% Solve system response by time integration

#plt.close('all')

# Initial conditions
u0=np.zeros((2,1))
udot0=np.zeros((2,1))

# Newmark parameters
gamma=0.5
beta=0.25
    
# Time integration
(u,udot,uddot)=ti.linear_newmark_krenk(M,C,K,p,u0,udot0,dt,gamma,beta)

# Plot time series
plt.figure()
plt.plot(t,u[0,:])
plt.plot(t,u[1,:])

plt.show()
plt.xlabel('Time [s]')
plt.ylabel('Response [m]')
plt.grid()

#%% Spectrum of response

#plt.close('all')

# Find DFT
(f,GU1)=ts.fft_function(u[0,:],dt)
(f,GU2)=ts.fft_function(u[1,:],dt)

# Find spectrum
S_U1=np.real( np.multiply(GU1,np.conj(GU1))*T )
S_U2=np.real( np.multiply(GU2,np.conj(GU2))*T )

plt.figure()
plt.plot(f,S_U1)
plt.plot(f,S_U2)
plt.show()
plt.xlabel('f [Hz]')
plt.ylabel('S(omega) [m^2/Hz]')
plt.grid()
plt.title('Spectrum of response')
plt.xlim(-10,10)
plt.yscale('log')

#%% Distribution of peaks

#plt.close('all')

# Find peaks
height=0
distance=np.round(0.1/dt)
(index_peak,properties)=signal.find_peaks(u[1,:],height,None,distance)
N_peak=len(index_peak)    

# Plot time series with peaks
plt.figure()
plt.plot(t,u[1,:])
plt.plot(t[index_peak],u[1,index_peak], 'o', color='red');
plt.show()
plt.xlabel('Time [s]')
plt.ylabel('Response [m]')
plt.grid()

# Plot histogram of peaks
plt.figure()
weights=np.ones_like(u[0,index_peak])/float(len(u[0,index_peak]))
plt.hist(u[0,index_peak],20,None,True,None,False,None) # Normalized version
# plt.hist(u[0,index_peak],20,None,False,weights=weights) # Relative frequency version
plt.show()
plt.title('Histogram of peaks (normalized)')
# plt.yscale('log')


