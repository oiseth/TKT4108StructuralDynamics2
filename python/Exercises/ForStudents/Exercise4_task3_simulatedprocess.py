# -*- coding: utf-8 -*-
"""

"""

#%% Import necessary modules and packages
import numpy as np
from scipy import signal
from scipy import stats

from matplotlib import pyplot as plt

import time_integration as ti
import time_simulation as ts

#%% Create spectrum of process

plt.close('all')

omega_input=np.arange(0,80+1e-2,1e-2)
S0=0.01

S=np.zeros((len(omega_input),1,1))
for k in range(0,len(omega_input)):
    if omega_input[k]>=6 and omega_input[k]<=8:
        S[k,0,0]=S0
    elif omega_input[k]<6 or omega_input[k]<16:
        S[k,0,0]=S0/1000;
            
plt.figure()
plt.plot(omega_input,np.squeeze(S_input))
plt.show()
plt.xlabel('omega [rad/s]')
plt.ylabel('S(omega) [N^2/(rad/s)^2]')
plt.grid()
plt.title('PSD')
plt.xlim(0,80)
plt.yscale('log')
    
#%% Simulate the process

#plt.close('all')

Nsim=1
(t,x,X)=ts.MCCholesky(omega_input,S,Nsim,0.001)

# Cut to one hour duration
index_cut=np.argmin(abs(t-3600))
t=t[0:index_cut]
x=x[:,0:index_cut]

dt=t[1]-t[0]
T=t[-1]

# Plot time series
plt.figure()
plt.plot(t,x[0,:])
plt.show()
plt.xlabel('Time [s]')
plt.ylabel('Response [m]')
plt.grid()

#%% Find peaks and plot

plt.close('all')

height=0
distance=np.round(0.1/dt)
(index_peak,properties)=signal.find_peaks(x[0,:],height,None,distance)
N_peak=len(index_peak)    

plt.figure()
plt.plot(t,x[0,:])
plt.plot(t[index_peak],x[0,index_peak], 'o', color='red');
plt.show()
plt.xlabel('Time [s]')
plt.ylabel('Response [m]')
plt.grid()

x_peak=x[0,index_peak]

#%% Find upcrossings and plot

index_upcross= np.argwhere ( np.logical_and(x[0,1:]>0,x[0,0:-1]<0) )+1
N_upcross=len(index_upcross)    

plt.figure()
plt.plot(t,x[0,:])
plt.plot(t[index_upcross],np.squeeze(x[0,index_upcross]), 'o', color='green');
plt.show()
plt.xlabel('Time [s]')
plt.ylabel('Response [m]')
plt.grid()

x_upcross=x[0,index_upcross]

