# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 09:49:27 2022

@author: oyvinpet
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as sps
from functions import sDynFunctions

# Structural properties
m = 1000.0; 
c = 2000.0;
EI = 2.1E11*3.36E-6;
L = 4.0;
k = 2*12*EI/L**3;
MM = np.diag([1, 1, 1])*m;
KK = np.array([ [2, -1, 0], [-1, 2, -1], [0, -1, 1]])*k
CC = np.diag([1, 0.5, 0])*c;


dt=0.01 # Time step
T=600 # Total time
t = np.arange(0,T,dt) # Time vector

f=np.random.normal(0,50, size=(3,len(t))) # White noise loading

u0=np.zeros((3,1)) # Initial displacement
udot0=np.zeros((3,1)) # Initial velocity

u, udot, u2dot=sDynFunctions.linear_newmark_krenk(MM,CC,KK,f,u0,udot0,dt,0.5,0.25)
    
# Plot displacements
plt.figure(figsize=(10,3))   
plt.plot(t,u[0,:],'-',label = '$u_1$')  
plt.plot(t,u[1,:],'-',label = '$u_2$')    
plt.plot(t,u[2,:],'-',label = '$u_3$')    
plt.xlabel('$t$')
plt.ylabel('$u(t)$');
plt.xlim(0,50)
plt.grid()
plt.legend()


from scipy.signal import welch, hanning, csd

Ndivisions=10 # Number of divisions of the time series
Nwindow=np.ceil(len(t)/Ndivisions) # Length of window

Nfft_pow2 = 2**(np.ceil(np.log2(Nwindow))) # Next power of 2


#%%



