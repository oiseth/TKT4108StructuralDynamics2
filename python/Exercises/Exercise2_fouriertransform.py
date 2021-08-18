# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:43:54 2021

@author: oiseth
"""

#%% import necessary modules and packages
import numpy as np
from matplotlib import pyplot as plt
import time_integration as ti
import time_simulation as ts

#%% Create signal in time domain

plt.close('all')

x0=1
dt=0.01
T=100
omega_1=3
omega_2=8
A1=1
A2=0.3

t=np.arange(0,T,dt)
x=A1*np.sin(omega_1*t)+A2*np.sin(omega_2*t)

# Plot time series
plt.figure()
plt.show()
plt.plot(t,x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.title('Sum of sine waves')

#%% Frequency domain (DFT)

# Take the DFT
(f,G)=ts.fft_function(x,dt,0)


# Plot time series
plt.figure()
plt.plot(f,np.abs(G))
plt.xlabel('f [Hz]')
plt.ylabel('|FFT(x(t))|')
plt.grid()
plt.title('Fourier transform')
plt.xlim(-3,3)



