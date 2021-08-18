# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:43:54 2021

@author: oiseth
"""

#%% import necessary modules and packages
import numpy as np
from matplotlib import pyplot as plt

#%% Time series plot
plt.close('all')

omega=1
x0=1
dt=0.01
T=6000
t=np.arange(0,T,dt)
x=x0*np.sin(omega*t)

# Plot time series
plt.figure()
plt.show()
plt.plot(t,x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.title('Sine wave')

#%% Plot histogram and theoretical distribution

plt.figure()
plt.hist(x,20,None,True,None,False,None)

x_axis=np.linspace(-x0,x0,100)
plt.plot(x_axis,1/(np.pi*np.sqrt(x0**2-x_axis**2)))
plt.show()
plt.title('Histogram of x')

#%% Plot histogram and theoretical distribution

N=10

# Sum 
x=np.zeros(np.shape(t))
for k in np.arange(0,N):
    phi=2*np.pi*np.random.uniform(0,1,np.shape(t)) # Random values between [0,2*pi]
    x=x+np.cos(omega*t-phi)
    
# Plot time series
plt.figure()
plt.plot(t,x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.title('Sum of sine waves')

# Plot histogram
plt.figure()
plt.hist(x,20,None,True,None,False,None)
plt.title('Histogram of x')



