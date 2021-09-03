# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:13:40 2021

@author: oiseth
"""

import numpy as np # Import numpy
from matplotlib import pyplot as plt # pyplot module for plotting

dt = 0.01 # Time step
t = np.arange(0,10.01,dt) # time vector
x = np.zeros(t.shape) # Initialize the x array
x[t<5] = 1.0 # Set the value of x to one for t<5
# Plot waveform
plt.figure()
plt.plot(np.hstack((t-20.0,t-10.0, t, t+10.0)),np.hstack((x,x,x,x))); # Plot four periods
plt.plot(t,x); #Plot one period
plt.ylim(-2, 2);
plt.xlim(-20,20);
plt.grid();
plt.xlabel('$t$');
plt.ylabel('$X(t)$');

nterms = 10 # Number of Fourier coefficeints
T = np.max(t) # The period of the waveform
a0 = 1/T*np.trapz(x,t) # Mean value
ak = np.zeros((nterms)) 
bk = np.zeros((nterms))
for k in range(nterms): # Integrate for all terms
    ak[k] = 1/T*np.trapz(x*np.cos(2.0*np.pi*(k+1.0)*t/T),t)
    bk[k] = 1/T*np.trapz(x*np.sin(2.0*np.pi*(k+1.0)*t/T),t)

# Plot Fourier coeffecients
fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

ax1 = axs[0]
ax1.plot(np.arange(1,nterms+1),ak)
ax1.set_ylim(-1, 1)
ax1.grid()
ax1.set_ylabel('$a_k$');
ax1.set_xlabel('$k$');

ax2 = axs[1]
ax2.plot(np.arange(1,nterms+1),bk)
ax2.set_ylim(-1, 1)
ax2.grid()
ax2.set_ylabel('$b_k$');
ax2.set_xlabel('$k$');
#%%
# Plot Fourier series approximation
tp  = np.linspace(-20,20,1000)
X_Fourier = np.ones(tp.shape[0])*a0
for k in range(nterms):
    X_Fourier = X_Fourier + 2.0*(ak[k]*np.cos(2.0*np.pi*(k+1.0)*tp/T) + bk[k]*np.sin(2.0*np.pi*(k+1.0)*tp/T))

plt.figure(figsize=(8,4))
plt.plot(np.hstack((t-20.0,t-10.0, t, t+10.0)),np.hstack((x,x,x,x))); # Plot four periods
plt.plot(tp,X_Fourier, label=('Fourier approximation Nterms='+str(nterms)));
plt.ylim(-2, 2)
plt.xlim(-20,20)
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.legend();
    