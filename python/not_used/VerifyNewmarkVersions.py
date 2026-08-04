# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:43:54 2021

@author: oiseth
"""

#%% import necessary modules and packages
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

import time_integration as ti
import time_simulation as ts

#%% Create signal in time domain

plt.close('all')

# Establish system
L=3.5
I=180e-6
E=210e9
k=12*E*I/L**3
m=12e3

M=np.diag([m,m])
K=np.array([ [4,-2] , [-2, 2] ])*k
C=3e-1*M+5e-4*K

(lambd,phi)=linalg.eigh(K,M)

phi=np.array(phi)
lambd=np.array(lambd)
omega_n=np.sqrt(lambd)

#%% Create time vector and load

dt=0.01
T=100
t=np.arange(0,T,dt)

X1=500*np.random.randn(len(t))
X2=600*np.random.randn(len(t))

# Plot time series
plt.figure()
plt.plot(t,X1)
plt.plot(t,X2)
plt.xlabel('Time [s]')
plt.ylabel('Load [N]')
plt.grid()

#%% Solve system response by time integration

plt.close('all')

f=np.vstack((X1,X2))

u0=np.zeros((2,1))
udot0=np.zeros((2,1))
gamma=0.5
beta=0.25

(u,udot,uddot)=ti.linear_newmark_krenk(M,C,K,f,u0,udot0,dt,gamma,beta)
(y,ydot,yddot)=ti.linear_newmark_chopra(M,C,K,f,u0,udot0,dt,gamma,beta)

plt.figure()
plt.plot(t,u[0,:])
plt.plot(t,y[0,:])
plt.xlabel('Time [s]')
plt.ylabel('Response [m]')
plt.title('DOF 1')
plt.grid()

plt.figure()
plt.plot(t,u[1,:])
plt.plot(t,y[1,:])
plt.xlabel('Time [s]')
plt.ylabel('Response [m]')
plt.title('DOF 2')
plt.grid()

#%% Statistics: the difference should be almost zero

delta_u=u-y
delta_udot=udot-ydot
delta_uddot=uddot-yddot

ratio_u=np.divide( np.std(delta_u,1) , np.std(u,1) )
ratio_udot=np.divide( np.std(delta_udot,1) , np.std(udot,1) )
ratio_uddot=np.divide( np.std(delta_uddot,1) , np.std(uddot,1) )

plt.figure()
plt.plot(ratio_u)
plt.plot(ratio_udot)
plt.plot(ratio_uddot)
plt.show()

#%%
