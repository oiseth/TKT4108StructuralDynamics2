# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:43:54 2021

@author: oiseth
"""

#%% import necessary modules and packages
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
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

(lambd,phi)=scipy.linalg.eigh(K,M)

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
plt.show()
plt.xlabel('Time [s]')
plt.ylabel('Load [N]')
plt.grid()

#%% Solve system response by time integration

f=np.vstack((X1,X2))

u0=np.zeros((2,1))
udot0=np.zeros((2,1))
gamma=0.5
beta=0.25

(u,udot,uddot)=ti.linear_newmark_krenk(M,C,K,f,u0,udot0,dt,gamma,beta)

plt.figure()
plt.plot(t,u[0,:])
plt.plot(t,u[1,:])
plt.show()
plt.xlabel('Time [s]')
plt.ylabel('Response [m]')
plt.grid()

#%% Statistics

sigma_u=np.std(u,1)

rho_u=np.corrcoef(u)

#%% Simulate multiple realizations

u0=np.zeros((2,1))
udot0=np.zeros((2,1))
gamma=0.5
beta=0.25

Nsim=50

U1=np.zeros((Nsim,len(t)))
U2=np.zeros((Nsim,len(t)))

for k in np.arange(0,Nsim):
    
    X1=500*np.random.randn(len(t))
    X2=600*np.random.randn(len(t))
    f=np.vstack((X1,X2))
    (u,udot,uddot)=ti.linear_newmark_krenk(M,C,K,f,u0,udot0,dt,gamma,beta)
    
    U1[k,]=u[0,:]
    U2[k,]=u[1,:]
    
#%% Spectrum

plt.close('all')

(f,G1)=ts.fft_function(U1,dt,1)
(f,G2)=ts.fft_function(U2,dt,1)

S_U1=np.multiply(G1,np.conj(G1))*T
S_U2=np.multiply(G2,np.conj(G2))*T

S_U1_avg=np.mean(S_U1,0)
S_U2_avg=np.mean(S_U2,0)

plt.figure()
plt.plot(f,S_U1_avg)
plt.plot(f,S_U2_avg)
plt.show()
plt.xlabel('f [Hz]')
plt.ylabel('S(omega) [m^2/Hz]')
plt.grid()
plt.title('PSD of response')
plt.xlim(-20,20)

plt.yscale('log')


