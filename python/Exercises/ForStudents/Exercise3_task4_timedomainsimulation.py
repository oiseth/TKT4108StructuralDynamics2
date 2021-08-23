# -*- coding: utf-8 -*-
"""

"""

#%% Import necessary modules and packages
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

import time_integration as ti
import time_simulation as ts

#%% Create signal in time domain

plt.close('all')

# Establish system
L=3
I=60e-6
E=210e9
k=12*E*I/L**3
m=20e3

M=np.diag([1.2*m,m,1.2*m])
K=np.array([ [2.2,-1.2,0] , [-1.2, 2.2, -1] , [0 , -1 , 1] ])*k

C=0.16*M+8e-4*K

(lambd,phi)=linalg.eigh(K,M)

phi=np.array(phi)
lambd=np.array(lambd)
omega_n=np.sqrt(lambd)

#%% Monte carlo simulation

#Choose the desired load spectrum 
#load_type='a'
# load_type='b'
load_type='c'

# Import the load spectrum
import Exercise3_task4_loadspectrum
(omega_load,Spp)=Exercise3_task4_loadspectrum.loadspectrum(load_type)

Nsim=15

(t,p,P)=ts.MCCholesky(omega_load,Spp,Nsim,0.01)

dt=t[1]-t[0]
T=t[-1]

# Plot time series
plt.figure()
plt.plot(t,P[0][0,:])
plt.plot(t,P[0][1,:])
plt.plot(t,P[0][2,:])
plt.show()
plt.xlabel('Time [s]')
plt.ylabel('Load [N]')
plt.grid()

#%% Spectrum of load

#plt.close('all')

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

# Statistics
std_u=np.zeros((3,Nsim))
max_uddot=np.zeros((3,Nsim))

# Initial conditions
u0=np.zeros((3,1))
udot0=np.zeros((3,1))

# Newmark parameters
gamma=0.5
beta=0.25

# Empty matrices for the response
U1=np.zeros((Nsim,len(t)))
U2=np.zeros((Nsim,len(t)))
U3=np.zeros((Nsim,len(t)))

# Solve response
for k in np.arange(0,Nsim):
    
    print('Solving simulation ' + str(k+1))
    
    # Time integration
    f=P[k]
    (u,udot,uddot)=ti.linear_newmark_krenk(M,C,K,f,u0,udot0,dt,gamma,beta)
   
    # Statistics
    std_u[:,k]=np.std(u,1)
    max_uddot[:,k]=np.max(uddot,1)
    
    # Save response time series for each simulation
    U1[k,]=u[0,:]
    U2[k,]=u[1,:]
    U3[k,]=u[2,:]


# Plot time series
plt.figure()
plt.plot(t,u[0,:])
plt.plot(t,u[1,:])
plt.plot(t,u[2,:])
plt.show()
plt.xlabel('Time [s]')
plt.ylabel('Response [m]')
plt.grid()

#%% Statistics

plt.figure()
plt.plot(std_u[0,:],'o')
plt.plot(std_u[1,:],'x')
plt.plot(std_u[2,:],'d')
plt.show()
plt.xlabel('Simulation')
plt.ylabel('SD [m]')
plt.grid()

plt.figure()
plt.plot(max_uddot[0,:],'o')
plt.plot(max_uddot[1,:],'x')
plt.plot(max_uddot[2,:],'d')
plt.show()
plt.xlabel('Simulation')
plt.ylabel('Max acc. [m/s^2]')
plt.grid()


#%% Spectrum

#plt.close('all')

# Find DFT
(f,GU1)=ts.fft_function(U1,dt,1)
(f,GU2)=ts.fft_function(U2,dt,1)
(f,GU3)=ts.fft_function(U3,dt,1)

# Find spectrum of each simulation
S_U1=np.real( np.multiply(GU1,np.conj(GU1))*T )
S_U2=np.real( np.multiply(GU2,np.conj(GU2))*T )
S_U3=np.real( np.multiply(GU3,np.conj(GU3))*T )

# Average the spectrum 
S_U1_avg=np.mean(S_U1,0)
S_U2_avg=np.mean(S_U2,0)
S_U3_avg=np.mean(S_U3,0)

plt.figure()
plt.plot(f,S_U1_avg)
plt.plot(f,S_U2_avg)
plt.plot(f,S_U3_avg)
plt.show()
plt.xlabel('f [Hz]')
plt.ylabel('S(omega) [m^2/Hz]')
plt.grid()
plt.title('PSD of response')
plt.xlim(-10,10)
plt.yscale('log')


