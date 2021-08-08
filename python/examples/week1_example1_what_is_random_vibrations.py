# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 15:08:26 2021

@author: oiseth
"""
#%%
import numpy as np
from scipy.linalg import eig
import sys
sys.path.append('C:/Users/oiseth/Documents/GitHub/TKT4108StructuralDynamics2/python/modules')
from time_integration import *
from matplotlib import pyplot as plt

#%% Define structural properties
m = 1.0 # Mass of each story
k = 100.0 # Stiffness
MM = np.eye(2)*m # Mass matrix
KK = np.array(([[2, -1], [-1, 1]]))*k # Stiffness matrix

#%% Calculate modes and frequencies
lam,v = eig(KK,MM) #Solve eigenvalue problem using scipy 
lam = np.reshape(lam, (1, lam.shape[0]))
v[:,0] = v[:,0]/np.max(v[:,0]) #Normalize the eigenvector
v[:,1] = v[:,1]/np.max(v[:,1])
f = np.real(lam)**0.5/2/np.pi #Natural frequencies in Hz
omega = f*2*np.pi # Natural frequencies in rad/s
zeta = np.array(([[5.0, 5.0]]))/100

#%% Rayleigh damping
alpha1 = 2*omega[0,0]*omega[0,1]*(zeta[0,1]*omega[0,0]-zeta[0,0]*omega[0,1])/(omega[0,0]**2-omega[0,1]**2)
alpha2 = 2*(zeta[0,0]*omega[0,0]-zeta[0,1]*omega[0,1])/(omega[0,0]**2-omega[0,1]**2)
CC = alpha1*MM + alpha2*KK

#%% Determenistic dynamic response due to harmonic load 
h = 0.05 #Time step
t = np.array([np.arange(0.,100.,h)]) # Time vector
fl = 2.0 # Load frequency
po = 100.0 # Load amplitude
u0 = np.array([[0.0], [0.0]]) #Initial displacement
udot0 = np.array([[0.0], [0.0]]) # Initial velocity
beta = 1.0/4.0 # Facor in Newmark's method
gamma = 1.0/2.0 # Factor in Newmark's method
X = np.vstack((np.sin(2.0*np.pi*fl*t),np.sin(2.0*np.pi*fl*t)))
y, ydot, y2dot = linear_newmark_krenk(MM,CC,KK,X,u0,udot0,h,gamma,beta)

# Plot deterministic dynamic load and response
fig, axs = plt.subplots(2,2)
axs[0,0].plot(t[0,:], X[0,:])
axs[0,0].set_ylabel('$X_1$')
axs[0,0].set_title('Loads')
axs[0,0].grid(True)

axs[1,0].plot(t[0,:], X[1,:])
axs[1,0].set_ylabel('$X_2$')
axs[1,0].set_xlabel('$t$')
axs[1,0].grid(True)


axs[0,1].plot(t[0,:], y[0,:])
axs[0,1].set_ylabel('$y_1$')
axs[0,1].set_title('Responses')
axs[0,1].grid(True)

axs[1,1].plot(t[0,:], y[1,:])
axs[1,1].set_ylabel('$y_2$')
axs[1,1].set_xlabel('$t$')
axs[1,1].grid(True)

#%% Dynamic response due to stochastic load
rho_X1_X2 = 0.8 # Load correlation coefficient
stdX1 = 100.0 # Standard deviation X1
stdX2 = 100.0 # Standard deviation X2

covmX = np.array(([[stdX1**2, rho_X1_X2*stdX1*stdX2], [rho_X1_X2*stdX1*stdX2, stdX2**2]])) # Covariance matrix of the loads
lam,v = eig(covmX) #Solve eigenvalue problem using scipy 
covmX_modal = np.matmul(np.matmul(v.T,covmX),v) # Transform covariance matrix to uncorrelated space

U = np.vstack((np.random.normal(0, covmX_modal[0,0]**0.5, t.shape[1]),np.random.normal(0, covmX_modal[1,1]**0.5, t.shape[1])))
X = np.matmul(v,U) # Transform to correlated space

y, ydot, y2dot = linear_newmark_krenk(MM,CC,KK,X,u0,udot0,h,gamma,beta)

# Plot stochastic dynamic load and response
fig, axs = plt.subplots(2,2)
axs[0,0].plot(t[0,:], X[0,:])
axs[0,0].set_ylabel('$X_1$')
axs[0,0].set_title('Loads')
axs[0,0].grid(True)

axs[1,0].plot(t[0,:], X[1,:])
axs[1,0].set_ylabel('$X_2$')
axs[1,0].set_xlabel('$t$')
axs[1,0].grid(True)


axs[0,1].plot(t[0,:], y[0,:])
axs[0,1].set_ylabel('$y_1$')
axs[0,1].set_title('Responses')
axs[0,1].grid(True)

axs[1,1].plot(t[0,:], y[1,:])
axs[1,1].set_ylabel('$y_2$')
axs[1,1].set_xlabel('$t$')
axs[1,1].grid(True)

#%% Plot scatter plot of dynamic loads
plt.figure()
plt.show()
plt.scatter(X[0,:],X[1,:])
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.grid()






