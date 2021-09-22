# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:43:54 2021

@author: oiseth
"""

#%% import necessary modules and packages
import numpy as np
from matplotlib import pyplot as plt

#%% Make scatter plot that illustrates linear correlation
mean = np.array([0, 0]) # Mean values
sigma1 = 1.0 # Standard deviation of X1
sigma2 = 1.0 # Standard deviation of X2
rho12 = 0.5 # correlation coefficient between -1 and 1
Nsim = 1000 # Number of points
cov = np.array([[sigma1**2, sigma1*sigma2*rho12], [sigma1*sigma2*rho12, sigma2**2]]) # Covariance matrix
X = np.random.multivariate_normal(mean,cov,Nsim) # Monte Carlo simulation of correlated Gaussian (normal) variables

# Scatter plot
plt.figure()
plt.show()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.grid()
plt.title(r'$\rho=' + str(rho12) + '$')

#%% Define auto-correlation function
dt = 0.2
t = np.arange(0.,100.,dt) # Time vector
tau = t-np.max(t)/2
omega_c = 1.0; # Cut-off frequency
sigma = 1.0
R = sigma**2/(omega_c*tau)*np.sin(omega_c*tau); # Auto-correlation function
plt.figure()
plt.show()
plt.plot(tau,R)
plt.ylim(-sigma,sigma)
plt.grid()
plt.ylabel(r'$R(\tau)$')
plt.xlabel(r'$\tau$')
#%% Use auto-correlation function to generate stochastic time series
tau_mat = np.abs(np.array([t])-np.array([t]).T) # Matrix of all possible time lags
tau_mat[tau_mat==0] = np.finfo(float).eps # Avoid the singularity when \tau = 0
mean = np.zeros((t.shape[0])) 
cov = sigma**2/(omega_c*tau_mat)*np.sin(omega_c*tau_mat); # Auto-correlation function
X = np.random.multivariate_normal(mean,cov,3)

fig, axs = plt.subplots(3,1)
axs[0].plot(tau, R)
axs[0].set_ylabel(r'$R(\tau)$')
axs[0].set_xlabel(r'$\tau$')
axs[0].set_ylim(-sigma,sigma)
axs[0].grid(True)

axs[1].plot(t, X[0,:])
axs[1].set_ylabel(r'$X_1$')
axs[1].grid(True)

axs[2].plot(t, X[1,:])
axs[2].set_ylabel(r'$X_1$')
axs[2].grid(True)

