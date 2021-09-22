# -*- coding: utf-8 -*-
"""

"""

#%% Import necessary modules and packages
import numpy as np
from scipy import linalg
from scipy import signal
from matplotlib import pyplot as plt


#%% Load wind velocity and plot

plt.close('all')

t=np.load('ProblemSet5_task2_t.npy')
V=np.load('ProblemSet5_task2_V.npy')
          
# Plot some of the wind time series
plt.figure()
plt.plot(t[0],V[0,:])
plt.plot(t[0],V[10,:])
plt.plot(t[0],V[20,:])
plt.show()
plt.xlabel('t [s]')
plt.ylabel('V [m/s]')
plt.grid()
plt.xlim(0,600)

