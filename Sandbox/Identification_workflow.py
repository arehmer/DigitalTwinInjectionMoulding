# -*- coding: utf-8 -*-
from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Modellklassen as Model
from OptimizationTools import *
from miscellaneous import *

''' Generate Identification Data '''
N = 100

u = np.zeros((10,N-1,2))
x = np.zeros((10,N,1))

for i in range(0,10):

    x_i = np.zeros((N,1))
    u_i = np.random.normal(0,1,(N-1,2))

    for k in range(1,100):
        x_i[k] = 0.1*x_i[k-1] + 0.1*u_i[k-1,0]**2 - 0.3*u_i[k-1,1]**3
    
    u[i,:,:] = u_i
    x[i,:,:] = x_i


init_state = x[:,0,:].reshape(10,1,1) 
data = {'u_train':u[0:8], 'x_train':x[0:8],'init_state_train': init_state[0:8],
        'u_val':u[8::], 'x_val':x[8::],'init_state_val': init_state[8::]}



''' Estimate Parameters MLP '''
model = Model.MLP(dim_u=2,dim_x=1,dim_hidden=10,name='test')
# ident_res = ModelTraining(model,data,initializations = 2)

param_bounds = {'dim_hidden':np.array([2,10])}
options = {'c1': 0.6, 'c2': 0.3, 'w': 0.4, 'k':5, 'p':1}
n_particles = 5
kwargs = {}

results, hist = HyperParameterPSO(model,data,param_bounds,n_particles,
                            options,**kwargs)


''' Estimate Parameters RNN'''


# model = Model.GRU(dim_u=2,dim_c=1,dim_hidden=2,dim_out=2,name='GRU')

# data['init_state_train'] = np.zeros((8,1,1))
# data['init_state_val'] = np.zeros((2,1,1))
# data['x_train'] = x[0:8,-1,:].reshape(8,1,1)
# data['x_val'] = x[8::,-1,:].reshape(2,1,1)

# ident_res = ModelTraining(model,data,initializations = 2)



''' Compare new and old model '''
  
# model.Parameters = new_params

# # Simulate Model
# x = model.Simulation(init_state[0,:,:],u_train[0,:,:])

# x = np.array(x)        


# np.linalg.norm(x_train[0,:,:]-x)
 


# plt.plot(x_train[0,-1,:])
# plt.plot(x)