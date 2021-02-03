# -*- coding: utf-8 -*-
from sys import path
#path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

import Modellklassen as Model
from OptimizationTools import *
from miscellaneous import *

''' Generate Identification Data '''
N = 100

u_train = np.zeros((10,N-1,2))
x_train = np.zeros((10,N,1))

for i in range(0,10):

    x = np.zeros((N,1))
    u = np.random.normal(0,1,(N-1,2))

    for k in range(1,100):
        x[k] = 0.1*x[k-1] + 0.1*u[k-1,0]**2 - 0.3*u[k-1,1]**3
    
    u_train[i,:,:] = u
    x_train[i,:,:] = x


''' Initialize Model '''
model = Model.MLP(dim_u=2,dim_x=1,dim_hidden=10,name='test')

''' Estimate Parameters FF'''
init_state = x_train[:,0,:].reshape(10,1,1) 


new_params = EstimateModelParams(model,u_train,x_train,init_state)



''' Estimate Parameters RNN'''


model = Model.GRU(dim_u=2,dim_c=1,dim_hidden=2,dim_out=2,name='GRU')

x_ref = x_train[:,-1,:]
x_ref = x_ref.reshape((10,1,1))

init_state = np.zeros((10,1,1))

new_params = EstimateModelParams(model,u_train,x_ref,init_state)


''' Compare new and old model '''
  
model.Parameters = new_params

# Simulate Model
x = model.Simulation(init_state[0,:,:],u_train[0,:,:])



# Concatenate list to casadiMX
x = vcat(x)    
x = np.array(x)        


# np.linalg.norm(x_train[0,:,:]-x)
 


# plt.plot(x_train[0,:,:])
# plt.plot(x)