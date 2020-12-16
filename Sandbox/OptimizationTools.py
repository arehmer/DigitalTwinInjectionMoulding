#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:25:16 2020

@author: alexander
"""

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

from miscellaneous import *

def SimulateModel(casadi_fun,x,u,params):
       # Casadi Function needs list of parameters as input
        params_new = []
        
        for name in  casadi_fun.name_in():
            try:
                params_new.append(params[name])                     # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x_new = casadi_fun(x,u,*params_new)     
                              
        return x_new
    
def CreateOptimVariables(opti, param_dict):
    
    Parameter = {}

    for key in param_dict.keys():
        
        dim0 = param_dict[key].shape[0]
        dim1 = param_dict[key].shape[1]
        
        Parameter[key] = opti.variable(dim0,dim1)

    opti_vars = Parameter
    
    return opti_vars

def MultiStageOptimization(model,ref):
    #Multi Stage Optimization for solving the optimal control problem
    
 
    # Create Instance of the Optimization Problem
    opti = casadi.Opti()
    
    # Translate Maschinenparameter into opti.variables
    Maschinenparameter_opti = CreateOptimVariables(opti, model.Maschinenparameter)
    
    # Number of time steps
    N = ref['data'].shape[0]
    
    # Create decision variables for states
    X = opti.variable(N,model.NumStates)
        
    # Initial Constraints
    opti.subject_to(X[0]==ref['data'][0])
    
    
    # System Dynamics as Path Constraints
    for k in range(N-1):
        
        if k<=ref['Umschaltpunkt']:
            U = model.ControlInput(Maschinenparameter_opti,k)
            opti.subject_to(SimulateModel(model.ModelInject,X[k],U,
                                          model.ModelParamsInject)==X[k+1])
            # opti.subject_to(model.SimulateInject(X[k],model.ControlInput(opti,k))==X[k+1])
        else:
            U = model.ControlInput(Maschinenparameter_opti,k)
            opti.subject_to(SimulateModel(model.ModelPress,X[k],U,
                                          model.ModelParamsPress)==X[k+1])
            # opti.subject_to(model.SimulatePress(X[k],model.ControlInput(opti,k))==X[k+1])
    
    ''' Further Path Constraints (to avoid values that might damage the machine or in 
    other ways harmful or unrealistic) '''
    
    # TO DO #
    
    
    # Final constraint
    opti.subject_to(X[-1]==ref['data'][-1])
    
    
    # Set initial values for Machine Parameters
    for key in Maschinenparameter_opti:
        opti.set_initial(Maschinenparameter_opti[key],model.Maschinenparameter[key])

    # Set initial values for state trajectory ??
    # for key in model.Maschinenparameter_opti:
    #     opti.set_initial(model.Maschinenparameter_opti[key],CurrentParams[key])      
    
    # Define Loss Function    
    opti.minimize(sumsqr(X-ref['data']))
    
    #Choose solver
    opti.solver('ipopt')
    
    # Get solution
    sol = opti.solve()
    
    # Extract real values from solution
    values = OptimValues_to_dict(Maschinenparameter_opti,sol)
    values['X'] = sol.value(X)

    
    return values

def QualityOptimization(model,ref):
    #
    # Create Instance of the Optimization Problem
    opti = casadi.Opti()
    
    # In this case, time is a decision variable...
    
    # Translate Maschinenparameter into opti.variables
    Maschinenparameter_opti = CreateOptimVariables(opti, model.Maschinenparameter)
    
    # Number of time steps
    N = ref['data'].shape[0]
    
    # Create decision variables for states
    X = opti.variable(N,model.NumStates)
        
    # Initial Constraints
    opti.subject_to(X[0]==ref['data'][0])    


    return values

def UpdateModelParams(casadi_fun,u,x_ref,params):
    
    # Create Instance of the Optimization Problem
    opti = casadi.Opti()
    
    params_opti = CreateOptimVariables(opti, params)
    
    if u.shape[0]+1 != x_ref.shape[0]:
        sys.exit('Shapes of input and output time series do not match!')
    
    N = u.shape[0]
       
    x = []
    
    # initial states
    x.append(x_ref[0])
   
           
    # Simulate Model
    for i in range(N):
        x.append(SimulateModel(casadi_fun,x[i],u[i],params_opti))
    
    # Concatenate list to casadiMX
    x = vcat(x)    
   
    e = sumsqr(x_ref - x)
    
    opti.minimize(e)

    opti.solver('ipopt')
    
    # Set initial values for Model Parameters
    for key in params_opti:
        opti.set_initial(params_opti[key],params[key])

    sol = opti.solve()
    values = OptimValues_to_dict(params_opti,sol)
    
    return values














