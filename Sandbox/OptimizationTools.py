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



def MultiStageOptimization(model,ref):
    #Multi State Optimization for solving the optimal control problem
    
 
    # Create Instance of the Optimization Problem
    opti = casadi.Opti()
    
    # Translate Maschinenparameter into opti.variables
    model.CreateOptimVariables(opti, model.Maschinenparameter)
    
    # Number of time steps
    N = ref['data'].shape[0]
    
    # Create decision variables for states
    X = opti.variable(N,model.NumStates)
        
    # Initial Constraints
    opti.subject_to(X[0]==ref['data'][0])
    
    
    # System Dynamics as Path Constraints
    for k in range(N-1):
        
        if k<=ref['Umschaltpunkt']:
            opti.subject_to(model.SimulateInject(X[k],model.ControlInput(opti,k))==X[k+1])
        else:
            opti.subject_to(model.SimulatePress(X[k],model.ControlInput(opti,k))==X[k+1])
    
    # Final constraint
    opti.subject_to(X[-1]==ref['data'][-1])
    
    
    # Set initial values for Machine Parameters
    for key in model.opti_vars:
        opti.set_initial(model.opti_vars[key],model.Maschinenparameter[key])

    # Set initial values for state trajectory ??
    # for key in model.opti_vars:
    #     opti.set_initial(model.opti_vars[key],CurrentParams[key])      
    
    # Define Loss Function    
    opti.minimize(sumsqr(X-ref['data']))
    
    #Choose solver
    opti.solver('ipopt')
    
    # Get solution
    sol = opti.solve()
    
    # Extract real values from solution
    values = OptimValues_to_dict(model.opti_vars,sol)
    values['X'] = sol.value(X)

    
    return values


def UpdateModelParams(model,u,x_ref,phase):
    
    
    # Create Instance of the Optimization Problem
    opti = casadi.Opti()
    
    if phase == 'inject':
        
        # Save Current Parameters
        CurrentParams = model.ModelParamsInject   
        
        # Make Model Parameters opti.variables
        model.CreateOptimVariables(opti, CurrentParams)

        # Overwrite current numerical parameter values with opti.variables 
        model.ModelParamsInject = model.opti_vars
        
        # create pointer to respective one step integrator
        ModelSimulator = model.SimulateInject
        
    elif phase == 'press':
        
        # Save Current Parameters
        CurrentParams = model.ModelParamsPress  
        
        # Make Model Parameters opti.variables
        model.CreateOptimVariables(opti, CurrentParams)

        # Overwrite current numerical parameter values with opti.variables 
        model.ModelParamsPress = model.opti_vars  
        
      
        # create pointer to respective one step integrator
        ModelSimulator = model.SimulatePress

       
    if u.shape[0]+1 != x_ref.shape[0]:
        sys.exit('Shapes of input and output time series do not match!')
    
    
    N = u.shape[0]
    
   
    x = []
    
    # initial states
    x.append(x_ref[0])
   
           
    # Save current model parameters
    for i in range(N):
        x.append(ModelSimulator(x[i],u[i]))
    
    # Concatenate list to casadiMX
    x = vcat(x)    
    
    # Calculate Simulation error
    # e = norm_2(x_ref - x)**2        # IMPORTANT: use squared L2-norm to prevent division by zero                            
    
    e = sumsqr(x_ref - x)
    
    opti.minimize(e)

    opti.solver('ipopt')
    
    # Set initial values for Model Parameters
    for key in model.opti_vars:
        opti.set_initial(model.opti_vars[key],CurrentParams[key])


    sol = opti.solve()
    values = OptimValues_to_dict(model.opti_vars,sol)
       
    # in the end return model parameters to numerical values again or hell breaks loose
    if phase == 'inject':
        model.ModelParamsInject = CurrentParams
    elif phase == 'press':
        model.ModelParamsPress = CurrentParams
    
    return values














