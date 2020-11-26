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

    # Specify Dimensions
    N1 = 40
    N2 = 20
    N=N1+N2
    
    
    # Create Instance of the Optimization Problem
    opti = casadi.Opti()
    
    
    # Translate Maschinenparameter into opti.variables
    model.CreateOptimVariables(opti, model.Maschinenparameter)
    
    # Create decision variables for states
    X = opti.variable(N,model.NumStates)
        
    
    # Initial Constraints
    opti.subject_to(X[0]==ref['data'][0])
    
    
    # System Dynamics as Path Constraints
    for k in range(ref['N']-1):
        
        if k<=N1:
            opti.subject_to(model.Einspritzphase(X[k],model.ControlInput(opti,k))==X[k+1])
        else:
            opti.subject_to(model.Nachdruckphase(X[k],model.ControlInput(opti,k))==X[k+1])
    
    # Final constraint
    opti.subject_to(X[-1]==ref['data'][-1])
    
    
    # Constraints on parameters if any 
    
    
    # Define Loss Function    
    opti.minimize(sumsqr(X-ref['data']))
    
    #Choose solver
    opti.solver('ipopt')
    
    # Get solution
    sol = opti.solve()
    
    # Extract real values from solution
    values = OptimValues_to_dict(model.opti_params,sol)
    values['X'] = sol.value(X)

    
    return values
