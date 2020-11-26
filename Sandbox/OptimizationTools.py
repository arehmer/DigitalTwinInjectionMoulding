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


def MultiStageOptimization(model,reference):

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
    opti.subject_to(X[0]==reference['data'][0])
    
    
    # System Dynamics as Path Constraints
    for k in range(reference['N']-1):
        
        if k<=N1:
            opti.subject_to(model.Einspritzphase(X[k],model.ControlInput(opti,k))==X[k+1])
        else:
            opti.subject_to(model.Nachdruckphase(X[k],model.ControlInput(opti,k))==X[k+1])
    
    # Final constraint
    opti.subject_to(X[-1]==reference['data'][-1])

    # Define Loss Function    
    opti.minimize(sumsqr(X-reference['data']))
    
    #Choose solver
    opti.solver('ipopt')
    
    sol = opti.solve()
    
    Xsol_MS = sol.value(X) # should be [-2.7038;-0.5430;0.2613;0.5840]
    # print(Usol_MS)
    
    opti_MS = opti
    # U_MS = U
    # sol_MS = sol
        
    plt.figure()
    plt.plot(reference['data'])
    plt.plot(Xsol_MS)
    
    return sol
