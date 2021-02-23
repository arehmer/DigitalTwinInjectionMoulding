# -*- coding: utf-8 -*-

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

from casadi import *
import matplotlib.pyplot as plt
import numpy as np


def OptimValues_to_dict(optim_variables_dict,sol):
    # reads optimized parameters from optim solution and writes into dictionary
    
    values = {}
    
    for key in optim_variables_dict.keys():
       dim0 = optim_variables_dict[key].shape[0]
       dim1 = optim_variables_dict[key].shape[1]
       
       values[key] = sol.value(optim_variables_dict[key]) 
       
       # Convert tu numpy array
       values[key] = np.array(values[key]).reshape((dim0,dim1))

      
    return values