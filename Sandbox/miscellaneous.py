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
       values[key] = sol.value(optim_variables_dict[key]) 
       
    return values