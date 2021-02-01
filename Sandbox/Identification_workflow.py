#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:39:31 2021

@author: alexander
"""

# -*- coding: utf-8 -*-
from sys import path
#path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi  as cs
import matplotlib.pyplot as plt
import numpy as np

import Modellklassen as Model
from OptimizationTools import *
from miscellaneous import *


test = Model.FeedForwardNeuralNetwork(2,1,3,'testname')


dim_u=2
dim_x=3
dim_hidden=5
function_name='test'

u = cs.MX.sym('u',dim_u,1)
x = cs.MX.sym('x',dim_x,1)

W_h = cs.MX.sym('W_h',dim_hidden,dim_u+dim_x)
b_h = cs.MX.sym('b_h',dim_hidden,1)

W_o = cs.MX.sym('W_out',dim_x,dim_hidden)
b_o = cs.MX.sym('b_out',dim_x,1)

h =  cs.tanh(cs.mtimes(W_h,cs.vertcat(u,x))+b_h)
x_new = cs.mtimes(W_o,h)+b_o


input = [x,u,W_h,b_h,W_o,b_o]
input_names = ['x','u','W_h','b_h','W_o','b_o']

output = [x_new]
output_names = ['x_new']

F = cs.Function(function_name, input, output, input_names,output_names)