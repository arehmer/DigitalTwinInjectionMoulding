# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:48:12 2021

@author: alexa
"""

# import os
# print (os.getcwd())


import numpy as np
import matplotlib.pyplot as plt

from models.NN import FirstOrderSystem

N = 100

u = np.ones((N,1))

PT1 = FirstOrderSystem(1,'injection_model')
PT1.Parameters['a']=-0.1


y_sim = PT1.Simulation(np.array([[0]]), u)

y_sim = np.array(y_sim)

plt.plot(y_sim)