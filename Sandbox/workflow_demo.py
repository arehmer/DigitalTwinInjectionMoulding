# -*- coding: utf-8 -*-
from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

from Modellklassen import Arburg320C
from OptimizationTools import *
from miscellaneous import *


''' Model Identification '''
############


''' Setup '''
model = Arburg320C()

N=60

h1 = np.array([[1]])
h2 = np.array([[2]])
T1 = np.array([[35]])


model.Maschinenparameter = {'h1': h1, 'h2': h2, 'T1': T1}
model.Führungsgrößen = {'U1': lambda param,k: param['h1']+(param['h2']-param['h1'])/(1+exp(-2*(k-param['T1'])))}



# Define Models for Injection and Pressure Phase

# Construct a CasADi function for the ODE right-hand side
x = MX.sym('x',1) # states: pos_x [m], pos_y [m], vel_x [m/s], vel_y [m/s]
u = MX.sym('u',1) # control force [N]
a1 = MX.sym('a1',1)
a2 = MX.sym('a2',1)

rhs1 = a1*x + u
rhs2 = a2*x + u

# Discrete system dynamics as a CasADi Function
model.ModelInject= Function('ModelInject', [x,u,a1], [rhs1],['x','u','a1'],['rhs1'])
model.ModelPress= Function('ModelPress', [x,u,a2], [rhs2],['x','u','a2'],['rhs2'])

model.ModelParamsInject = {'a1':np.array([[0.9]])}
model.ModelParamsPress = {'a2':np.array([[0.9]])}

model.NumStates = 1

''' Everything from here on needs to run in a loop'''


''' Gather Data from Last Shot '''
N = 100;
x = np.zeros((N,1))
u = np.random.normal(0,1,(N-1,1))

for i in range(1,100):
    x[i] = 0.8*x[i-1] + u[i-1]

# u = [u,None]
# x = [x,None]
    
''' Reestimate Parameters if need be '''
values = UpdateModelParams(model,u,x,'inject')



''' Decide somehow if old values should be overwritten by new values'''


''' Solve Optimal Control Problem '''

# Ermittle erforderlichen Prozessgrößenverlauf um geforderte Bauteilqualität zu erreichen
# Ergebnis ist reference dictionary mit Referenztrajektorien und Umschaltpunkt

# Gebe Prozessgrößenverlauf als Referenztrajektorie vor und ermittle erforderliche Maschinenparameter
N=60

reference = {}
reference['Umschaltpunkt'] = 40
reference['data'] = sin(np.linspace(0,3/2*np.pi,N))

# values = MultiStageOptimization(model,reference)



''' Ein Postprocessing bei dem die ermittelten Parameter, die damit verbundenen 
Kosten und der Verlauf über die Zeit angezeigt wird ?'''






 


