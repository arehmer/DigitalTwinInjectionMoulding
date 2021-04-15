# -*- coding: utf-8 -*-
from sys import path
#path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

from models import injection_models, NN
from optim import control_optim
# from miscellaneous import *


''' Load identified models '''
ProcessModel = Model.InjectionMouldingMachine()

results_press = pkl.load(open('results_press','rb'))
results_inject = pkl.load(open('results_inject','rb'))

"""
PressurePhaseModel = Model.MLP(dim_u=2,dim_x=1,dim_hidden=8,name='PressurePhaseModel')
PressurePhaseModel.Parameters = results_press.loc[8,'model_params'].values[0][0]

InjectionPhaseModel = Model.MLP(dim_u=2,dim_x=1,dim_hidden=8,name='InjectionPhaseModel')
InjectionPhaseModel.Parameters = results_inject.loc[8,'model_params'].values[0][0]


ProcessModel.ModelInject = InjectionPhaseModel
ProcessModel.ModelPress = PressurePhaseModel

''' Parameterize reference trajectories '''
model = Model.InjectionMouldingMachine()
partmodel = Model.Part()

N=60

h1 = np.array([[1]])
h2 = np.array([[2]])
T1 = np.array([[35]])
h3 = np.array([[4]])
h4 = np.array([[8]])
T2 = np.array([[10]])

w1 = np.array([[3]])
w2 = np.array([[5]])
H1 = np.array([[6]])
w3 = np.array([[2]])
w4 = np.array([[50]])
H2 = np.array([[5]])


ProcessModel.RefTrajectoryParams = {'h1': h1, 'h2': h2, 'T1': T1,'h3': h3, 
                                    'h4': h4, 'T2': T2, 'w1': w1, 'w2': w2, 
                                    'H1': H1,'w3': w3, 'w4': w4, 'H2': H2}
ProcessModel.RefTrajectoryInject = {'U1': lambda param,k: param['h1']+(param['h2']-param['h1'])/(1+exp(-2*(k-param['T1']))),
                        'U2': lambda param,k: param['h3']+(param['h4']-param['h3'])/(1+exp(-2*(k-param['T2'])))}

ProcessModel.RefTrajectoryPress = {'U1': lambda param,k: param['w1']+(param['w2']-param['w1'])/(1+exp(-2*(k-param['H1']))),
                        'U2': lambda param,k: param['w3']+(param['w4']-param['w3'])/(1+exp(-2*(k-param['H2'])))}


ProcessModel.NumStates = 2


""" Model of Part """

# x = MX.sym('x',1) 
# y = MX.sym('y',1)
# u = MX.sym('u',1) 

# a3 = MX.sym('a3',1)
# c3 = MX.sym('c3',1)

# rhs3 = a3*x + u
# out3 = c3*rhs3

# partmodel.ModelQuality = Function('ModelQuality', [x,u,a3,c3], [rhs3,out3],
#                                   ['x','u','a3','c3'],['rhs3','out3'])

# partmodel.ModelParamsQuality = {'a3':np.array([[0.9]]),'c3':np.array([[2]])}

# U = SingleStageOptimization(partmodel,2,100)


''' Everything from here on needs to run in a loop'''


''' Reestimate Part Quality Model if need be '''




''' Gather Data from Last Shot '''
N = 100;
x = np.zeros((N,1))
u = np.random.normal(0,1,(N-1,1))

for i in range(1,100):
    x[i] = 0.1*x[i-1] + u[i-1]

# u = [u,None]
# x = [x,None]
    
''' Reestimate Parameters if need be '''
# values = UpdateModelParams(model.ModelInject,u,x,model.ModelParamsInject)

' Decide somehow if old values should be overwritten by new values'''


''' Solve Optimal Control Problem '''

# Ermittle erforderlichen Prozessgrößenverlauf um geforderte Bauteilqualität zu erreichen
# Ergebnis ist reference dictionary mit Referenztrajektorien und Umschaltpunkt

# Gebe Prozessgrößenverlauf als Referenztrajektorie vor und ermittle erforderliche Maschinenparameter
N=60

reference = {}
reference['Umschaltpunkt'] = 40
reference['data'] = sin(np.linspace(0,3/2*np.pi,N))

values = MultiStageOptimization(ProcessModel,reference)



''' Ein Postprocessing bei dem die ermittelten Parameter, die damit verbundenen 
Kosten und der Verlauf über die Zeit angezeigt wird ?'''






 


