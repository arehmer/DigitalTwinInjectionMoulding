#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:25:16 2020

@author: alexander
"""

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')


    


class Arburg320C():
    """
    GRUCell in Reihe mit einem Linearen Zustandsraummodell
    
    """

    def __init__(self):
        
        self.Maschinenparameter = {}
        self.Führungsgrößen = {}
        
        self.ModelParamEinspritz = {}
        self.ModelParamNachdruck = {}

        self.opti_params = None
        
    def Einspritzphase(self,x,u):
        # Modell Einspritzphase
        x_new = 0.9*x+u                                                         # replace with actual model
        return x_new
        
    def Nachdruckphase(self,x,u):
        # Modell Nachdruckphase
        x_new = 0.9*x+u                                                         # replace with actual model
        return x_new

    def ControlInput(self,opti,k):
        # Übersetzt Führungsgrößen in optimierbare control inputs
        
        control = []
                
        for key in self.Führungsgrößen.keys():
            control.append(self.Führungsgrößen[key](self.opti_params,k))
        
        control = vcat(control)

        return control
        
    def CreateOptimVariables(self,opti, param_dict):
        
        Parameter = {}
    
        for key in param_dict.keys():
            Parameter[key] = opti.variable(1)
    
        self.opti_params = Parameter
        
        return None



Maschine = Arburg320C()

h1 = 1
h2 = 2
T1 = 35
p1 = 10


Maschine.Maschinenparameter = {'h1': h1, 'h2': h2, 'T1': T1, 'p1':10}
# Maschine.Führungsgrößen = {'U1': lambda h1,h2,k: h1+h2*tanh(2*(k+T1))}
Maschine.Führungsgrößen = {'U1': lambda param,k: param['h1']+(param['h2']-param['h1'])/(1+exp(-2*(k-param['T1'])))}

                           # 'U2': lambda param,k: param['p1']}




N1 = 40
N2 = 20
N=N1+N2

x_ref = horzcat(sin(np.linspace(0,3/2*np.pi,N+1)))

# F1 = lambda x,u: 0.9*x+u
# F2 = lambda x,u: 0.9*x+u
## 1.1

opti = casadi.Opti()


X = opti.variable(N+1)
# U = opti.variable(N)

# logistic = lambda L,k,x0,x: L/(1+exp(-k*(x-x0)))
# heaviside = lambda h1,h2,T1,k: h1+logistic(h2-h1,2,T1,k)

# k=0

# test = Maschine.ControlInput(Maschinenparameter,k)

# opti.subject_to(Maschine.Einspritzphase(X[k],Maschine.ControlInput(opti,k))==X[k+1])

# Create opti variables of the parameters one wants to optimize
Maschine.CreateOptimVariables(opti, Maschine.Maschinenparameter)

# for k in range(N):
    
#     if k<=N1:
#         # opti.subject_to(F1(X[k],U[k])==X[k+1])
#         opti.subject_to(Maschine.Einspritzphase(X[k],U[k])==X[k+1])
        
#     else:
#         # opti.subject_to(F2(X[k],U[k])==X[k+1])
#         opti.subject_to(Maschine.Nachdruckphase(X[k],U[k])==X[k+1])


for k in range(N):
    
    if k<=N1:
        opti.subject_to(Maschine.Einspritzphase(X[k],Maschine.ControlInput(opti,k))==X[k+1])
    else:
        opti.subject_to(Maschine.Nachdruckphase(X[k],Maschine.ControlInput(opti,k))==X[k+1])



opti.subject_to(X[0]==0.5)

# logistic = lambda L,k,x0,x: L/(1+exp(-k*(x-x0)))

# heaviside = h1+logistic(h2-h1,2,T1,np.linspace(0,N,N))
# opti.subject_to(U==heaviside)

# opti.subject_to(X[1:-1]>=0)
opti.subject_to(X[-1]==-1)

opti.minimize(sumsqr(X-x_ref))

# opti.set_initial(T1, 10)


opti.solver('ipopt')
sol = opti.solve()

# Usol_MS = sol.value(U) # should be [-2.7038;-0.5430;0.2613;0.5840]
Xsol_MS = sol.value(X) # should be [-2.7038;-0.5430;0.2613;0.5840]
# print(Usol_MS)

opti_MS = opti
# U_MS = U
# sol_MS = sol

plt.figure()
plt.plot(x_ref)
plt.plot(Xsol_MS)
# plt.plot(Usol_MS)

