#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:25:16 2020

@author: alexander
"""

from casadi import *
import matplotlib.pyplot as plt
import numpy as np




class Arburg320C():
    """
    GRUCell in Reihe mit einem Linearen Zustandsraummodell
    
    """

    def __init__(self):
        
        self.Maschinenparameter = {}
        self.Führungsgrößen = {}
        
        self.ModelParamEinspritz = {}
        self.ModelParamNachdruck = {}


    def Einspritzphase(self,x,u):
        x_new = 0.9*x+u                                                         # replace with actual model
        return x_new
        
    def Nachdruckphase(self,x,u):
        x_new = 0.9*x+u                                                         # replace with actual model
        return x_new




Maschine = Arburg320C()

Maschine.Führungsgrößen = {'U1': lambda h1,h2,k: h1+(h2-h1)*tanh(k+T1)}


N1 = 40
N2 = 20
N=N1+N2

x_ref = horzcat(sin(np.linspace(0,3/2*np.pi,N+1)))




F1 = lambda x,u: 0.9*x+u
F2 = lambda x,u: 0.9*x+u
## 1.1

opti = casadi.Opti()

X = opti.variable(N+1)
U = opti.variable(N)

h1 = opti.variable(1)
h2 = opti.variable(1)
T1 = opti.variable(1)

heaviside = lambda h1,h2,k: h1+(h2-h1)*tanh(k+T1)


for k in range(N):
    
    if k<=N1:
        # opti.subject_to(F1(X[k],U[k])==X[k+1])
        opti.subject_to(Maschine.Einspritzphase(X[k],U[k])==X[k+1])
        
    else:
        # opti.subject_to(F2(X[k],U[k])==X[k+1])
        opti.subject_to(Maschine.Nachdruckphase(X[k],U[k])==X[k+1])


# for k in range(N):
    
#     if k<=N1:
#         opti.subject_to(F1(X[k],heaviside(h1,h2,k))==X[k+1])
#     else:
#         opti.subject_to(F2(X[k],heaviside(h1,h2,k))==X[k+1])



opti.subject_to(X[0]==0)

heaviside = h1+(h2-h1)*tanh(np.linspace(0,N,N)+T1)
opti.subject_to(U==heaviside)

# opti.subject_to(X[1:-1]>=0)
opti.subject_to(X[-1]==-1)

opti.minimize(sumsqr(X-x_ref))

# opti.set_initial(T1, 10)


opti.solver('ipopt')
sol = opti.solve()

Usol_MS = sol.value(U) # should be [-2.7038;-0.5430;0.2613;0.5840]
Xsol_MS = sol.value(X) # should be [-2.7038;-0.5430;0.2613;0.5840]
print(Usol_MS)

opti_MS = opti
U_MS = U
sol_MS = sol

plt.figure()
plt.plot(x_ref)
plt.plot(Xsol_MS)
plt.plot(Usol_MS)

