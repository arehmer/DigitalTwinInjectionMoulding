# -*- coding: utf-8 -*-
from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

from Modellklassen import Arburg320C
from OptimizationTools import MultiStageOptimization
from miscellaneous import *

Maschine = Arburg320C()

N=60

h1 = 1
h2 = 2
T1 = 35
p1 = 10

Maschine.Maschinenparameter = {'h1': h1, 'h2': h2, 'T1': T1}#, 'p1':10}
# Maschine.Führungsgrößen = {'U1': lambda h1,h2,k: h1+h2*tanh(2*(k+T1))}
Maschine.Führungsgrößen = {'U1': lambda param,k: param['h1']+(param['h2']-param['h1'])/(1+exp(-2*(k-param['T1'])))}
# Maschine.MaschinenparameterConstr = {'h1':}


Maschine.NumStates = 1


''' Everything from here on needs to run in a loop'''


''' Gather Data from Last Shot '''

''' Reestimate Parameters if need be '''


''' Solve Optimal Control Problem '''

# Ermittle erforderlichen Prozessgrößenverlauf um geforderte Bauteilqualität zu erreichen
# Ergebnis ist reference dictionary mit Referenztrajektorien und Umschaltpunkt

# Gebe Prozessgrößenverlauf als Referenztrajektorie vor und ermittle erforderliche Maschinenparameter
reference = {}
reference['Umschaltpunkt'] = 40
reference['N'] = 60
reference['data'] = sin(np.linspace(0,3/2*np.pi,N))

values = MultiStageOptimization(Maschine,reference)



''' Ein Postprocessing bei dem die ermittelten Parameter, die damit verbundenen 
Kosten und der Verlauf über die Zeit angezeigt wird ?'''






 


