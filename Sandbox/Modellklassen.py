# -*- coding: utf-8 -*-

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

   


class Arburg320C():
    """
    GRUCell in Reihe mit einem Linearen Zustandsraummodell
    
    """

    def __init__(self):
        
        self.NumStates = None
        
        self.Maschinenparameter = {}
        self.MaschinenparameterConstr = {}
        
        self.Führungsgrößen = {}
        
        self.ModelParamEinspritz = {}
        self.ModelParamNachdruck = {}

        self.opti_params = None
        
        self.ModelEinspritzphase = None
        self.ModelNachdruckphase = None
        
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

