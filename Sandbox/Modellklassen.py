# -*- coding: utf-8 -*-

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

from miscellaneous import *


class Arburg320C():
    """
    GRUCell in Reihe mit einem Linearen Zustandsraummodell
    
    """

    def __init__(self):
        
        self.NumStates = None
        
        self.Maschinenparameter = {}
        self.MaschinenparameterConstr = {}
        
        self.Führungsgrößen = {}
        
        self.ModelParamsInject = {}
        self.ModelParamsPress = {}

        self.opti_vars = None
        
        self.ModelInject = None
        self.ModelPress = None
        
    def SimulateInject(self,x,u):
        # Modell Einspritzphase
        
        # Casadi Function needs list of parameters as input
        params = []
        
        for name in  self.ModelInject.name_in():
            try:
                params.append(self.ModelParamsInject[name])                     # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x_new = self.ModelInject(x,u,*params)                                  
        
        return x_new
        
    def SimulatePress(self,x,u):
        # Modell Nachdruckphase
        
        # Casadi Function needs list of parameters as input
        params = []
        
        for name in  self.ModelInject.name_in():
            try:
                params.append(self.ModelParamsInject[name])                     # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x_new = self.ModelPress(x,u,*params)     
                              
        return x_new

    def ControlInput(self,opti,k):
        # Übersetzt Führungsgrößen in optimierbare control inputs
        
        control = []
                
        for key in self.Führungsgrößen.keys():
            control.append(self.Führungsgrößen[key](self.opti_vars,k))
        
        control = vcat(control)

        return control
        
    def CreateOptimVariables(self,opti, param_dict):
        
        Parameter = {}
    
        for key in param_dict.keys():
            
            dim0 = param_dict[key].shape[0]
            dim1 = param_dict[key].shape[1]
            
            Parameter[key] = opti.variable(dim0,dim1)
    
        self.opti_vars = Parameter
        
        return None
    
        
        




    
    
    
