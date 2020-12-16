# -*- coding: utf-8 -*-

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

from miscellaneous import *


class InjectionMouldingMachine():
    """
    
    
    """

    def __init__(self):
        
        self.NumStates = None
        
        self.Maschinenparameter = {}
        
        self.Führungsgrößen = {}
        
        self.ModelParamsInject = {}
        self.ModelParamsPress = {}

        self.ModelInject = None
        self.ModelPress = None
        
    def ControlInput(self,opti_vars,k):
        # Übersetzt Führungsgrößen in optimierbare control inputs
        
        control = []
                
        for key in self.Führungsgrößen.keys():
            control.append(self.Führungsgrößen[key](opti_vars,k))
        
        control = vcat(control)

        return control
    


class PartQuality():
    """
    
    
    """

    def __init__(self):
        
        self.NumStates = None
       
        self.Maschinenparameter = {}
        self.Prozessgrößen = {}
        
        self.Model = None
        self.ModelParams = {}    

        




    
    
    
