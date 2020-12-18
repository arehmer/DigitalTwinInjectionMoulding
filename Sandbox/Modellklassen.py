# -*- coding: utf-8 -*-

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

from miscellaneous import *


class InjectionMouldingMachine():
    """
    Modell der Spritzgießmaschine, welches Führungsgrößen (parametriert durch 
    an der Maschine einstellbare Größen) auf die resultierenden Prozessgrößen
    abbildet.    
    """

    def __init__(self):
        
        self.NumStates = None
        
        self.Maschinenparameter = {}
        self.Führungsgrößen = {}
        
        self.ModelParamsInject = {}
        self.ModelParamsPress = {}
        self.ModelParamsCool = {}   

        self.ModelInject = None
        self.ModelPress = None
        self.ModelCool = None
        
    def ControlInput(self,opti_vars,k):
        """
        Übersetzt durch Maschinenparameter parametrierte
        Führungsgrößenverläufe in optimierbare control inputs
        """
        
        control = []
                
        for key in self.Führungsgrößen.keys():
            control.append(self.Führungsgrößen[key](opti_vars,k))
        
        control = vcat(control)

        return control
    


class Part():
    """
    Modell des Bauteils, welches die einwirkenden Prozessgrößen auf die 
    resultierenden Bauteilqualität abbildet.    
    """

    def __init__(self):
        
        self.NumStates = None
       
        self.ModelQuality = None
        self.ModelParamsQuality = {}    

        




    
    
    
