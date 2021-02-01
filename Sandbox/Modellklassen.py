# -*- coding: utf-8 -*-

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
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

        

def MLP(dim_u,dim_x,dim_hidden,function_name):
    ''' Generates a Casadi Function of an MLP with hidden tanh-Layer'''
    
    
    u = cs.MX.sym('u',dim_u,1)
    x = cs.MX.sym('x',dim_x,1)
    
    W_h = cs.MX.sym('W_h',dim_hidden,dim_u+dim_x)
    b_h = cs.MX.sym('b_h',dim_hidden,1)
    
    W_o = cs.MX.sym('W_out',dim_x,dim_hidden)
    b_o = cs.MX.sym('b_out',dim_x,1)
    
    h =  cs.tanh(cs.mtimes(W_h,cs.vertcat(u,x))+b_h)
    x_new = cs.mtimes(W_o,h)+b_o
    
    input = [x,u,W_h,b_h,W_o,b_o]
    input_names = ['x','u','W_h','b_h','W_o','b_o']
    
    output = [x_new]
    output_names = ['x_new']
    
    F = cs.Function(function_name, input, output, input_names,output_names)
    
    return F

def GRU(dim_u,dim_x,dim_hidden,function_name):
    ''' Generates a Casadi Function of a GRU '''
    
    
    u = cs.MX.sym('u',dim_u,1)
    x = cs.MX.sym('x',dim_x,1)
    
    W_h = cs.MX.sym('W_h',dim_hidden,dim_u+dim_x)
    b_h = cs.MX.sym('b_h',dim_hidden,1)
    
    W_o = cs.MX.sym('W_out',dim_x,dim_hidden)
    b_o = cs.MX.sym('b_out',dim_x,1)
    
    h =  cs.tanh(cs.mtimes(W_h,cs.vertcat(u,x))+b_h)
    x_new = cs.mtimes(W_o,h)+b_o
    
    input = [x,u,W_h,b_h,W_o,b_o]
    input_names = ['x','u','W_h','b_h','W_o','b_o']
    
    output = [x_new]
    output_names = ['x_new']
    
    F = cs.Function(function_name, input, output, input_names,output_names)
    
    return F
    
    
    
