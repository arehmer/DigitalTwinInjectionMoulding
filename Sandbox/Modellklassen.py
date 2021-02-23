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
        
        self.Fuehrungsparameter = {}
        self.Führungsgrößen = {}
        
        self.RefTrajectoryParams = None
        # self.RefParamsPress = None
        # self.RefParamsCool = None
        
        self.RefTrajectoryInject = None
        self.RefTrajectoryPress = None
        self.RefTrajectoryCool = None 

        self.ModelInject = None
        self.ModelPress = None
        self.ModelCool = None
        
    # def ControlInput(self,opti_vars,k):
    #     """
    #     Übersetzt durch Maschinenparameter parametrierte
    #     Führungsgrößenverläufe in optimierbare control inputs
    #     """
        
    #     control = []
                
    #     for key in self.Führungsgrößen.keys():
    #         control.append(self.Führungsgrößen[key](opti_vars,k))
        
    #     control = cs.vcat(control)

    #     return control
    


class Part():
    """
    Modell des Bauteils, welches die einwirkenden Prozessgrößen auf die 
    resultierenden Bauteilqualität abbildet.    
    """

    def __init__(self):
        
        self.NumStates = None
       
        self.ModelQuality = None
        self.ModelParamsQuality = {}    

class LinearSSM():
    """
    
    """

    def __init__(self,dim_u,dim_x,dim_y,name):
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.name = name
        
        self.Initialize()

    def Initialize(self):
            
            # For convenience of notation
            dim_u = self.dim_u
            dim_x = self.dim_x 
            dim_y = self.dim_y             
            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',dim_u,1)
            x = cs.MX.sym('x',dim_x,1)
            y = cs.MX.sym('y',dim_y,1)
            
            # Define Model Parameters
            A = cs.MX.sym('A',dim_x,dim_x)
            B = cs.MX.sym('B',dim_x,dim_u)
            C = cs.MX.sym('C',dim_y,dim_x)

            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = {'A':np.random.rand(dim_x,dim_x),
                               'B':np.random.rand(dim_x,dim_u),
                               'C':np.random.rand(dim_y,dim_x)}
        
            # self.Input = {'u':np.random.rand(u.shape)}
            
            # Define Model Equations
            x_new = cs.mtimes(A,x) + cs.mtimes(B,u)
            y_new = cs.mtimes(C,x_new) 
            
            
            input = [x,u,A,B,C]
            input_names = ['x','u','A','B','C']
            
            output = [x_new,y_new]
            output_names = ['x_new','y_new']  
            
            self.Function = cs.Function(name, input, output, input_names,output_names)
            
            return None
   
    def OneStepPrediction(self,x0,u0,params=None):
        '''
        Estimates the next state and output from current state and input
        x0: Casadi MX, current state
        u0: Casadi MX, current input
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''
        
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x1,y1 = self.Function(x0,u0,*params_new)     
                              
        return x1,y1
   
    def Simulation(self,x0,u,params=None):
        '''
        A iterative application of the OneStepPrediction in order to perform a
        simulation for a whole input trajectory
        x0: Casadi MX, inital state a begin of simulation
        u: Casadi MX,  input trajectory
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''

        x = []
        y = []

        # initial states
        x.append(x0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x_new,y_new = self.OneStepPrediction(x[k],u[[k],:],params)
            x.append(x_new)
            y.append(y_new)
        

        # Concatenate list to casadiMX
        y = cs.hcat(y).T    
        x = cs.hcat(x).T
       
        return y


class MLP():
    """
    
    """

    def __init__(self,dim_u,dim_x,dim_hidden,name):
        
        self.dim_u = dim_u
        self.dim_hidden = dim_hidden
        self.dim_x = dim_x
        self.name = name
        
        self.Initialize()

    def Initialize(self):
                
            dim_u = self.dim_u
            dim_hidden = self.dim_hidden
            dim_x = self.dim_x 
            name = self.name
        
            u = cs.MX.sym('u',dim_u,1)
            x = cs.MX.sym('x',dim_x,1)
            
            # Parameters
            W_h = cs.MX.sym('W_h',dim_hidden,dim_u+dim_x)
            b_h = cs.MX.sym('b_h',dim_hidden,1)
            
            W_o = cs.MX.sym('W_out',dim_x,dim_hidden)
            b_o = cs.MX.sym('b_out',dim_x,1)
            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = {'W_h':np.random.rand(W_h.shape[0],W_h.shape[1]),
                               'b_h':np.random.rand(b_h.shape[0],b_h.shape[1]),
                               'W_o':np.random.rand(W_o.shape[0],W_o.shape[1]),
                               'b_o':np.random.rand(b_o.shape[0],b_o.shape[1])}
        
            # self.Input = {'u':np.random.rand(u.shape)}
            
            # Equations
            h =  cs.tanh(cs.mtimes(W_h,cs.vertcat(u,x))+b_h)
            x_new = cs.mtimes(W_o,h)+b_o
            
            
            input = [x,u,W_h,b_h,W_o,b_o]
            input_names = ['x','u','W_h','b_h','W_o','b_o']
            
            output = [x_new]
            output_names = ['x_new']  
            
            self.Function = cs.Function(name, input, output, input_names,output_names)
            
            return None
   
    def OneStepPrediction(self,x0,u0,params=None):
        # Casadi Function needs list of parameters as input
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x1 = self.Function(x0,u0,*params_new)     
                              
        return x1
   
    def Simulation(self,x0,u,params=None):
        # Casadi Function needs list of parameters as input
        
        x = []

        # initial states
        x.append(x0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x.append(self.OneStepPrediction(x[k],u[[k],:],params))
        
        # Concatenate list to casadiMX
        x = cs.vcat(x) 
       
        return x


    
def logistic(x):
    
    y = 0.5 + 0.5 * cs.tanh(0.5*x)

    return y

class GRU():
    """
    Modell des Bauteils, welches die einwirkenden Prozessgrößen auf die 
    resultierenden Bauteilqualität abbildet.    
    """

    def __init__(self,dim_u,dim_c,dim_hidden,dim_out,name):
        
        self.dim_u = dim_u
        self.dim_c = dim_c
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.name = name
        
        self.Initialize()  
 

    def Initialize(self):
        
        dim_u = self.dim_u
        dim_c = self.dim_c
        dim_hidden = self.dim_hidden
        dim_out = self.dim_out
        name = self.name      
        
        u = cs.MX.sym('u',dim_u,1)
        c = cs.MX.sym('c',dim_c,1)
        
        # Parameters
        # RNN part
        W_r = cs.MX.sym('W_r',dim_c,dim_u+dim_c)
        b_r = cs.MX.sym('b_r',dim_c,1)
    
        W_z = cs.MX.sym('W_z',dim_c,dim_u+dim_c)
        b_z = cs.MX.sym('b_z',dim_c,1)    
        
        W_c = cs.MX.sym('W_c',dim_c,dim_u+dim_c)
        b_c = cs.MX.sym('b_c',dim_c,1)    
    
        # MLP part
        W_h = cs.MX.sym('W_z',dim_hidden,dim_c)
        b_h = cs.MX.sym('b_z',dim_hidden,1)    
        
        W_o = cs.MX.sym('W_c',dim_out,dim_hidden)
        b_o = cs.MX.sym('b_c',dim_out,1)  
        
        # Put all Parameters in Dictionary with random initialization
        self.Parameters = {'W_r':np.random.rand(W_r.shape[0],W_r.shape[1]),
                           'b_r':np.random.rand(b_r.shape[0],b_r.shape[1]),
                           'W_z':np.random.rand(W_z.shape[0],W_z.shape[1]),
                           'b_z':np.random.rand(b_z.shape[0],b_z.shape[1]),
                           'W_c':np.random.rand(W_c.shape[0],W_c.shape[1]),
                           'b_c':np.random.rand(b_c.shape[0],b_c.shape[1]),                          
                           'W_h':np.random.rand(W_h.shape[0],W_h.shape[1]),
                           'b_h':np.random.rand(b_h.shape[0],b_h.shape[1]),                           
                           'W_o':np.random.rand(W_o.shape[0],W_o.shape[1]),
                           'b_o':np.random.rand(b_o.shape[0],b_o.shape[1])}
        
        # Equations
        f_r = logistic(cs.mtimes(W_r,cs.vertcat(u,c))+b_r)
        f_z = logistic(cs.mtimes(W_z,cs.vertcat(u,c))+b_z)
        
        c_r = f_r*c
        
        f_c = cs.tanh(cs.mtimes(W_c,cs.vertcat(u,c_r))+b_c)
        
        
        c_new = f_z*c+(1-f_z)*f_c
        
        h =  cs.tanh(cs.mtimes(W_h,c_new)+b_h)
        x_new = cs.mtimes(W_o,h)+b_o    
    
        
        # Casadi Function
        input = [c,u,W_r,b_r,W_z,b_z,W_c,b_c,W_h,b_h,W_o,b_o]
        input_names = ['c','u','W_r','b_r','W_z','b_z','W_c','b_c','W_h','b_h',
                        'W_o','b_o']
        
        output = [c_new,x_new]
        output_names = ['c_new','x_new']
    
        self.Function = cs.Function(name, input, output, input_names,output_names)

        return None
    
    def OneStepPrediction(self,c0,u0,params=None):
        # Casadi Function needs list of parameters as input
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        c1,x1 = self.Function(c0,u0,*params_new)     
                              
        return c1,x1
   
    def Simulation(self,c0,u,params=None):
        # Casadi Function needs list of parameters as input
        
        # print('GRU Simulation ignores given initial state, initial state is set to zero!')
        c0 = np.zeros((self.dim_c,1))
        
        c = []
        x = []
        
        # initial cell state
        c.append(c0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            c_new,x_new = self.OneStepPrediction(c[k],u[k,:],params)
            c.append(c_new)
            x.append(x_new)
        
        # Concatenate list to casadiMX
        c = cs.hcat(c).T    
        x = cs.hcat(x).T
        
        return x[-1]