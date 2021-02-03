#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:25:16 2020

@author: alexander
"""

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

# Import sphere function as objective function
#from pyswarms.utils.functions.single_obj import sphere as f

# Import backend modules
# import pyswarms.backend as P
# from pyswarms.backend.topology import Star
# from pyswarms.discrete.binary import BinaryPSO

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


from miscellaneous import *


def SimulateModel(model,x,u,params=None):
    # Casadi Function needs list of parameters as input
    if params==None:
        params = model.Parameters
    
    params_new = []
        
    for name in  model.Function.name_in():
        try:
            params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
        except:
            continue
    
    x_new = model.Function(x,u,*params_new)     
                          
    return x_new
   
    
def CreateOptimVariables(opti, param_dict):
    
    Parameter = {}

    for key in param_dict.keys():
        
        dim0 = param_dict[key].shape[0]
        dim1 = param_dict[key].shape[1]
        
        Parameter[key] = opti.variable(dim0,dim1)

    opti_vars = Parameter
    
    return opti_vars

def MultiStageOptimization(model,ref):
    #Multi Stage Optimization for solving the optimal control problem
    
 
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    # Translate Maschinenparameter into opti.variables
    Fuehrungsparameter_opti = CreateOptimVariables(opti, model.Parameters)
    
    # Number of time steps
    N = ref['data'].shape[0]
    
    # Create decision variables for states
    NumStates = model.ModelInject.dim_x
    X = opti.variable(N,NumStates)
        
    # Initial Constraints
    opti.subject_to(X[0]==ref['data'][0])
    
    
    # System Dynamics as Path Constraints
    for k in range(N-1):
        
        if k<=ref['Umschaltpunkt']:
            U = model.ControlInput(Fuehrungsparameter_opti,k)
            opti.subject_to(SimulateModel(model.ModelInject,X[k],U)==X[k+1])

        elif k>ref['Umschaltpunkt']:
            U = model.ControlInput(Fuehrungsparameter_opti,k)
            opti.subject_to(SimulateModel(model.ModelPress,X[k],U)==X[k+1])

        else:
             U=None # HIER MUSS EIN MODELL FÜR DIE ABKÜHLPHASE HIN
    
    ''' Further Path Constraints (to avoid values that might damage the machine or in 
    other ways harmful or unrealistic) '''
    
    # TO DO #
    
    
    # Final constraint
    opti.subject_to(X[-1]==ref['data'][-1])
    
    
    # Set initial values for Machine Parameters
    for key in Fuehrungsparameter_opti:
        opti.set_initial(Fuehrungsparameter_opti[key],model.Fuehrungsparameter[key])

    # Set initial values for state trajectory ??
    # for key in model.Maschinenparameter_opti:
    #     opti.set_initial(model.Maschinenparameter_opti[key],CurrentParams[key])      
    
    # Define Loss Function    
    opti.minimize(sumsqr(X-ref['data']))
    
    #Choose solver
    opti.solver('ipopt')
    
    # Get solution
    sol = opti.solve()
    
    # Extract real values from solution
    values = OptimValues_to_dict(Fuehrungsparameter_opti,sol)
    values['X'] = sol.value(X)

    
    return values



def ModelTraining(model,data,initializations = 20, BFR=False, p_opts=None, s_opts=None):
    
    # Split in Training and Validation data
    
    results = [] 
    
    for i in range(0,initializations):
        print(i)
        # In first iteration, initialize with current model parameters (useful for online training)
        # if(i > 0):
            # continue
        
        # Estimate Parameters on training data
        new_params = ModelParameterEstimation(model,data)
        
        # Assign new parameters to model
        model.Parameters = new_params
        
        # Evaluate on Validation data
        u_val = data['u_val']
        x_ref_val = data['x_val']
        init_state_val = data['init_state_val']

        # Loop over all experiments
        
        e = 0
        
        for j in range(0,u_val.shape[0]):   
               
            # Simulate Model
            x = model.Simulation(init_state_val[j],u_val[j])
            x = np.array(x)
                     
            e = e + cs.sumsqr(x_ref_val[j] - x) 
        
        # Calculate mean error over all validation batches
        e = e / u_val.shape[0]
        
        # save parameters and performance in list
        results.append([e,new_params])
        
    
    return results 

def ModelParameterEstimation(model,data,p_opts=None,s_opts=None):
    """
    Schätzt Parameter des Maschinenmodell nach, muss für das Teilequalitätsmodell
    noch erweitert werden
    """
    
    
    u = data['u_train']
    x_ref = data['x_train']
    init_state = data['init_state_train']
    
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    params_opti = CreateOptimVariables(opti, model.Parameters)
    
    e = 0
    
    # Loop over all experiments
    for i in range(0,u.shape[0]):   
           
        # Simulate Model
        x = model.Simulation(init_state[i],u[i],params_opti)
        
        e = e + sumsqr(x_ref[i,:,:] - x)
    
    opti.minimize(e)
    
    # Solver options
    if p_opts is None:
        p_opts = {"expand":False}
    if s_opts is None:
        s_opts = {"max_iter": 1000, "print_level":0}
    
    # Create Solver
    opti.solver("ipopt",p_opts, s_opts)
    
    
    # Set initial values of Opti Variables as current Model Parameters
    for key in params_opti:
        opti.set_initial(params_opti[key], model.Parameters[key])
    
    
    # Solve NLP, if solver does not converge, use last solution from opti.debug
    try: 
        sol = opti.solve()
    except:
        sol = opti.debug
        
    values = OptimValues_to_dict(params_opti,sol)
    
    return values

def SingleStageOptimization(model,ref,N):
    """ 
    single shooting procedure for optimal control of a scalar final value
    
    model: Quality Model
    ref: skalarer Referenzwert für Optimierungsproblem
    N: Anzahl an Zeitschritten
    """
    
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    # Create decision variables for states
    U = opti.variable(N,1)
        
    # Initial quality 
    x = 0
    y = 0
    X = [x]
    Y = [y]
    
    # Simulate Model
    for k in range(N):
        out = SimulateModel(model.ModelQuality,X[k],U[k],model.ModelParamsQuality)
        X.append(out[0])
        Y.append(out[1])
            
    X = hcat(X)
    Y = hcat(Y)
    
    # Define Loss Function  
    opti.minimize(sumsqr(Y[-1]-ref))
                  
    #Choose solver
    opti.solver('ipopt')
    
    # Get solution
    sol = opti.solve()   

    # Extract real values from solution
    values = {}
    values['U'] = sol.value(U)
    
    return values

# def ParticleSwarmOptimization(quality_model,ref,bounds,u0):
#     """
    
#     Parameters
#     ----------
#     quality_model : Instanz der Part-Klasse
#         Dynamisches Modell der Bauteilqualität.
#     ref : float
#         Referenz-Bauteilqualität
#     bounds : list of integers
#         bounds[0]/bounds[1] ist untere/obere Grenze der Zykluszeit in 
#         diskreten Zeitschritten
#     u0 : list?
#         Initialisierung der Prozessgrößentrajektorien

#     Returns
#     -------
#     u_opt: Dictionary?
#         Optimierte Prozessgrößenverläufe

#     """

#     if bounds = None:
#         pass
#         """    
#         Dann keine Partikelschwarmoptimierung über die Zeit sondern direkt die 
#         Prozessgrößentrajektorie optimieren
#         """  
    
#     else:
#         """
#         Partikelschwarmoptimierung über die Zeit
#         """
        
#         """
#         Kostenfunktion f muss noch implementiert werden. Diese muss folgendes
#         leisten:
#             Rechne diskrete Zeitschritte in Binärzahl um, und zwar so, dass die 
#             definierten bounds für die Zeit eingehalten werden
#             Loop über SingleStageOptimization, da eine parallele Auswertung 
#             hier nicht möglich ist
        
#         """
        
        
#         n_particles = 100
#         dimensions = 10000000 # ergibt sich aus den bounds!
#         options = {'c1':1, 'c2':1, 'w':1, 'k':10, 'p':1} 
        
#         # Initialisiere Optimizer
#         SwarmOptimizer = BinaryPSO(n_particles, dimensions, options, 
#                                    init_pos=None, velocity_clamp=None, 
#                                    vh_strategy='unmodified', ftol=-inf, 
#                                    ftol_iter=1)
        

        
        
#         # Perform optimization
#         cost, pos = SwarmOptimizer.optimize(f, iters=1000)
    
    
    
    
    
#     return u_opt
    
# def ParticleSwarmCostFunction()    
    
    
#     my_topology = Star() # The Topology Class
#     my_options = {'c1': 0.6, 'c2': 0.3, 'w': 0.4} # arbitrarily set
#     my_swarm = P.create_swarm(n_particles=50, dimensions=2, options=my_options) # The Swarm Class
    
#     print('The following are the attributes of our swarm: {}'.format(my_swarm.__dict__.keys()))    

#     iterations = 100 # Set 100 iterations
#     for i in range(iterations):
#         # Part 1: Update personal best
#         my_swarm.current_cost = f(my_swarm.position) # Compute current cost
#         my_swarm.pbest_cost = f(my_swarm.pbest_pos)  # Compute personal best pos
#         my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm) # Update and store
    
#         # Part 2: Update global best
#         # Note that gbest computation is dependent on your topology
#         if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
#             my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)
    
#         # Let's print our output
#         if i%20==0:
#             print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i+1, my_swarm.best_cost))
    
#         # Part 3: Update position and velocity matrices
#         # Note that position and velocity updates are dependent on your topology
#         my_swarm.velocity = my_topology.compute_velocity(my_swarm)
#         my_swarm.position = my_topology.compute_position(my_swarm)
    
#     print('The best cost found by our swarm is: {:.4f}'.format(my_swarm.best_cost))
#     print('The best position found by our swarm is: {}'.format(my_swarm.best_pos))

    
#     SingleStageOptimization(model,ref,N)








