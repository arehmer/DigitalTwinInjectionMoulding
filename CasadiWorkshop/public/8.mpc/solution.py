from pylab import *
from casadi import *
import time

# Physical constants
g = 9.81    # gravitation [m/s^2]
L = 0.2     # pendulum length [m]
m = 1       # pendulum mass [kg]
mcart = 0.5 # cart mass [kg]

T = 2.0 # control horizon [s]
N = 160 # Number of control intervals

dt = T/N # length of 1 control interval [s]

# System is composed of 4 states
nx = 4


speedup = 4 # 0..4

##
# ----------------------------------
#    continuous system dot(x)=f(x,u)
# ----------------------------------

# Construct a CasADi function for the ODE right-hand side
x = MX.sym('x',nx) # states: pos [m], theta [rad], dpos [m/s], dtheta [rad/s]
u = MX.sym('u') # control force [N]
ddpos = ((u+m*L*x[3]*x[3]*sin(x[1])-m*g*sin(x[1])*cos(x[1]))/(mcart+m-m*cos(x[1])*cos(x[1])))
rhs = vertcat(x[2],x[3],ddpos,g/L*sin(x[1])-cos(x[1])*ddpos)

# Continuous system dynamics as a CasADi Function
f = Function('f', [x, u],[rhs])

##
# -----------------------------------
#    Discrete system x_next = F(x,u)
# -----------------------------------

# Integrator options
intg_options = dict()
intg_options["number_of_finite_elements"] = 1
if speedup>=3:
    intg_options["simplify"] = 1
intg_options["tf"] = dt/intg_options["number_of_finite_elements"]

# Reference Runge-Kutta implementation
intg = integrator('intg','rk',{'x':x,'p':u,'ode':f(x,u)},intg_options)
print(intg)
res = intg(x0=x,p=u)
xf = res["xf"]

# Discretized (sampling time dt) system dynamics as a CasADi Function
F = Function('F', [x, u], [xf])

##
# -----------------------------------------------
#    Optimal control problem, multiple shooting
# -----------------------------------------------

opti = casadi.Opti()

# Decision variables for states
X = opti.variable(nx,N+1)
# Aliases for states
pos    = X[0,:]
theta  = X[1,:]
dpos   = X[2,:]
dtheta = X[3,:]

# Decision variables for control vector
U =  opti.variable(N,1) # force [N]

# 1.2: Parameter for initial state
x0 = opti.parameter(nx)

# Gap-closing shooting constraints
for k in range(N):
   opti.subject_to(X[:,k+1]==F(X[:,k],U[k]))

# Path constraints
opti.subject_to(opti.bounded(-3,pos,3))
opti.subject_to(opti.bounded(-1.2,U,1.2))

# Initial and terminal constraints
opti.subject_to(X[:,0]==x0)
opti.subject_to(X[:,-1]==vertcat(0,0,0,0))

# Objective: regularization of controls
# 1.1: added regularization
opti.minimize(sumsqr(U)+1000*sumsqr(pos))

# solve optimization problem
options = dict()
options["print_time"] = False

if speedup>=3:
    options["expand"] = True # expand makes function evaluations faster but requires more memory

if speedup>=2:
    options["qpsol"] = 'qrqp';
    options["qpsol_options"] = {"print_iter": False, "print_header": False}
    options["print_iteration"] = False
    options["print_header"] = False
    options["print_status"] = False
    opti.solver('sqpmethod',options)
else:
    options["ipopt"] = {"print_level": 0}
    opti.solver('ipopt',options)


opti.set_value(x0,vertcat(0.5,0,0,0))

sol = opti.solve()
##
# -----------------------------------------------
#    MPC loop
# -----------------------------------------------

current_x = vertcat(0.5,0,0,0)

x_history = DM.zeros(nx,400)
u_history = DM.zeros(1,400)

np.random.seed(0)

disp('MPC running')

if speedup<4:
  
  for i in range(400):
      t0 = time.time()
      # What control signal should I apply?
      u_sol = sol.value(U[0])
      
      u_history[:,i] = u_sol
      
      # Simulate the system over dt
      current_x = F(current_x,u_sol)
      if i>200:
          current_x = current_x + vertcat([0,0,0,0.01*np.random.rand()])
  
      if speedup>=1:
        # Set the initial values to the previous results
        opti.set_initial(opti.x, sol.value(opti.x)) # decision variables
        opti.set_initial(opti.lam_g, sol.value(opti.lam_g)) # multipliers
    
      # Set the value of parameter x0 to the current x
      opti.set_value(x0, current_x)

      # Solve the NLP
      sol = opti.solve()
      
      x_history[:,i] = current_x
      print(time.time()-t0)

if speedup>=4:
  inputs = [x0,opti.x,opti.lam_g]
  outputs = [U[0],opti.x,opti.lam_g]
  mpc_step = opti.to_function('mpc_step',inputs,outputs)
  print(mpc_step)

  u = sol.value(U[0])
  x = sol.value(opti.x)
  lam = sol.value(opti.lam_g)
  
  for i in range(400):
      t0 = time.time()
      u_history[:,i] = u
      
      # Simulate the system over dt
      current_x = F(current_x,u)
      if i>200:
          current_x = current_x + vertcat([0,0,0,0.01*np.random.rand()])

      [u,x,lam] = mpc_step(current_x,x,lam)
      
      x_history[:,i] = current_x
      print(time.time()-t0)



##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

figure()
plot(x_history.T)
title('simulated states')
xlabel('sample')
figure()
plot(u_history.T)
title('applied control signal')
ylabel('Force [N]')
xlabel('sample')

show()

