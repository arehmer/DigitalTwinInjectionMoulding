from pylab import *
from casadi import *

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
intg_options = dict();
intg_options["number_of_finite_elements"] = 1;
intg_options["tf"] = dt/intg_options["number_of_finite_elements"]

# Reference Runge-Kutta implementation
intg = integrator('intg','rk',{'x':x,'p':u,'ode':f(x,u)},intg_options)
res = intg(x0=x,p=u)

# Discretized (sampling time dt) system dynamics as a CasADi Function
F = Function('F', [x, u], [res["xf"]])

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

# Gap-closing shooting constraints
for k in range(N):
   opti.subject_to(X[:,k+1]==F(X[:,k],U[k]))


# Path constraints
opti.subject_to(opti.bounded(-3,  pos, 3)) # Syntax -3 <= pos <= 3 not supported in Python
opti.subject_to(opti.bounded(-1.2, U, 1.2))

# Initial and terminal constraints
opti.subject_to(X[:,0]==vertcat(1,0,0,0))
opti.subject_to(X[:,-1]==vertcat(0,0,0,0))

# Objective: regularization of controls
opti.minimize(sumsqr(U))

# solve optimization problem
opti.solver('ipopt')

sol = opti.solve()

##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

pos_opt = sol.value(pos)
theta_opt = sol.value(theta)
dpos_opt = sol.value(dpos)
dtheta_opt = sol.value(dtheta)

u_opt = sol.value(U)

# time grid for printing
tgrid = np.linspace(0,T, N+1)

figure
subplot(3,1,1)
plot(tgrid, theta_opt, 'b')
plot(tgrid, pos_opt, 'b')
legend(('theta [rad]','pos [m]'))
xlabel('Time [s]')
subplot(3,1,2)
plot(tgrid, dtheta_opt, 'b')
plot(tgrid, dpos_opt, 'b')
legend(('dtheta [rad/s]','dpos [m/s]'))
xlabel('Time [s]')
subplot(3,1,3)
step(tgrid[:-1], u_opt, 'b')
legend('u [m/s^2]')
xlabel('Time [s]')

cart = vertcat(pos,0*pos)
ee   = vertcat(pos+L*sin(theta),L*cos(theta))

cart_sol = sol.value(cart)
ee_sol   = sol.value(ee)

figure()
for k in range(0,N+1,8):
    plot([cart_sol[0,k],ee_sol[0,k]],[cart_sol[1,k],ee_sol[1,k]],'k',linewidth=1)

axis('equal')
show()
