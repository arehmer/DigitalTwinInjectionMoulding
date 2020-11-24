from pylab import *
from casadi import *

# Physical constants

N = 20 # Number of control intervals

##
# ----------------------------------
#    continuous system dot(x)=f(x,u)
# ----------------------------------
nx = 4

# Construct a CasADi function for the ODE right-hand side
x = MX.sym('x',nx) # states: pos_x [m], pos_y [m], vel_x [m/s], vel_y [m/s]
u = MX.sym('u',2) # control force [N]
rhs = vertcat(x[2:4],u)

# Continuous system dynamics as a CasADi Function
f = Function('f', [x, u], [rhs])


##
# -----------------------------------
#    Discrete system x_next = F(x,u)
# -----------------------------------
dt = MX.sym('dt') # length of 1 control interval [s]

k1 = f(x, u)
k2 = f(x + dt/2 * k1, u)
k3 = f(x + dt/2 * k2, u)
k4 = f(x + dt * k3, u)
xf = x+dt/6*(k1 +2*k2 +2*k3 +k4)

F = Function('F', [x, u, dt], [xf])

##
# ------------------------------------------------
# Waypoints
# ------------------------------------------------

ref = horzcat(sin(np.linspace(0,2,N+1)),cos(np.linspace(0,2,N+1))).T

##
# -----------------------------------------------
#    Optimal control problem, multiple shooting
# -----------------------------------------------

opti = casadi.Opti()

# Decision variables for states
X = opti.variable(nx,N+1)
# Decision variables for control vector
U =  opti.variable(2,N) # force [N]

# Time
T = opti.variable()

# Gap-closing shooting constraints
for k in range(N):
  opti.subject_to(X[:,k+1]==F(X[:,k],U[:,k],T/N))

# Path constraints
opti.subject_to(opti.bounded(-3,X[0,:],3)) # pos_x limits
opti.subject_to(opti.bounded(-3,X[1,:],3)) # pos_y limits
opti.subject_to(opti.bounded(-3,X[2,:],3)) # vel_x limits
opti.subject_to(opti.bounded(-3,X[3,:],3)) # vel_y limits
opti.subject_to(opti.bounded(-10,U[0,:],10)) # force_x limits
opti.subject_to(opti.bounded(-10,U[1,:],10)) # force_x limits

# Initial constraints
opti.subject_to(X[:,0]==vertcat(0,0,0,0))

opti.subject_to(X[:2,-1]==vertcat(2,1))

# Obstacle avoidance
p = vertcat(1,0.75)
r = 0.6

opti.subject_to(sum1((X[:2,:]-p)**2)>=r**2)

# Time is bounded
opti.subject_to(opti.bounded(0.5,T,2))
opti.set_initial(T, 1)

opti.set_initial(X[0,:],np.linspace(0,2,N+1)) # Leads to 8.4867232e-01
#opti.set_initial(X[1,:],np.linspace(0,2,N+1)) # Leads to 1.0437847e+00

# Try to follow the waypoints
opti.minimize(T)

opti.solver('ipopt')

sol = opti.solve()

##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

xsol = sol.value(X)
usol = sol.value(U)

tgrid = np.linspace(0,sol.value(T),N+1)

figure()
plot(xsol[0,:].T,xsol[1,:].T,'bs-','linewidth',2)
t = np.linspace(0,2*pi,1000)
plot(r*sin(t)+p[0],r*cos(t)+p[1])
legend(('OCP trajectory','obstacle'))
title('Top view')
axis('equal')

xlabel('x')
xlabel('y')
figure()
step(tgrid,horzcat(usol,usol[:,-1]).T)
title('applied control signal')
legend(('force_x','force_y'))
ylabel('Force [N]')
xlabel('Time [s]')
show()
