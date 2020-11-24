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

dt = MX.sym('dt')
xf = x+dt*f(x,u)
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
T =  opti.variable(N+1) # Time [s]

# Gap-closing shooting constraints
for k in range(N):
  opti.subject_to(X[:,k+1]==F(X[:,k],U[:,k],T[k]/N))
  opti.subject_to(T[k]==T[k+1])

# Path constraints
opti.subject_to(opti.bounded(-3,X[0,:],3)) # pos_x limits
opti.subject_to(opti.bounded(-3,X[1,:],3)) # pos_y limits
opti.subject_to(opti.bounded(-3,X[2,:],3)) # vel_x limits
opti.subject_to(opti.bounded(-3,X[3,:],3)) # vel_y limits
opti.subject_to(opti.bounded(-10,U[0,:],10)) # force_x limits
opti.subject_to(opti.bounded(-10,U[1,:],10)) # force_x limits

# Initial constraints
opti.subject_to(X[:,0]==vertcat(ref[:,0],0,0))

# Time is bounded
opti.subject_to(opti.bounded(0.5,T,2))
opti.set_initial(T, 1)


# Try to follow the waypoints
opti.minimize(sumsqr(X[:2,:]-ref))

opti.solver('sqpmethod',{"convexify_strategy": "eigen-clip"})

sol = opti.solve()

# Note: in 1.3, ~5ms was spent in convexification, here only ~0.2ms
# The new structure can be transformed to block diagonal Hessian.
# Convexification can be performed block-wise

figure()
spy(sol.value(jacobian(opti.g,opti.x)))

figure()
spy(sol.value(hessian(opti.f+dot(opti.lam_g,opti.g),opti.x)[0]))


##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

xsol = sol.value(X)
usol = sol.value(U)

tgrid = linspace(0,sol.value(T[0]),N+1)

figure()
plot(xsol[0,:].T,xsol[1,:].T,'bs-','linewidth',2)
plot(ref[0,:].T,ref[1,:].T,'ro','linewidth',3)
legend(('OCP trajectory','Reference trajecory'))
title('Top view')
xlabel('x')
xlabel('y')
figure()
step(tgrid,horzcat(usol,usol[:,-1]).T)
title('applied control signal')
legend(('force_x','force_y'))
ylabel('Force [N]')
xlabel('Time [s]')
show()
