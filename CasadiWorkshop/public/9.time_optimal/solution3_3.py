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


##
# -----------------------------------
#    Discrete system x_next = F(x,u)
# -----------------------------------

T = MX.sym('T')

opts = dict()
opts["tf"] = 1
intg = integrator('intg','cvodes',{'x':x,'p':vertcat(u,T),'ode':T*rhs},opts)

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

T = opti.variable()

# Gap-closing shooting constraints
for k in range(N):
  res = intg(x0=X[:,k],p=vertcat(U[:,k],T/N))
  opti.subject_to(X[:,k+1]==res["xf"])

# Path constraints
opti.subject_to(opti.bounded(-3,X[0,:],3)) # pos_x limits
opti.subject_to(opti.bounded(-3,X[1,:],3)) # pos_y limits
opti.subject_to(opti.bounded(-3,X[2,:],3)) # vel_x limits
opti.subject_to(opti.bounded(-3,X[3,:],3)) # vel_y limits
opti.subject_to(opti.bounded(-10,U[0,:],10)) # force_x limits
opti.subject_to(opti.bounded(-10,U[1,:],10)) # force_x limits

# Initial constraints
opti.subject_to(X[:,0]==vertcat(ref[:,0],0,0))

opti.subject_to(X[:2,-1]==vertcat(sin(2),cos(2)));

# Try to follow the waypoints
opti.minimize(10*T+sumsqr(X[:2,:]-ref))

# Time is bounded
opti.subject_to(opti.bounded(0.5,T,2))
opti.set_initial(T, 1)

opti.solver('ipopt')

sol = opti.solve()


print(sol.value(T)) # a=1: 0.96184 | a=10: 0.62242  

##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

xsol = sol.value(X)
usol = sol.value(U)

tgrid = linspace(0,sol.value(T),N+1)

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
