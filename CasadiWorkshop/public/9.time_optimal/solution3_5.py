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

ref = lambda s: vertcat(sin(2*s),cos(2*s))

##
# -----------------------------------------------
#    Optimal control problem, multiple shooting
# -----------------------------------------------

opti = casadi.Opti()

# Decision variables for states
X = opti.variable(nx,N+1)
# Decision variables for control vector
U =  opti.variable(2,N) # force [N]

S = opti.variable(1,N+1)

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
opti.subject_to(X[:,0]==vertcat(0,1,0,0))

opti.subject_to(X[:2,-1]==vertcat(sin(2),cos(2)))

opti.subject_to(sum1((X[:2,:]-ref(S))**2).T<=0.2**2)

opti.set_initial(S, linspace(0,1,N+1))

# Try to follow the waypoints
opti.minimize(T);

# Time is bounded
opti.subject_to(opti.bounded(0.5,T,2))
opti.set_initial(T, 1)

opti.solver('ipopt')

sol = opti.solve()


print(sol.value(T)) # 0.66866

##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

xsol = sol.value(X)
usol = sol.value(U)

tgrid = linspace(0,sol.value(T),N+1)

figure()
plot(xsol[0,:].T,xsol[1,:].T,'bs-','linewidth',2)

R = ref(DM(np.linspace(0,1,100)).T)
plot(R[0,:].T,R[1,:].T,'r','linewidth',3)
circle_x = 0.2*cos(np.linspace(0,2*pi))
circle_y = 0.2*sin(np.linspace(0,2*pi))
for i in range(100):
  plot(R[0,i]+circle_x,R[1,i]+circle_y,'r.','linewidth',3)

legend(('OCP trajectory','Reference trajecory'))
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
