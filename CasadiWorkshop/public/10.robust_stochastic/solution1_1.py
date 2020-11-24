from pylab import *

from casadi import *

# Physical constants

T = 1.0 # control horizon [s]
N = 40 # Number of control intervals

dt = T/N # length of 1 control interval [s]

tgrid = np.linspace(0,T,N+1)

##
# ----------------------------------
#    continuous system dot(x)=f(x,u)
# ----------------------------------
nx = 2

# Construct a CasADi function for the ODE right-hand side
x1 = MX.sym('x1')
x2 = MX.sym('x2')
u = MX.sym('u') # control

P = MX.sym('x',2,2)
rhs = vertcat(x2,-0.1*(1-x1**2)*x2 - x1 + u)
x = vertcat(x1,x2)

A = jacobian(rhs,x)
B = jacobian(rhs,u)

x = vertcat(x,vec(P))
dotP = A @ P+P @ A.T

rhs = vertcat(rhs,vec(dotP))

x1_bound = lambda t: 2+0.1*cos(10*t)

##
# -----------------------------------
#    Discrete system x_next = F(x,u)
# -----------------------------------

opts = dict()
opts["tf"] = dt
intg = integrator('intg','cvodes',{'x':x,'p':u,'ode':rhs},opts)

##
# -----------------------------------------------
#    Optimal control problem, multiple shooting
# -----------------------------------------------
x0 = vertcat(0.5,0)

opti = casadi.Opti();

# Decision variable for states
x = opti.variable(nx)
P = opti.variable(nx,nx)

# Initial constraints
opti.subject_to(x==x0)
opti.subject_to(P==diag([0.01**2,0.1**2]))

U = []
X = [x]
Ps = [P]
# Gap-closing shooting constraints
for k in range(N):
  u = opti.variable()
  U.append(u)

  x_next = opti.variable(nx)
  P_next = opti.variable(nx,nx)
  res = intg(x0=vertcat(x,vec(P)),p=u)
  
  xf = res["xf"][:nx]
  Pf = reshape(res["xf"][nx:],nx,nx)
  opti.subject_to(x_next==xf)
  opti.subject_to(vec(P_next)==vec(Pf))

  opti.subject_to(opti.bounded(-40,u,40))
  opti.subject_to(opti.bounded(-0.25,x[0],x1_bound(tgrid[k])))
  
  x = x_next
  P = P_next

  Ps.append(P)
  X.append(x)

opti.subject_to(opti.bounded(-0.25,x_next[0],x1_bound(tgrid[N])))
U = hcat(U)
X = hcat(X)

opti.minimize(sumsqr(X[0,:]-3))

opti.solver('ipopt')

sol = opti.solve()

figure()
spy(sol.value(jacobian(opti.g,opti.x)))

print(sol.value(Ps[-1]))

##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------


# Simulate forward in time using an initial state and control vector
usol = sol.value(U)
xsol = sol.value(X)

print(xsol.shape)
figure()
plot(tgrid,xsol[0,:].T,'bs-','linewidth',2)
plot(tgrid,x1_bound(tgrid),'r--','linewidth',4)

for k in range(N+1):
  var = sol.value(horzcat(1,0) @ Ps[k] @ vertcat(1,0))
  sigma = sqrt(var)
  t = tgrid[k]
  plot([t,t],[xsol[0,k]-sigma,xsol[0,k]+sigma],'k','linewidth',2)

legend(('OCP trajectory x1','bound on x1'))
xlabel('Time [s]')
ylabel('x1')
figure()
step(tgrid,vertcat(usol,usol[-1]))
title('applied control signal')
ylabel('Force [N]')
xlabel('Time [s]')

show()
