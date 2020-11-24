from pylab import *
from recurse import recurse
from recurse_dummy import recurse_dummy
from casadi import *

# Physical constants

T = 1.0 # control horizon [s]
N = 6 # Number of control intervals

dt = T/N # length of 1 control interval [s]

tgrid = np.linspace(0,T,N+1)

# The perturbation of delta
delta_num = 1

##
# ----------------------------------
#    continuous system dot(x)=f(x,u)
# ----------------------------------
nx = 2

# Construct a CasADi function for the ODE right-hand side
x1 = MX.sym('x1')
x2 = MX.sym('x2')
u = MX.sym('u') # control
delta = MX.sym('delta')
rhs = vertcat(x2,-0.1*(1-x1**2+delta)*x2 - x1 + u)
x = vertcat(x1,x2)

x1_bound = lambda t: 2+0.1*cos(10*t)

##
# -----------------------------------
#    Discrete system x_next = F(x,u)
# -----------------------------------

opts = dict()
opts["tf"] = dt
intg = integrator('intg','cvodes',{'x':x,'p':vertcat(u,delta),'ode':rhs},opts)

## Traversing all branches

# The core of the algorithm will be a recursion that enumerates all possible branches/series of events

# Our implementation prodcues a cell with each entry corresponding to a unique sequence of events
H = recurse_dummy(delta_num,N,[])
print(vcat(H))

# The actual 'recurse' function will return a cell of state variable sequences
