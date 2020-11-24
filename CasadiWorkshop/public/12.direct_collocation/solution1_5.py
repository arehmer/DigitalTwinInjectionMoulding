from pylab import *
from casadi import *

T = 1.0 # control horizon [s]
N = 40 # Number of control intervals

dt = T/N # length of 1 control interval [s]

##
# ----------------------------------
#    continuous system dot(x)=f(x,u)
# ----------------------------------

# Construct a CasADi function for the ODE right-hand side

A = np.array(
    [[0,0,0,0,0,0,1,0],
     [0,0,0,0,0,1,0,1],
     [0,0,0,0,1,0,0,1],
     [0,0,0,1,0,0,0,1],
     [0,0,1,1,1,1,1,1],
     [0,1,0,0,0,0,0,1],
     [1,0,0,0,0,0,0,1],
     [1,1,1,0,0,0,1,1]])
nx = A.shape[0]
B = np.array(
 [[-0.073463,-0.073463],
  [-0.146834,-0.146834],
  [-0.146834,-0.146834],
  [-0.146834,-0.146834],
  [-0.446652,-0.446652],
  [-0.147491,-0.147491],
  [-0.147491,-0.147491],
  [-0.371676,-0.371676]])

nu = B.shape[1]

x  = MX.sym('x',nx)
u  = MX.sym('u',nu)

dx = sparsify(A) @ sqrt(x)+sparsify(B) @ u

x_steady = (-solve(A,B @ vertcat(1,1)))**2


# Continuous system dynamics as a CasADi Function
f = Function('f', [x, u], [dx])

##
# -----------------------------------------------
#    Optimal control problem, multiple shooting
# -----------------------------------------------

opti = casadi.Opti()

# Decision variables for states
X = opti.variable(nx,N+1)
# Decision variables for control vector
U = opti.variable(nu,N)

degree = 3
method = 'radau'

tau = collocation_points(degree,method)
[C,D,B] = collocation_coeff(tau)

# Dynamic constraints
for k in range(N):
   # Decision variables for helper states at each collocation point
   Xc = opti.variable(nx, degree)
   Z = horzcat(X[:,k],Xc)

   Pidot = (Z @ C)/dt
   opti.subject_to(Pidot==f(Xc,U[:,k]))

   # Continuity constraints
   opti.subject_to(Z @ D==X[:,k+1])
   opti.set_initial(Xc, repmat(x_steady,1,degree))

# Path constraints
opti.subject_to(opti.bounded(0.01,vec(X),0.1))

# Initial guesses
opti.set_initial(X, repmat(x_steady,1,N+1))
opti.set_initial(U, 1)

# Initial and terminal constraints
opti.subject_to(X[:,0]==x_steady);
# Objective: regularization of controls

xbar = opti.variable();
opti.minimize(1e-6*sumsqr(U)+sumsqr(X[:,-1]-xbar))

# solve optimization problem
opti.solver('ipopt')

sol = opti.solve()


# Here also, sparsity of the dynamic system is recovered

figure()
spy(sol.value(jacobian(opti.g,opti.x)))

# Total CPU secs in NLP function evaluations is much less than in solution1_2
# Total CPU secs in IPOPT (w/o function evaluations) (mostly time spent in factorising linear systems) went up
#    ->    we need bigger sparse dynamics for the sparsity benefit to kick in
# Also increased number of iterations is atypical


##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

figure()
Xsol = sol.value(X)
plot(Xsol.T,'o-')

show()
