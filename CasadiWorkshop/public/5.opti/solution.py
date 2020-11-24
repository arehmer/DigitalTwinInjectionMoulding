from pylab import *
from casadi import *

N = 25
m = 40/float(N) # mass [kg]
D = 70*N        # spring constant [J/m^2]
g = 9.81        # gravitational constant [m/s^2]
L = 5/float(N)  # reference length [m]

## 1.1

opti = Opti()

x = opti.variable(N)
y = opti.variable(N)

V = 0.5*D*sum1(diff(x)**2+diff(y)**2)
V = V + g*sum1(m*y)

opti.minimize(V)
opti.subject_to(vertcat(x[0],y[0]) == vertcat(-2,0))
opti.subject_to(vertcat(x[-1],y[-1]) == vertcat(2,0))
opti.solver('ipopt')

sol = opti.solve()

plot(sol.value(x),sol.value(y),'-o')


## 1.2
hessian(opti.f,opti.x)[0].sparsity().spy()

# We observe a band structure: there is only coupling between immediately neighbouring points in the objective.
# We also note that there is no coupling between x and y.

## 1.3

lag = opti.f+opti.lam_g.T @ opti.g

grad_lag = sol.value(gradient(lag,opti.x))

print(norm(grad_lag))

## 2.1
opti = casadi.Opti()

x = opti.variable(N)
y = opti.variable(N)

V = 0.5*D*sum1((sqrt(diff(x)**2+diff(y)**2)-L)**2)
V = V + g*sum1(m*y)

opti.minimize(V)
opti.subject_to(vertcat(x[0],y[0]) == vertcat(-2,0))
opti.subject_to(vertcat(x[-1],y[-1]) == vertcat(2,0))
opti.solver('ipopt')

try:
  sol = opti.solve() # raises error
except Exception as e:
  print(e)



## 2.1

opti = casadi.Opti()

x = opti.variable(N)
y = opti.variable(N)

V = 0.5*D*sum1((sqrt(diff(x)**2+diff(y)**2)-L)**2)
V = V + g*sum1(m*y)

opti.minimize(V)
opti.subject_to(vertcat(x[0],y[0]) == vertcat(-2,0))
opti.subject_to(vertcat(x[-1],y[-1]) == vertcat(2,0))
opti.solver('ipopt')

opti.set_initial(x,np.linspace(-2,2,N)) # needed to avoid NaN

sol = opti.solve()

figure()
plot(sol.value(x),sol.value(y),'-o')
axis('equal')


## 2.2
hessian(opti.f,opti.x)[0].sparsity().spy()

opti = casadi.Opti()

z = opti.variable(2*N,1)
x = z[::2]
y = z[1::2]

V = 0.5*D*sum1((sqrt(diff(x)**2+diff(y)**2)-L)**2)
V = V + g*sum1(m*y)

opti.minimize(V)
opti.subject_to(vertcat(x[0],y[0]) == vertcat(-2,0))
opti.subject_to(vertcat(x[-1],y[-1]) == vertcat(2,0))
opti.solver('ipopt')

opti.set_initial(x,np.linspace(-2,2,N)) # needed to avoid NaN

sol = opti.solve()

figure()
plot(sol.value(x),sol.value(y),'-o')
axis('equal')

hessian(opti.f,opti.x)[0].sparsity().spy()

## 2.3

opti = casadi.Opti()

x = opti.variable(N)
y = opti.variable(N)

V = 0.5*D*sum1((sqrt(diff(x)**2+diff(y)**2)-L)**2)
V = V + g*sum1(m*y)

opti.minimize(V)
opti.subject_to(vertcat(x[0],y[0]) == vertcat(-2,0))
opti.subject_to(vertcat(x[-1],y[-1]) == vertcat(2,0))
opti.solver('ipopt')

from plot_iteration import plot_iteration

opti.callback(lambda i: plot_iteration(i,opti.debug.value(x),opti.debug.value(y)))

opti.set_initial(x,np.linspace(-2,2,N)) # needed to avoid NaN

sol = opti.solve()

figure()
plot(sol.value(x),sol.value(y),'-o')
axis('equal')


## 3.1
opti = casadi.Opti()

x = opti.variable(N)
y = opti.variable(N)

opti.minimize(g*sum1(m*y))

opti.subject_to(diff(x)**2+diff(y)**2==L**2)
opti.subject_to(vertcat(x[0],y[0]) == vertcat(-2,0))
opti.subject_to(vertcat(x[-1],y[-1]) == vertcat(2,0))

opti.set_initial(x,np.linspace(-2,2,N))
opti.set_initial(y,-sin(np.linspace(0,pi,N)))

opti.solver('ipopt')

sol = opti.solve()

J = jacobian(opti.g,opti.x)

print(opti.debug.value(J))
print(np.linalg.matrix_rank(opti.debug.value(J).toarray()))

figure()
plot(sol.value(x),sol.value(y),'-o')
axis('equal')
## 3.2

opti = casadi.Opti()

x = opti.variable(N)
y = opti.variable(N)

opti.minimize(g*sum1(m*y))

opti.subject_to(diff(x)**2+diff(y)**2==L**2)
opti.subject_to(vertcat(x[0],y[0]) == vertcat(-2,0))
opti.subject_to(vertcat(x[-1],y[-1]) == vertcat(2,0))

opti.set_initial(x,np.linspace(-2,2,N))

opti.solver('ipopt')

try:
  sol = opti.solve(); # Raises error
except Exception as e:
  print(e)


J = jacobian(opti.g,opti.x)
print(J.shape)
print(np.linalg.matrix_rank(opti.debug.value(J).toarray())) # Check LICQ: rank is 27 in stead of 28.

show()

