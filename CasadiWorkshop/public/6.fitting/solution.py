from pylab import *
from casadi import *
N = 100

D = np.loadtxt('data.mat')


p_hat = vertcat(1.43,0.99,0.22,0.022,0.022,0.011)

a = p_hat[0]
b = p_hat[1]
alpha = p_hat[2]
beta = p_hat[3]
gamma = p_hat[4]
delta = p_hat[5]

## 1.1


x = 10
y = 0.1

for i in range(10):
    x_next = (a*x-alpha*x*y)/(1+gamma*x)
    y_next = (b*y+beta*x*y)/(1+delta*y)

    x = x_next
    y = y_next

print(x,y) # Should be [13.1929 1.28294]


## 1.2

e = 0

x = 10
y = 0.1

for i in range(10):
    e = e + (x-D[i,0])**2+(y-D[i,1])**2
    
    x_next = (a*x-alpha*x*y)/(1+gamma*x)
    y_next = (b*y+beta*x*y)/(1+delta*y)

    x = x_next
    y = y_next

e = e + (x-D[10,0])**2+(y-D[10,1])**2

print(e) # Should be 56.893

## 1.3

N = 100

opti = Opti()

a = opti.variable()
b = opti.variable()
alpha = opti.variable()
beta = opti.variable()

gamma = opti.variable()
delta = opti.variable()

p = vertcat(a,b,alpha,beta,gamma,delta)

x = 10
y = 0.1

f = 0

for i in range(N):
    f = f + (x-D[i,0])**2+(y-D[i,1])**2
    x_next = (a*x-alpha*x*y)/(1+gamma*x)
    y_next = (b*y+beta*x*y)/(1+delta*y)

    x = x_next
    y = y_next

f = f + (x-D[100,0])**2+(y-D[100,1])**2

opti.minimize(f)

opti.solver('ipopt')

opti.set_initial(p,p_hat)

sol = opti.solve()

print(sol.value(p)) #   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]

## 2.1


opti = Opti()

a = opti.variable()
b = opti.variable()
alpha = opti.variable()
beta = opti.variable()

gamma = opti.variable()
delta = opti.variable()

p = vertcat(a,b,alpha,beta,gamma,delta)

x = 10
y = 0.1

F = []

for i in range(N):
    F.append(vertcat(x-D[i,0],y-D[i,1]))
    x_next = (a*x-alpha*x*y)/(1+gamma*x)
    y_next = (b*y+beta*x*y)/(1+delta*y)

    x = x_next
    y = y_next

F.append(vertcat(x-D[100,0],y-D[100,1]))

F = vcat(F)

opti.minimize(F.T @ F)

opti.solver('ipopt')

opti.set_initial(p,p_hat)

sol = opti.solve()

print(sol.value(p)) #   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]

## 2.2

opti = Opti()

a = opti.variable()
b = opti.variable()
alpha = opti.variable()
beta = opti.variable()

gamma = opti.variable()
delta = opti.variable()

p = vertcat(a,b,alpha,beta,gamma,delta)

x = 10
y = 0.1

F = []

for i in range(N):
    F.append(vertcat(x-D[i,0],y-D[i,1]))
    x_next = (a*x-alpha*x*y)/(1+gamma*x)
    y_next = (b*y+beta*x*y)/(1+delta*y)

    x = x_next
    y = y_next

F.append(vertcat(x-D[100,0],y-D[100,1]))

F = vcat(F)

opti.minimize(F.T @ F)

J = jacobian(F,p)

H = 2*J.T @ J # Factor two to be consistent with missing 0.5 in objective.

sigma = MX.sym('sigma')
opts = dict()
opts["hess_lag"] = Function('hess_lag',[opti.x,opti.p,sigma,opti.lam_g], [sigma*triu(H)])
opti.solver('ipopt',opts)

opti.set_initial(p,p_hat)

sol = opti.solve()

print(sol.value(p)) #   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]

## 2.3



opti = Opti()

a = opti.variable()
b = opti.variable()
alpha = opti.variable()
beta = opti.variable()
gamma = opti.variable()
delta = opti.variable()

p = vertcat(a,b,alpha,beta,gamma,delta)

x = 10
y = 0.1

S = []

for i in range(N):
    S.append(horzcat(x,y))
    x_next = (a*x-alpha*x*y)/(1+gamma*x)
    y_next = (b*y+beta*x*y)/(1+delta*y)

    x = x_next
    y = y_next

S.append(horzcat(x,y))

S = vcat(S)

S = Function('S',[p],[S])

F = vec(S(p)-D)

opti.minimize(F.T @ F)

opti.set_initial(p,p_hat)

opti.solver('ipopt')
sol = opti.solve()

print(sol.value(p)) #   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]



## 2.4

D_modified = D

D_modified[9,0] = 1
F = vec(S(p)-D_modified)

opti.minimize(F.T @ F)
sol = opti.solve()

p_fit = sol.value(p)

D_fit = S(p_fit)

figure()
plot(D_fit)
plot(D_modified)

# That's quite a bad fit around the first bump

## 2.5

opti = Opti()
p = opti.variable(6,1)
L = opti.variable(2*N+2)
F = vec(S(p)-D_modified)

opti.minimize(sum1(L))
opti.subject_to(opti.bounded(-L,F,L))

opti.solver('ipopt')

opti.set_initial(p,p_hat)

sol = opti.solve()

p_fit = sol.value(p)
print(p_fit)  #   [1.3; 0.9; 0.2; 0.02; 0.02; 0.01]

D_fit = S(p_fit)

figure()
plot(D_fit)
plot(D_modified)


show()

