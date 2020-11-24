from casadi import *

def g(w):
  x = w[0]
  y = w[1]
  g1 = tanh((x+2)*y**2/25-0.5)
  g2 = sin(x)-0.5*y + 1
  return vertcat(g1,g2)


## 1.1

res = g([-0.8,2]) # Should be   [-0.2986,-0.7174]
print(res.shape)
print(res)


## 1.2

x = vertcat(-0.8,2)

nominal = g(x)
epsilon = 1e-8
perturbed_x = (g(x+epsilon*vertcat(1,0))-nominal)/epsilon
perturbed_y = (g(x+epsilon*vertcat(0,1))-nominal)/epsilon

J = horzcat(perturbed_x,perturbed_y) # Should be  [0.1457    0.1749; 0.6967   -0.5000]

print(J)



## 1.3

x = vertcat(-0.8,2)

for i in range(5):
    nominal = g(x)
    epsilon = 1e-8
    perturbed_x = (g(x+epsilon*vertcat(1,0))-nominal)/epsilon
    perturbed_y = (g(x+epsilon*vertcat(0,1))-nominal)/epsilon

    J = horzcat(perturbed_x,perturbed_y)

    x = x - solve(J,nominal) # newton step


print(x) # Should be [0.1945;2.3866]



## 2.1

X = MX.sym('x',2)
print(type(g(X)))
print(g(X).shape)

print(g(X)) # evaluate g symbolically

# @1=x[0], @2=x[1], vertcat(tanh(((((2+@1)*sq(@2))/25)-0.5)), (1+(sin(@1)-(0.5*@2))))
# You see tanh and sin applied to expressions involving components of the symbol x


J = jacobian(g(X),X) # compute Jacobian

print(type(J))
print(J.shape)

## 2.2
# Construct a CasADi function that computes the Jacobian
Jf = Function('Jf',[X],[J])

print(Jf)
# Jf:(i0[2])->(o0[2x2]) MXFunction
#        \->            [2] means a 2-vector
#                 \->   [2x2] means a 2-by-2 matrix


## 2.3

print(Jf([-0.8,2]))

## 2.4
x = vertcat(-0.8,2)

for i in range(5):
    x = x - solve(Jf(x),g(x))

print(x)


## 3.1

rf =  rootfinder('rf','newton',{'x':X,'g':g(X)})
print(rf)

## 3.2
print(rf([-0.8,2],[]))

## 3.3
print(rf.stats()) # 6 iterations

## 3.4

options = {'print_iteration': True}
rf = rootfinder('rf','newton',{'x':X,'g':g(X)},options)
rf([-0.8,2],[])

