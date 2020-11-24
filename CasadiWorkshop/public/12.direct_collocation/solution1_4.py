from casadi import *

from LagrangePolynomialEval import LagrangePolynomialEval

# Declare model variables
x1 = MX.sym('x1')
x2 = MX.sym('x2')
x = vertcat(x1,x2)

# Time dependence
t = MX.sym('t')

t0 = 2
dt = 0.1
tf = t0+dt
x0 = vertcat(1,0.5)

# ODE right-hand side
rhs = vertcat((1-x2**2)*x1 - x2+t, x1)

f = Function('f',[t,x],[rhs],['t','x'],['rhs'])

d = 4
tau = DM(collocation_points(d,'legendre')).T
t_coll = t0+tau*dt

n = x.shape[0]
Xc = MX.sym('Xc',n,d)
X0 = MX.sym('X0',n)

Pi_expr = LagrangePolynomialEval(horzcat(t0,t_coll),horzcat(X0,Xc),t)

Pi = Function('Pi',[t,X0,Xc],[Pi_expr],['t','X0','Xc'],['Pi'])

dot_Pi = Function('dot_Pi',[t,X0,Xc],[jacobian(Pi_expr,t)],['t','X0','Xc'],['dPi'])

Xc_SX = SX.sym('Xc_SX',n,d)
X0_SX = SX.sym('X0_SX',n)

print(jacobian(dot_Pi(t_coll,X0_SX,Xc_SX),horzcat(X0_SX,Xc_SX)))