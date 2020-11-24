from casadi import *

from LagrangePolynomialEval import LagrangePolynomialEval

DM.set_precision(13)

## 1
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
print(f)
print(f(t0,x0)) # Should be [2.25; 1]

## 2

d = 4


tau = DM(collocation_points(d,'legendre')).T

## 3

X = vertcat(0,0.5,1)
Y = vertcat(7,1,3).T
xs = np.linspace(0,1)

ys = LagrangePolynomialEval(X,Y,xs)

from pylab import figure, plot, show # requires matplotlib
figure()
plot(X,Y.T,'ro')
plot(xs,ys)

## 4

t_coll = t0+tau*dt

## 5

# Coefficients of polynomial
n = x.shape[0]
Xc = MX.sym('Xc',n,d)
X0 = MX.sym('X0',n)

Pi_expr = LagrangePolynomialEval(horzcat(t0,t_coll),horzcat(X0,Xc),t)

Pi = Function('Pi',[t,X0,Xc],[Pi_expr],['t','X0','Xc'],['Pi'])
print(Pi(t0+0.05,x0,horzcat(x0+1,x0+2,x0+4,x0+5))) # should be [3.6501; 3.1501]

## 6
dot_Pi = Function('dot_Pi',[t,X0,Xc],[jacobian(Pi_expr,t)],['t','X0','Xc'],['dPi'])

print(dot_Pi(t0+0.05,x0,horzcat(x0+1,x0+2,x0+4,x0+5))) # should be [61.1122; 61.1122]

## 7
# Construct collocation equations
g = []
for j in range(d):
  g.append( dot_Pi(t_coll[j],X0,Xc)-f(t_coll[j],Xc[:,j]) )

g = vcat(g)
# Equivalent:  g = vec(dot_Pi(t_coll,X0,Xc)-f(t_coll,Xc))


gf = Function('g',[X0,Xc],[g])
print(gf(x0,horzcat(x0+1,x0+2,x0+4,x0+5))) # should be [107.130;103.137;35.726;16.509;187.686;84.003;108.022;-76.884]

## 8
solver = rootfinder('solver','newton',{'x':vec(Xc),'p':X0,'g':g})

res = solver(x0=repmat(x0,d,1),p=x0)
print(Pi(tf,x0,reshape(res["x"],n,d))) # should be [1.226454824197, 0.6113162319035]

## 9

ode = {'x':x,'t':t,'ode':rhs}
options = dict()
options["t0"] = t0
options["tf"] = tf
options["number_of_finite_elements"] = 1
options["interpolation_order"] = 4
options["collocation_scheme"] = 'legendre'
intg = integrator('intg','collocation',ode,options)
res=intg(x0=x0)
print(res["xf"]) # should be [1.226454824197, 0.6113162319035]

show()
