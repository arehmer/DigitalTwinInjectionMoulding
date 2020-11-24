from pylab import *
from casadi import *

m = 0.04593 # [kg]
g = 9.81    # [m/s^2]
c = 17.5e-5 # friction [kg/m]

## 1.1

DM.set_precision(6)

p = MX.sym('p',2)
v = MX.sym('v',2)

states = vertcat(p,v)

speed = norm_2(v)

rhs = vertcat(v,-c*speed*v/m)
rhs[3] = rhs[3] - g

f = Function('rhs',[vertcat(p,v)],[rhs])
print(f([0,0,35,30])) # should be [35, 30, -6.14737, -15.0792]


## 1.2

ode = {'x':vertcat(p,v),'ode':rhs}

options = dict()
options["tf"] = 1

intg = integrator('intg','cvodes',ode,options)
print(intg)
res = intg(x0=[0,0,35,30])

print("xf = ", res["xf"])

## 2.1

X0 = MX.sym('X0',4)
res = intg(x0=X0)

fly1sec = Function('fly1sec',[X0],[res["xf"]])

print(fly1sec([0,0,35,20])) # should be [32.6405, 13.9607, 30.5608, 8.26437]

## 2.2

T = MX.sym('T')
ode_T = {'x':vertcat(p,v),'ode':T*rhs,'p':T}
intg = integrator('intg','cvodes',ode_T,{'tf':1})
res = intg(x0=X0,p=T)

fly = Function('fly',[X0,T],[res["xf"]])
print(fly([0,0,35,30],5)) # should be [130.338, 8.27205, 19.8961, -21.2868]


## 2.3
theta = MX.sym('theta')
v     = MX.sym('v')

theta_rad = theta/180*pi

res = fly(vertcat(0,0,v*cos(theta_rad),v*sin(theta_rad)),T)
shoot = Function('shoot',[v,theta,T],[res])

print(shoot(50,30,5)) # should be [155.243, -11.0833, 22.6012, -23.8282]

## 2.4
x = shoot(50,30,T)
height = x[1]

rf = rootfinder('rf','newton',{'x':T,'g':height})
res = rf(x0=5)

print(res["x"]) # should be 4.49773

## 2.5
x = shoot(v,theta,T)
height = x[1]

rf = rootfinder('rf','newton',{'x':T,'p':vertcat(v,theta),'g':height})
res = rf(x0=5,p=vertcat(v,theta))

T_landing = res["x"]
xf = shoot(v,theta,T_landing)

shoot_distance = Function('shoot_distance',[v,theta],[xf[0]])

print(shoot_distance(50,30)) # should be 143.533

## 3.1

nlp = dict()
nlp["x"] = theta
nlp["f"] = -shoot_distance(30,theta)

solver = nlpsol('solver','ipopt',nlp)

res = solver(x0=30)
print(res["x"]) # should be 43.2223


# Note: the convergence is not so clean due to tolerances in the integrator process
# You might increase integrator tolerance with e.g. 'abstol' 1e-14, 'reltol' 1e-14

# Not 45 dues to friction; set c=0 and you will obtain 45 degree as optimum

## 3.2
cov_vtheta = diag([1**2,1.2**2])

d = shoot_distance(v,theta)

J = jacobian(d,vertcat(v,theta))

sigma_shoot_distance = J @ cov_vtheta @ J.T

nlp = dict()
nlp["x"] = vertcat(v,theta)
nlp["g"] = d
nlp["f"] = sigma_shoot_distance

solver = nlpsol('solver','ipopt',nlp)

res = solver(x0=vertcat(30,45),lbg=80,ubg=80)
print(res["x"]) # should be [34.1584, 57.0871]



