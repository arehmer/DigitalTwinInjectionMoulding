import casadi.*

% Declare model variables
x1 = MX.sym('x1');
x2 = MX.sym('x2');
x = vertcat(x1,x2);

% Time dependence
t = MX.sym('t');

t0 = 2;
dt = 0.1;
tf = t0+dt;
x0 = vertcat(1,0.5);

% ODE right-hand side
rhs = [(1-x2^2)*x1 - x2+t; x1];

f = Function('f',{t,x},{rhs},{'t','x'},{'rhs'});

d = 4;
tau = collocation_points(d,'legendre');
t_coll = t0+tau*dt;

n = size(x,1);
Xc = MX.sym('Xc',n,d);
X0 = MX.sym('X0',n);

Pi_expr = LagrangePolynomialEval([t0,t_coll],[X0,Xc],t);

Pi = Function('Pi',{t,X0,Xc},{Pi_expr},{'t','X0','Xc'},{'Pi'});

dot_Pi = Function('dot_Pi',{t,X0,Xc},{jacobian(Pi_expr,t)},{'t','X0','Xc'},{'dPi'});

dot_Pi = dot_Pi.expand();

Xc_SX = SX.sym('Xc_SX',n,d);
X0_SX = SX.sym('X0_SX',n);

evalf(jacobian(dot_Pi(t_coll,X0_SX,Xc_SX),[X0_SX,Xc_SX]))